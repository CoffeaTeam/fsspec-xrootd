"""Test basic IO against a xrootd server fixture"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import time

import fsspec
import pytest

from fsspec_xrootd.xrootd import (
    XRootDFileSystem,
    _chunks_to_vectors,
    _vectors_to_chunks,
)

XROOTD_PORT = 1094
TESTDATA1 = "apple\nbanana\norange\ngrape"
TESTDATA2 = "red\ngreen\nyellow\nblue"
sleep_time = 0.2
expiry_time = 0.1


def require_port_availability(port: int) -> bool:
    """Raise an exception if the given port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", port)) == 0:
            raise RuntimeError(f"This test requires port {port} to be available")


@pytest.fixture(scope="module")
def localserver(tmpdir_factory):
    require_port_availability(XROOTD_PORT)

    srvdir = tmpdir_factory.mktemp("srv")
    tempPath = os.path.join(srvdir, "Folder")
    os.mkdir(tempPath)
    xrdexe = shutil.which("xrootd")
    proc = subprocess.Popen([xrdexe, "-p", str(XROOTD_PORT), srvdir])
    time.sleep(2)  # give it some startup
    yield "root://localhost/" + str(tempPath), tempPath
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture()
def clear_server(localserver):
    remoteurl, localpath = localserver
    fs, _, _ = fsspec.get_fs_token_paths(remoteurl)
    # The open file handles on client side imply an open file handle on the server,
    # so removing the directory doesn't actually work until the client closes its handles!
    fs.invalidate_cache()
    shutil.rmtree(localpath)
    os.mkdir(localpath)
    yield


def test_ping(localserver, clear_server):
    remoteurl, localpath = localserver
    from XRootD import client

    fs = client.FileSystem(remoteurl)
    status, _n = fs.ping()
    if not status.ok:
        raise OSError(f"Server did not run properly: {status.message}")


def test_invalid_server():
    with pytest.raises(ValueError):
        fsspec.core.url_to_fs("root://")


def test_invalid_parameters():
    with pytest.raises(TypeError):
        fsspec.filesystem(protocol="root")


def test_async_impl():
    cls = fsspec.get_filesystem_class(protocol="root")
    assert cls == XRootDFileSystem
    assert cls.async_impl, "XRootDFileSystem should have async_impl=True"


def test_broken_server():
    with pytest.raises(OSError):
        # try to connect on the wrong port should fail
        with fsspec.open("root://localhost:12345/", "rt", timeout=5) as f:
            _ = f.read()


def test_path_parsing():
    fs, _, (path,) = fsspec.get_fs_token_paths("root://server.com")
    assert fs.protocol == "root"
    assert path == "/"
    fs, _, (path,) = fsspec.get_fs_token_paths("root://server.com/")
    assert path == "/"
    fs, _, (path,) = fsspec.get_fs_token_paths("root://server.com/blah")
    assert path == "blah"
    fs, _, (path,) = fsspec.get_fs_token_paths("root://server.com//blah")
    assert path == "/blah"
    fs, _, paths = fsspec.get_fs_token_paths(
        [
            "root://server.com//blah",
            "root://server.com//more",
            "root://server.com/dir/",
            "root://serv.er//dir/",
        ]
    )
    assert paths == ["/blah", "/more", "dir", "/dir"]


def test_pickle(localserver, clear_server):
    import pickle

    remoteurl, localpath = localserver

    fs, _, (path,) = fsspec.get_fs_token_paths(remoteurl)
    assert fs.ls(path) == []
    fs = pickle.loads(pickle.dumps(fs))
    assert fs.ls(path) == []
    time.sleep(1)


def test_read_xrd(localserver, clear_server):
    remoteurl, localpath = localserver
    with open(localpath + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)

    from XRootD import client

    with client.File() as f:
        status, _ = f.open(remoteurl + "/testfile.txt")
        if not status.ok:
            raise RuntimeError(status)
        status, res = f.read()
        if not status.ok:
            raise RuntimeError(status)
        assert res.decode("ascii") == TESTDATA1
        f.close()


def test_read_fsspec(localserver, clear_server):
    remoteurl, localpath = localserver
    with open(localpath + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)

    with fsspec.open(remoteurl + "/testfile.txt", "rt") as f:
        assert f.read() == TESTDATA1
        f.seek(0)
        assert f.readline() == "apple\n"
        f.seek(0)
        lns = f.readlines()
        assert lns[2] == "orange\n"
        f.seek(1)
        assert f.read(1) == "p"

    with fsspec.open(remoteurl + "/testfile.txt", "rb") as f:
        assert f.readuntil(b"e") == b"apple"

    fs, token, path = fsspec.get_fs_token_paths(remoteurl + "/testfile.txt", "rt")
    assert fs.read_block(path[0], 0, 4) == b"appl"


def test_write_fsspec(localserver, clear_server):
    remoteurl, localpath = localserver
    with fsspec.open(remoteurl + "/testfile.txt", "wt") as f:
        f.write(TESTDATA1)
        f.flush()
    with open(localpath + "/testfile.txt") as f:
        assert f.read() == TESTDATA1


@pytest.mark.parametrize("start, end", [(None, None), (None, 10), (1, None), (1, 10)])
def test_read_bytes_fsspec(localserver, clear_server, start, end):
    remoteurl, localpath = localserver
    with open(localpath + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)

    fs, _ = fsspec.core.url_to_fs(remoteurl)
    data = fs.read_bytes(localpath + "/testfile.txt", start=start, end=end)
    assert data == TESTDATA1.encode("utf-8")[start:end]


def test_append_fsspec(localserver, clear_server):
    remoteurl, localpath = localserver
    with open(localpath + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)
    with fsspec.open(remoteurl + "/testfile.txt", "at") as f:
        f.write(TESTDATA2)
        f.flush()
    with open(localpath + "/testfile.txt") as f:
        assert f.read() == TESTDATA1 + TESTDATA2


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_mk_and_rm_dir_fsspec(localserver, cache_expiry, clear_server):
    remoteurl, localpath = localserver
    os.makedirs(localpath + "/Folder1")
    os.makedirs(localpath + "/Folder2")
    with open(localpath + "/Folder1/testfile1.txt", "w") as fout:
        fout.write(TESTDATA2)
    with open(localpath + "/Folder2/testfile2.txt", "w") as fout:
        fout.write(TESTDATA2)
    fs, token, path = fsspec.get_fs_token_paths(
        remoteurl, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    time.sleep(sleep_time)

    assert set(fs.ls(path[0], False)) == {"Folder1", "Folder2"}
    fs.mkdir(path[0] + "/Folder3/Folder33")
    time.sleep(sleep_time)
    assert set(fs.ls(path[0], False)) == {
        "Folder1",
        "Folder2",
        "Folder3",
    }

    with pytest.raises(OSError):
        fs.mkdir(path[0] + "/Folder4/Folder44", False)
    fs.mkdirs(path[0] + "/Folder4/Folder44")
    time.sleep(sleep_time)
    assert set(fs.ls(path[0], False)) == {
        "Folder1",
        "Folder2",
        "Folder3",
        "Folder4",
    }
    fs.mkdirs(path[0] + "/Folder4", True)
    time.sleep(sleep_time)
    with pytest.raises(OSError):
        fs.mkdirs(path[0] + "/Folder4", False)
    fs.rm(path[0] + "/Folder4", True)
    time.sleep(sleep_time)
    assert set(fs.ls(path[0], False)) == {
        "Folder1",
        "Folder2",
        "Folder3",
    }

    with pytest.raises(OSError):
        fs.rm(path[0] + "/Folder3", False)
    with pytest.raises(OSError):
        fs.rmdir(path[0] + "/Folder3")


def test_touch_modified(localserver, clear_server):
    remoteurl, localpath = localserver
    time.sleep(sleep_time)
    with open(localpath + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)
    fs, token, path = fsspec.get_fs_token_paths(
        remoteurl, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    t1 = fs.modified(path[0] + "/testfile.txt")
    assert fs.read_block(path[0] + "/testfile.txt", 0, 4) == b"appl"
    time.sleep(1)
    fs.touch(path[0] + "/testfile.txt", False)
    t2 = fs.modified(path[0] + "/testfile.txt")
    assert fs.read_block(path[0] + "/testfile.txt", 0, 4) == b"appl"
    time.sleep(1)
    fs.touch(path[0] + "/testfile.txt", True)
    t3 = fs.modified(path[0] + "/testfile.txt")
    assert fs.read_block(path[0] + "/testfile.txt", 0, 4) == b""
    assert t1 < t2 and t2 < t3


def test_dir_cache(localserver, clear_server):
    remoteurl, localpath = localserver
    fs, token, path = fsspec.get_fs_token_paths(
        remoteurl, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    fs.mkdir(path[0] + "/Folder1")
    fs.mkdir(path[0] + "/Folder2")
    time.sleep(sleep_time)
    dirs = fs.ls(path[0], True)
    dirs_cached = fs._ls_from_cache(path[0])
    assert dirs == dirs_cached


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_info(localserver, cache_expiry, clear_server):
    remoteurl, localpath = localserver
    with open(localpath + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)
    fs, token, path = fsspec.get_fs_token_paths(
        remoteurl, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    time.sleep(sleep_time)
    assert fs.info(path[0] + "/testfile.txt") in fs.ls(path[0], True)
    _ = fs.ls(path[0], True)
    assert fs.info(path[0] + "/testfile.txt") in fs.ls(path[0], True)


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_walk_find(localserver, cache_expiry, clear_server):
    remoteurl, localpath = localserver
    fs, token, path = fsspec.get_fs_token_paths(
        remoteurl, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    os.makedirs(localpath + "/WalkFolder")
    os.makedirs(localpath + "/WalkFolder/InnerFolder")
    with open(localpath + "/WalkFolder/testfile1.txt", "w") as fout:
        fout.write(TESTDATA2)
    with open(localpath + "/WalkFolder/InnerFolder/testfile2.txt", "w") as fout:
        fout.write(TESTDATA2)
    out = fs.walk(path[0] + "/WalkFolder")
    listing = []
    for item in out:
        listing.append(item)
    assert listing == [
        (path[0] + "/WalkFolder", ["InnerFolder"], ["testfile1.txt"]),
        (path[0] + "/WalkFolder/InnerFolder", [], ["testfile2.txt"]),
    ]
    # unable to use sets here^, would rather
    out = fs.find(path[0] + "/WalkFolder")
    listing = []
    for item in out:
        listing.append(item)
    assert set(listing) == {
        path[0] + "/WalkFolder/InnerFolder/testfile2.txt",
        path[0] + "/WalkFolder/testfile1.txt",
    }


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_du(localserver, cache_expiry, clear_server):
    remoteurl, localpath = localserver
    os.makedirs(localpath + "/WalkFolder")
    os.makedirs(localpath + "/WalkFolder/InnerFolder")
    with open(localpath + "/WalkFolder/testfile1.txt", "w") as fout:
        fout.write(TESTDATA2)
    with open(localpath + "/WalkFolder/InnerFolder/testfile2.txt", "w") as fout:
        fout.write(TESTDATA2)
    fs, token, path = fsspec.get_fs_token_paths(
        remoteurl, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    assert fs.du(path[0] + "/WalkFolder", False) == {
        path[0] + "/WalkFolder/InnerFolder/testfile2.txt": 21,
        path[0] + "/WalkFolder/testfile1.txt": 21,
    }
    assert fs.du(path[0] + "/WalkFolder", True) == 42


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_glob(localserver, cache_expiry, clear_server):
    remoteurl, localpath = localserver
    os.makedirs(localpath + "/WalkFolder")
    with open(localpath + "/WalkFolder/testfile1.txt", "w") as fout:
        fout.write(TESTDATA2)
    with open(localpath + "/WalkFolder/testfile2.txt", "w") as fout:
        fout.write(TESTDATA2)
    time.sleep(sleep_time)
    fs, token, path = fsspec.get_fs_token_paths(
        remoteurl, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    assert set(fs.glob(path[0] + "/WalkFolder/*.txt")) == {
        path[0] + "/WalkFolder/testfile1.txt",
        path[0] + "/WalkFolder/testfile2.txt",
    }


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_cat(localserver, cache_expiry, clear_server):
    remoteurl, localpath = localserver
    os.makedirs(localpath + "/WalkFolder")
    with open(localpath + "/WalkFolder/testfile1.txt", "w") as fout:
        fout.write(TESTDATA1)
    with open(localpath + "/testfile2.txt", "w") as fout:
        fout.write(TESTDATA2)
    time.sleep(sleep_time)
    fs, token, path = fsspec.get_fs_token_paths(
        remoteurl, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    assert fs.cat_file(path[0] + "/testfile2.txt", 4, 9) == b"green"

    paths = [
        path[0] + "/testfile2.txt",
        path[0] + "/WalkFolder/testfile1.txt",
        path[0] + "/testfile2.txt",
    ]
    starts = [4, 6, 4]
    ends = [9, 12, 10]
    assert fs.cat_ranges(paths, starts, ends, batch_size=100) == [
        b"green",
        b"banana",
        b"green\n",
    ]


def test_chunks_to_vectors():
    assert _chunks_to_vectors([(0, 10), (10, 30), (30, 35)], 100, 15) == [
        [(0, 10), (10, 15), (25, 5), (30, 5)]
    ]
    assert _chunks_to_vectors([(0, 10), (10, 30), (30, 35)], 2, 100) == [
        [(0, 10), (10, 20)],
        [(30, 5)],
    ]


def test_vectors_to_chunks(localserver, clear_server):
    from dataclasses import dataclass

    @dataclass
    class MockVectorReadInfo:
        offset: int
        length: int
        buffer: bytes

    res = [
        MockVectorReadInfo(0, 10, b"0" * 10),
        MockVectorReadInfo(10, 10, b"0" * 10),
        MockVectorReadInfo(20, 10, b"0" * 10),
        MockVectorReadInfo(30, 10, b"0" * 10),
    ]

    assert _vectors_to_chunks([(0, 10), (10, 30), (30, 40)], [res]) == [
        b"0" * 10,
        b"0" * 20,
        b"0" * 10,
    ]


def test_glob_full_names(localserver, clear_server):
    remoteurl, localpath = localserver
    os.makedirs(localpath + "/WalkFolder")
    with open(localpath + "/WalkFolder/testfile1.txt", "w") as fout:
        fout.write(TESTDATA1)
    with open(localpath + "/WalkFolder/testfile2.txt", "w") as fout:
        fout.write(TESTDATA2)
    time.sleep(sleep_time)

    full_names = [
        f.full_name for f in fsspec.open_files(remoteurl + "/WalkFolder/*.txt")
    ]

    for name in full_names:
        with fsspec.open(name) as f:
            assert f.read() in [bytes(data, "utf-8") for data in [TESTDATA1, TESTDATA2]]


@pytest.mark.parametrize("protocol_prefix", ["", "simplecache::"])
def test_cache(localserver, clear_server, protocol_prefix):
    data = TESTDATA1 * int(1e7 / len(TESTDATA1))  # bigger than the chunk size
    remoteurl, localpath = localserver
    with open(localpath + "/testfile.txt", "w") as fout:
        fout.write(data)

    with fsspec.open(protocol_prefix + remoteurl + "/testfile.txt", "rb") as f:
        contents = f.read()
        assert contents == data.encode("utf-8")


def test_cache_directory(localserver, clear_server, tmp_path):
    remoteurl, localpath = localserver
    with open(localpath + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)

    cache_directory = tmp_path / "cache"
    with fsspec.open(
        "simplecache::" + remoteurl + "/testfile.txt",
        "rb",
        simplecache={"cache_storage": str(cache_directory)},
    ) as f:
        contents = f.read()
        assert contents == TESTDATA1.encode("utf-8")

    assert len(os.listdir(cache_directory)) == 1
    with open(cache_directory / os.listdir(cache_directory)[0], "rb") as f:
        contents = f.read()
        assert contents == TESTDATA1.encode("utf-8")


def test_close_while_reading(localserver, clear_server):
    remoteurl, localpath = localserver
    data = TESTDATA1 * int(1e8 / len(TESTDATA1))
    with open(localpath + "/testfile.txt", "w") as fout:
        fout.write(data)

    fs, _, (path,) = fsspec.get_fs_token_paths(remoteurl + "/testfile.txt")

    async def reader():
        tic = time.monotonic()
        await fs._cat_file(path, start=0, end=None)
        toc = time.monotonic()
        return tic, toc

    async def closer():
        await asyncio.sleep(0.001)
        tic = time.monotonic()
        await fs._readonly_filehandle_cache._close(path, 1)
        toc = time.monotonic()
        return tic, toc

    async def run():
        (read_start, read_stop), (close_start, close_stop) = await asyncio.gather(
            reader(), closer()
        )
        assert read_start < close_start < read_stop
        assert read_start < close_stop < read_stop

    asyncio.run(run())
