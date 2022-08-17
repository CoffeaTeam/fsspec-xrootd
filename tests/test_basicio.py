"""Test basic IO against a xrootd server fixture"""
from __future__ import annotations

import os
import shutil
import subprocess
import time

import fsspec
import pytest

from fsspec_xrootd.xrootd import _chunks_to_vectors, _vectors_to_chunks

TESTDATA1 = "apple\nbanana\norange\ngrape"
TESTDATA2 = "red\ngreen\nyellow\nblue"
sleep_time = 0.2
expiry_time = 0.1


@pytest.fixture(scope="module")
def localserver(tmpdir_factory):
    srvdir = tmpdir_factory.mktemp("srv")
    tempPath = os.path.join(srvdir, "Folder")
    os.mkdir(tempPath)
    xrdexe = shutil.which("xrootd")
    proc = subprocess.Popen([xrdexe, srvdir])
    time.sleep(2)  # give it some startup
    yield "root://localhost/" + str(tempPath), tempPath
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture()
def clear_server(localserver):
    remoteurl, localpath = localserver
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


def test_broken_server(localserver):
    with pytest.raises(OSError):
        # try to connect on the wrong port should fail
        with fsspec.open("root://localhost:12345/", "rt", timeout=5) as f:
            _ = f.read()


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
