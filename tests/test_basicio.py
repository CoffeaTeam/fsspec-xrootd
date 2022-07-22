"""Test basic IO against a xrootd server fixture"""
from __future__ import annotations

import shutil
import subprocess
import time

import fsspec
import pytest

TESTDATA1 = "apple\nbanana\norange\ngrape"
TESTDATA2 = "red\ngreen\nyellow\nblue"
sleep_time = 0.2
expiry_time = 0.1


@pytest.fixture(scope="module")
def localserver(tmpdir_factory):
    srvdir = tmpdir_factory.mktemp("srv")
    with open(srvdir.join("testfile.txt"), "w") as fout:
        fout.write(TESTDATA1)

    xrdexe = shutil.which("xrootd")
    proc = subprocess.Popen([xrdexe, srvdir])
    time.sleep(2)  # give it some startup
    yield "root://localhost/" + str(srvdir)
    proc.terminate()
    proc.wait(timeout=10)


def test_broken_server():
    with pytest.raises(OSError):
        # try to connect on the wrong port should fail
        _ = fsspec.open("root://localhost:12345/")


def test_ping(localserver):
    from XRootD import client

    fs = client.FileSystem(localserver)
    status, _n = fs.ping()
    if not status.ok:
        raise OSError(f"Server did not run properly: {status.message}")


def test_read_xrd(localserver):
    from XRootD import client

    with client.File() as f:
        status, _ = f.open(localserver + "/testfile.txt")
        if not status.ok:
            raise RuntimeError(status)
        status, res = f.read()
        if not status.ok:
            raise RuntimeError(status)
        assert res.decode("ascii") == TESTDATA1
        f.close()


def test_read_fsspec(localserver):
    with fsspec.open(localserver + "/testfile.txt", "rt") as f:
        assert f.read() == TESTDATA1
        f.seek(0)
        assert f.readline() == "apple\n"
        f.seek(0)
        lns = f.readlines()
        assert lns[2] == "orange\n"
        f.seek(1)
        assert f.read(1) == "p"

    with fsspec.open(localserver + "/testfile.txt", "rb") as f:
        assert f.readuntil(b"e") == b"apple"

    fs, token, path = fsspec.get_fs_token_paths(localserver + "/testfile.txt", "rt")
    assert fs.read_block(path[0], 0, 4) == b"appl"
    fs.rm(path[0], True)


def test_write_fsspec(localserver):
    with fsspec.open(localserver + "/testfile.txt", "wt") as f:
        f.write(TESTDATA1)
        f.flush()
    with fsspec.open(localserver + "/testfile.txt", "rt") as f:
        assert f.read() == TESTDATA1
    fs, token, path = fsspec.get_fs_token_paths(localserver + "/testfile.txt", "rt")
    fs.rm(path[0], True)


def test_append_fsspec(localserver):
    with fsspec.open(localserver + "/testfile.txt", "wt") as f:
        f.write(TESTDATA1)
        f.flush()
    with fsspec.open(localserver + "/testfile.txt", "at") as f:
        f.write(TESTDATA2)
        f.flush()
    with fsspec.open(localserver + "/testfile.txt", "rt") as f:
        assert f.read() == TESTDATA1 + TESTDATA2
    fs, token, path = fsspec.get_fs_token_paths(localserver + "/testfile.txt", "rt")
    fs.rm(path[0], True)


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_mk_and_rm_dir_fsspec(localserver, cache_expiry):
    with fsspec.open(localserver + "/Folder1/testfile1.txt", "wt") as f:
        f.write(TESTDATA2)
        f.flush()
    with fsspec.open(localserver + "/Folder2/testfile2.txt", "wt") as f:
        f.write(TESTDATA2)
        f.flush()
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": cache_expiry}
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
    fs.rm(path[0] + "/Folder1", True)
    fs.rm(path[0] + "/Folder2", True)
    fs.rm(path[0] + "/Folder3", True)


def test_touch_modified(localserver):
    time.sleep(sleep_time)
    with fsspec.open(localserver + "/testfile.txt", "wt") as f:
        f.write(TESTDATA1)
        f.flush()
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": expiry_time}
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
    fs.rm(path[0] + "/testfile.txt", True)


def test_dir_cache(localserver):
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    fs.mkdir(path[0] + "/Folder1")
    fs.mkdir(path[0] + "/Folder2")
    time.sleep(sleep_time)
    dirs = fs.ls(path[0], True)
    dirs_cached = fs._ls_from_cache(path[0])
    assert dirs == dirs_cached
    fs.rm(path[0] + "/Folder1")
    fs.rm(path[0] + "/Folder2")


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_info(localserver, cache_expiry):
    with fsspec.open(localserver + "/testfile.txt", "wt") as f:
        f.write(TESTDATA1)
        f.flush()
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    time.sleep(sleep_time)
    assert fs.info(path[0] + "/testfile.txt") in fs.ls(path[0], True)
    _ = fs.ls(path[0], True)
    assert fs.info(path[0] + "/testfile.txt") in fs.ls(path[0], True)
    fs.rm(path[0] + "/testfile.txt")


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_walk_find(localserver, cache_expiry):
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    with fsspec.open(localserver + "/WalkFolder/testfile1.txt", "wt") as f:
        f.write(TESTDATA2)
        f.flush()
    with fsspec.open(localserver + "/WalkFolder/InnerFolder/testfile2.txt", "wt") as f:
        f.write(TESTDATA2)
        f.flush()
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
    fs.rm(path[0] + "/WalkFolder", True)


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_du(localserver, cache_expiry):
    with fsspec.open(localserver + "/WalkFolder/testfile1.txt", "wt") as f:
        f.write(TESTDATA2)
        f.flush()
    with fsspec.open(localserver + "/WalkFolder/InnerFolder/testfile2.txt", "wt") as f:
        f.write(TESTDATA2)
        f.flush()
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    assert fs.du(path[0] + "/WalkFolder", False) == {
        path[0] + "/WalkFolder/InnerFolder/testfile2.txt": 21,
        path[0] + "/WalkFolder/testfile1.txt": 21,
    }
    assert fs.du(path[0] + "/WalkFolder", True) == 42
    fs.rm(path[0] + "/WalkFolder", True)


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_glob(localserver, cache_expiry):
    with fsspec.open(localserver + "/WalkFolder/testfile1.txt", "wt") as f:
        f.write(TESTDATA2)
        f.flush()
    with fsspec.open(localserver + "/WalkFolder/testfile2.txt", "wt") as f:
        f.write(TESTDATA2)
        f.flush()
    time.sleep(sleep_time)
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    print(fs.glob(path[0] + "/*.txt"))
    assert set(fs.glob(path[0] + "/WalkFolder/*.txt")) == {
        path[0] + "/WalkFolder/testfile1.txt",
        path[0] + "/WalkFolder/testfile2.txt",
    }
    fs.rm(path[0] + "/WalkFolder", True)
