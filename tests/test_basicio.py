"""Test basic IO against a xrootd server fixture"""
from __future__ import annotations

import shutil
import subprocess
import time

import fsspec
import pytest

TESTDATA = "apple\nbanana\norange\ngrape"
TESTWRITEDATA = "the end is never the end is never the end"
sleep_time = 1
expiry_time = 0.1


@pytest.fixture(scope="module")
def localserver(tmpdir_factory):
    srvdir = tmpdir_factory.mktemp("srv")
    with open(srvdir.join("testfile.txt"), "w") as fout:
        fout.write(TESTDATA)

    xrdexe = shutil.which("xrootd")
    proc = subprocess.Popen([xrdexe, srvdir])
    time.sleep(2)  # give it some startup
    yield "root://localhost/" + str(srvdir)
    proc.terminate()
    proc.wait(timeout=10)


@pytest.mark.skip("not implemented")
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
        assert res.decode("ascii") == TESTDATA
        f.close()


def test_read_fsspec(localserver):
    with fsspec.open(localserver + "/testfile.txt", "rt") as f:
        assert f.read() == TESTDATA
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


def test_write_fsspec(localserver):
    time.sleep(sleep_time)
    with fsspec.open(localserver + "/testfile2.txt", "wt") as f:
        f.write(TESTWRITEDATA)
        f.flush()
    with fsspec.open(localserver + "/testfile2.txt", "rt") as f:
        assert f.read() == TESTWRITEDATA


def test_append_fsspec(localserver):
    with fsspec.open(localserver + "/testfile2.txt", "at") as f:
        f.write(TESTWRITEDATA)
        f.flush()
    with fsspec.open(localserver + "/testfile2.txt", "rt") as f:
        assert f.read() == TESTWRITEDATA + TESTWRITEDATA


def test_mk_and_rm_dir_fsspec(localserver):
    with fsspec.open(localserver + "/Folder/test1.txt", "wt") as f:
        f.write(TESTWRITEDATA)
        f.flush()
    with fsspec.open(localserver + "/Folder/test2.txt", "wt") as f:
        f.write(TESTWRITEDATA)
        f.flush()
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    time.sleep(sleep_time)

    assert fs.ls(path[0], False) == ["testfile.txt", "testfile2.txt", "Folder"]
    fs.mkdir(path[0] + "/Folder2/Folder3")
    time.sleep(sleep_time)
    assert set(fs.ls(path[0], False)) == {
        "testfile.txt",
        "testfile2.txt",
        "Folder2",
        "Folder",
    }
    with pytest.raises(OSError):
        fs.mkdir(path[0] + "/Folder4/Folder5", False)

    fs.mkdirs(path[0] + "/testfolder4")
    time.sleep(sleep_time)
    assert set(fs.ls(path[0], False)) == {
        "testfile.txt",
        "testfile2.txt",
        "Folder2",
        "testfolder4",
        "Folder",
    }
    fs.mkdirs(path[0] + "/testfolder4", True)

    with pytest.raises(OSError):
        fs.mkdirs(path[0] + "/testfolder4", False)

    fs.rm(path[0] + "/Folder", True)
    time.sleep(sleep_time)
    assert set(fs.ls(path[0], False)) == {
        "testfile.txt",
        "testfile2.txt",
        "Folder2",
        "testfolder4",
    }

    with pytest.raises(OSError):
        fs.rm(path[0] + "/Folder2", False)
    with pytest.raises(OSError):
        fs.rmdir(path[0] + "/Folder2")
    fs.mkdir(path[0] + "/Folder3", False)
    time.sleep(1)


@pytest.mark.skip("not implemented")
def test_touch_modified(localserver):
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    t1 = fs.modified(path[0] + "/testfile.txt")
    assert fs.read_block(path[0] + "/testfile.txt", 0, 4) == b"appl"
    time.sleep(sleep_time)
    fs.touch(path[0] + "/testfile.txt", False)
    t2 = fs.modified(path[0] + "/testfile.txt")
    assert fs.read_block(path[0] + "/testfile.txt", 0, 4) == b"appl"
    time.sleep(sleep_time)
    fs.touch(path[0] + "/testfile.txt", True)
    t3 = fs.modified(path[0] + "/testfile.txt")
    assert fs.read_block(path[0] + "/testfile.txt", 0, 4) == b""
    assert (
        t1 < t2 and t2 < t3
    )  # for some reason, touching without truncation doesn't update the mod time!?


def test_dir_cache(localserver):
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    dirs = fs.ls(path[0], True)
    dirs_cached = fs._ls_from_cache(path[0])
    assert dirs == dirs_cached


def test_info(localserver):
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    time.sleep(sleep_time)
    assert fs.info(path[0] + "/testfile.txt") in fs.ls(path[0], True)
    assert fs.info(path[0] + "/testfile.txt") in fs.ls(path[0], True)


@pytest.mark.skip("not implemented")
def test_sign(localserver):
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    print(fs.sign(path[0]))
    print(localserver)
    assert fs.sign(path[0]) == localserver  # why do these disagree?


def test_walk_find(localserver):
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    with fsspec.open(localserver + "/WalkFolder/testfile1.txt", "wt") as f:
        f.write(TESTWRITEDATA)
        f.flush()
    with fsspec.open(localserver + "/WalkFolder/InnerFolder/testfile2.txt", "wt") as f:
        f.write(TESTWRITEDATA)
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


def test_du(localserver):
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    assert fs.du(path[0] + "/WalkFolder", False) == {
        path[0] + "/WalkFolder/InnerFolder/testfile2.txt": 41,
        path[0] + "/WalkFolder/testfile1.txt": 41,
    }
    assert fs.du(path[0] + "/WalkFolder", True) == 82


def test_glob(localserver):
    fs, token, path = fsspec.get_fs_token_paths(
        localserver, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    assert set(fs.glob(path[0] + "/*.txt")) == {
        path[0] + "/testfile.txt",
        path[0] + "/testfile2.txt",
    }
