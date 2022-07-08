"""Test basic IO against a xrootd server fixture"""
from __future__ import annotations

import shutil
import subprocess
import time

import fsspec
import pytest

TESTDATA = "apple\nbanana\norange\ngrape"
TESTWRITEDATA = "the end is never the end is never the end"


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


def test_mkdir_fsspec(localserver):
    with fsspec.open(localserver + "/Folder/Test1/test1.txt", "wt") as f:
        f.write(TESTWRITEDATA)
        f.flush()
    with fsspec.open(localserver + "/Folder/Test2/test2.txt", "wt") as f:
        f.write(TESTWRITEDATA)
        f.flush()
    fs, token, path = fsspec.get_fs_token_paths(localserver, "rt")
    assert fs.ls(path[0], False) == ["testfile.txt", "testfile2.txt", "Folder"]
    fs.mkdir(path[0] + "/Folder2/testfile3.txt")
    assert set(fs.ls(path[0], False)) == {
        "testfile.txt",
        "testfile2.txt",
        "Folder2",
        "Folder",
    }

    try:
        fs.mkdir(path[0] + "/Folder3/testfile3.txt", False)
        a = False
    except:
        a = True
    assert a

    fs.mkdirs(path[0] + "/testfile4.txt")
    assert set(fs.ls(path[0], False)) == {
        "testfile.txt",
        "testfile2.txt",
        "Folder2",
        "testfile4.txt",
        "Folder",
    }
    fs.mkdirs(path[0] + "/testfile4.txt", True)

    with pytest.raises(OSError):
        fs.mkdirs(path[0] + "/testfile4.txt", False)

    fs.rm(path[0] + "/Folder", True)
    assert set(fs.ls(path[0], False)) == {
        "testfile.txt",
        "testfile2.txt",
        "Folder2",
        "testfile4.txt",
    }

    with pytest.raises(OSError):
        fs.rm(path[0] + "/Folder2", False)

#def test_misc_fsspec(localserver):

