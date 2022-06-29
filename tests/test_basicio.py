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
    with pytest.raises(IOError):
        # try to connect on the wrong port should fail
        _ = fsspec.open("root://localhost:12345/")


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
        f.seek(0)
        # assert f.readuntil(b"e") == b"apple"


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
