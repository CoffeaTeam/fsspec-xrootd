"""Test basic IO against a xrootd server fixture"""
from __future__ import annotations

import os
import sys
import time

import fsspec
import pytest
import XRootD

from fsspec_xrootd.xrootd import (
    XRootDFileSystem,
    _chunks_to_vectors,
    _vectors_to_chunks,
)

TESTDATA1 = "apple\nbanana\norange\ngrape"
TESTDATA2 = "red\ngreen\nyellow\nblue"
sleep_time = 0.2
expiry_time = 0.1

macos = sys.platform == "darwin"


def test_ping(server, clear_server):
    url, path = server

    fs = XRootD.client.FileSystem(url)
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


def test_pickle(server, clear_server):
    import pickle

    url, path = server

    fs, _, (path,) = fsspec.get_fs_token_paths(url)
    assert fs.ls(path) == []
    fs = pickle.loads(pickle.dumps(fs))
    assert fs.ls(path) == []
    time.sleep(1)


def test_read_xrd(server, clear_server):
    url, path = server
    with open(path + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)

    with XRootD.client.File() as f:
        status, _ = f.open(url + "/testfile.txt")
        if not status.ok:
            raise RuntimeError(status)
        status, res = f.read()
        if not status.ok:
            raise RuntimeError(status)
        assert res.decode("ascii") == TESTDATA1
        f.close()


def test_read_fsspec(server, clear_server):
    url, path = server
    with open(path + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)

    with fsspec.open(url + "/testfile.txt", "rt") as f:
        assert f.read() == TESTDATA1
        f.seek(0)
        assert f.readline() == "apple\n"
        f.seek(0)
        lns = f.readlines()
        assert lns[2] == "orange\n"
        f.seek(1)
        assert f.read(1) == "p"

    with fsspec.open(url + "/testfile.txt", "rb") as f:
        assert f.readuntil(b"e") == b"apple"

    fs, token, path = fsspec.get_fs_token_paths(url + "/testfile.txt", "rt")
    assert fs.read_block(path[0], 0, 4) == b"appl"


def test_write_fsspec(server, clear_server):
    url, path = server
    with fsspec.open(url + "/testfile.txt", "wt") as f:
        f.write(TESTDATA1)
        f.flush()
    with open(path + "/testfile.txt") as f:
        assert f.read() == TESTDATA1


@pytest.mark.skipif(macos, reason="Not working on macos")
def test_append_fsspec(server, clear_server):
    url, path = server
    with open(path + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)
    with fsspec.open(url + "/testfile.txt", "at") as f:
        f.write(TESTDATA2)
        f.flush()
    with open(path + "/testfile.txt") as f:
        assert f.read() == TESTDATA1 + TESTDATA2


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_mk_and_rm_dir_fsspec(server, cache_expiry, clear_server):
    url, path = server
    os.makedirs(path + "/Folder1")
    os.makedirs(path + "/Folder2")
    with open(path + "/Folder1/testfile1.txt", "w") as fout:
        fout.write(TESTDATA2)
    with open(path + "/Folder2/testfile2.txt", "w") as fout:
        fout.write(TESTDATA2)
    fs, token, path = fsspec.get_fs_token_paths(
        url, "rt", storage_options={"listings_expiry_time": cache_expiry}
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


def test_touch_modified(server, clear_server):
    url, path = server
    time.sleep(sleep_time)
    with open(path + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)
    fs, token, path = fsspec.get_fs_token_paths(
        url, "rt", storage_options={"listings_expiry_time": expiry_time}
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


def test_dir_cache(server, clear_server):
    url, path = server
    fs, token, path = fsspec.get_fs_token_paths(
        url, "rt", storage_options={"listings_expiry_time": expiry_time}
    )
    fs.mkdir(path[0] + "/Folder1")
    fs.mkdir(path[0] + "/Folder2")
    time.sleep(sleep_time)
    dirs = fs.ls(path[0], True)
    dirs_cached = fs._ls_from_cache(path[0])
    assert dirs == dirs_cached


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_info(server, cache_expiry, clear_server):
    url, path = server
    with open(path + "/testfile.txt", "w") as fout:
        fout.write(TESTDATA1)
    fs, token, path = fsspec.get_fs_token_paths(
        url, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    time.sleep(sleep_time)
    assert fs.info(path[0] + "/testfile.txt") in fs.ls(path[0], True)
    _ = fs.ls(path[0], True)
    assert fs.info(path[0] + "/testfile.txt") in fs.ls(path[0], True)


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_walk_find(server, cache_expiry, clear_server):
    url, local_path = server
    fs, token, path_list = fsspec.get_fs_token_paths(
        url, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    os.makedirs(local_path + "/WalkFolder")
    os.makedirs(local_path + "/WalkFolder/InnerFolder")
    with open(local_path + "/WalkFolder/testfile1.txt", "w") as fout:
        fout.write(TESTDATA2)
    with open(local_path + "/WalkFolder/InnerFolder/testfile2.txt", "w") as fout:
        fout.write(TESTDATA2)
    out = fs.walk(path_list[0] + "/WalkFolder")
    listing = []
    for item in out:
        listing.append(item)
    assert listing == [
        (path_list[0] + "/WalkFolder", ["InnerFolder"], ["testfile1.txt"]),
        (path_list[0] + "/WalkFolder/InnerFolder", [], ["testfile2.txt"]),
    ]
    # unable to use sets here^, would rather
    out = fs.find(path_list[0] + "/WalkFolder")
    listing = []
    for item in out:
        listing.append(item)
    assert set(listing) == {
        path_list[0] + "/WalkFolder/InnerFolder/testfile2.txt",
        path_list[0] + "/WalkFolder/testfile1.txt",
    }


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_du(server, cache_expiry, clear_server):
    url, path = server
    os.makedirs(path + "/WalkFolder")
    os.makedirs(path + "/WalkFolder/InnerFolder")
    with open(path + "/WalkFolder/testfile1.txt", "w") as fout:
        fout.write(TESTDATA2)
    with open(path + "/WalkFolder/InnerFolder/testfile2.txt", "w") as fout:
        fout.write(TESTDATA2)
    fs, token, path = fsspec.get_fs_token_paths(
        url, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    assert fs.du(path[0] + "/WalkFolder", False) == {
        path[0] + "/WalkFolder/InnerFolder/testfile2.txt": 21,
        path[0] + "/WalkFolder/testfile1.txt": 21,
    }
    assert fs.du(path[0] + "/WalkFolder", True) == 42


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_glob(server, cache_expiry, clear_server):
    url, path = server
    os.makedirs(path + "/WalkFolder")
    with open(path + "/WalkFolder/testfile1.txt", "w") as fout:
        fout.write(TESTDATA2)
    with open(path + "/WalkFolder/testfile2.txt", "w") as fout:
        fout.write(TESTDATA2)
    time.sleep(sleep_time)
    fs, token, path = fsspec.get_fs_token_paths(
        url, "rt", storage_options={"listings_expiry_time": cache_expiry}
    )
    assert set(fs.glob(path[0] + "/WalkFolder/*.txt")) == {
        path[0] + "/WalkFolder/testfile1.txt",
        path[0] + "/WalkFolder/testfile2.txt",
    }


@pytest.mark.parametrize("cache_expiry", [0, expiry_time])
def test_cat(server, cache_expiry, clear_server):
    url, path = server
    os.makedirs(path + "/WalkFolder")
    with open(path + "/WalkFolder/testfile1.txt", "w") as fout:
        fout.write(TESTDATA1)
    with open(path + "/testfile2.txt", "w") as fout:
        fout.write(TESTDATA2)
    time.sleep(sleep_time)
    fs, token, path = fsspec.get_fs_token_paths(
        url, "rt", storage_options={"listings_expiry_time": cache_expiry}
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


def test_vectors_to_chunks(server, clear_server):
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
