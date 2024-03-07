from __future__ import annotations

import asyncio
import io
import os.path
import time
import warnings
import weakref
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Coroutine, Iterable, TypeVar

from fsspec.asyn import AsyncFileSystem, _run_coros_in_chunks, sync, sync_wrapper
from fsspec.exceptions import FSTimeoutError
from fsspec.spec import AbstractBufferedFile
from XRootD import client
from XRootD.client.flags import (
    DirListFlags,
    MkDirFlags,
    OpenFlags,
    QueryCode,
    StatInfoFlags,
)
from XRootD.client.responses import HostList, XRootDStatus


class ErrorCodes(IntEnum):
    INVALID_PATH = 400


T = TypeVar("T")
# TODO: Protocol typing when kwargs is supported


def _async_wrap(
    func: Callable[..., XRootDStatus | tuple[XRootDStatus, T]]
) -> Callable[..., Coroutine[Any, Any, tuple[XRootDStatus, T]]]:
    """Wraps pyxrootd functions to run asynchronously. Returns an async callable

    Parameters
    ----------
    func: XRootD function that implements, needs to have a callback option

    Returns
    -------
    A function with the same signature as func, but with an implicit `callback` argument
    """
    future: asyncio.Future[tuple[XRootDStatus, T]] = (
        asyncio.get_running_loop().create_future()
    )

    def callback(status: XRootDStatus, content: T, servers: HostList) -> None:
        if future.cancelled():
            return
        loop = future.get_loop()
        try:
            loop.call_soon_threadsafe(future.set_result, (status, content))
        except Exception as exc:
            loop.call_soon_threadsafe(future.set_exception, exc)

    async def wrapped(*args: Any, **kwargs: Any) -> tuple[XRootDStatus, T]:
        submit_status: XRootDStatus = func(*args, **kwargs, callback=callback)
        if not submit_status.ok:
            raise OSError(
                f"Failed to submit {func!r} request: {submit_status.message.strip()}"
            )
        return await future

    return wrapped


def _chunks_to_vectors(
    file_ranges: list[tuple[int, int]],
    max_num_chunks: int,
    max_chunk_size: int,
) -> list[list[tuple[int, int]]]:
    """Reformats chunks to work with pyxrootd vector_read()

    Parameters
    ----------
    file_ranges: list of 2-tuples, each with a start and end position
    max_num_chunks: int, max # of chunks per vector_read allowed
    max_chunk_size: int, max size of chunk allowed

    Returns
    -------
    A list of vectors. Each vector is a list of tuples.
    Each tuple contains the start position and length of chunk.
    Note the format of the returned tuples is different from the given tuples.
    """

    def split_convert_range(start: int, stop: int) -> Iterable[tuple[int, int]]:
        last = start
        for pos in range(start + max_chunk_size, stop, max_chunk_size):
            yield (last, pos - last)
            last = pos
        yield (last, stop - last)

    groups = []
    grp = []
    for start, stop in file_ranges:
        for r in split_convert_range(start, stop):
            grp.append(r)
            if len(grp) == max_num_chunks:
                groups.append(grp)
                grp = []
    groups.append(grp)
    return groups


def _vectors_to_chunks(
    chunks: list[tuple[int, int]], result_bufs: list[list[Any]]
) -> list[bytes]:
    """Reformats the results of vector_read

    Parameters
    ----------
    chunks: list of 2-tuples, each in format (start, end)
    result_bufs: list of VectorReadInfo objects from pyxrootd

    Returns
    -------
    List of bytes
    """
    subchunks = (buf for buffers in result_bufs for buf in buffers)

    deets: list[bytes] = []
    for chunk in chunks:
        chunk_length = chunk[1] - chunk[0]
        chunk_data = b""
        while len(chunk_data) < chunk_length:
            chunk_data += next(subchunks).buffer
        deets.append(chunk_data)
    return deets


@dataclass
class _CacheItem:
    accessed: float
    handle: client.File


class ReadonlyFileHandleCache:
    def __init__(self, loop: Any, max_items: int | None, ttl: int):
        self.loop = loop
        self._max_items = max_items
        self._ttl = int(ttl)
        self._cache: dict[str, _CacheItem] = {}
        sync(loop, self._start_pruner)
        weakref.finalize(self, self._close_all, loop, self._cache)

    @staticmethod
    def _close_all(loop: Any, cache: dict[str, _CacheItem]) -> None:
        if loop is not None and loop.is_running():

            async def closure() -> None:
                await asyncio.gather(
                    *(_async_wrap(item.handle.close)() for item in cache.values())
                )

            try:
                sync(loop, closure, timeout=0.5)
            except (TimeoutError, FSTimeoutError, NotImplementedError):
                pass
        else:
            # fire and forget
            for item in cache.values():
                item.handle.close(callback=lambda *args: None)
        cache.clear()

    def close_all(self) -> None:
        self._close_all(self.loop, self._cache)

    async def _close(self, url: str, timeout: int) -> None:
        item = self._cache.pop(url, None)
        if item:
            status, _ = await _async_wrap(item.handle.close)(timeout=timeout)
            if not status.ok:
                raise OSError(f"Failed to close file: {status.message}")

    close = sync_wrapper(_close)

    async def _start_pruner(self) -> None:
        self._prune_task = asyncio.create_task(self._pruner())

    async def _pruner(self) -> None:
        while True:
            await self._prune_cache(self._ttl // 2)
            await asyncio.sleep(self._ttl)

    async def _prune_cache(self, timeout: int) -> None:
        now = time.monotonic()
        oldest_keys = sorted((item.accessed, key) for key, item in self._cache.items())
        to_close = []
        if self._max_items:
            to_close += oldest_keys[: -self._max_items]
            oldest_keys = oldest_keys[-self._max_items :]
        for last_access, key in oldest_keys:
            if now - last_access > self._ttl:
                to_close.append((last_access, key))
        await asyncio.gather(*(self._close(key, timeout) for _, key in to_close))

    async def _open(self, url: str, timeout: int) -> Any:  # client.File
        if url in self._cache:
            item = self._cache[url]
            item.accessed = time.monotonic()
            return item.handle
        handle = client.File()
        status, _ = await _async_wrap(handle.open)(
            url,
            OpenFlags.READ,
            timeout=timeout,
        )
        if not status.ok:
            raise OSError(f"Failed to open file: {status.message}")
        self._cache[url] = _CacheItem(accessed=time.monotonic(), handle=handle)
        await self._prune_cache(timeout)
        return handle


class XRootDFileSystem(AsyncFileSystem):  # type: ignore[misc]
    protocol = "root"
    root_marker = "/"
    default_timeout = 60
    async_impl = True
    default_max_num_chunks = 1024
    default_max_chunk_size = 2097136

    _dataserver_info_cache: dict[str, Any] = defaultdict(dict)

    def __init__(
        self,
        hostid: str,
        asynchronous: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        **storage_options: Any,
    ) -> None:
        """
        Direct construction of XRootDFileSystem

        Parameters
        ----------
        hostid: str
            The hostname, optionally including port, username, or password in the
            standard format (user:pass@host.name:port)
        asynchronous: bool
            If true, synchronous methods will not be available in this instance
        loop:
            Bring your own loop (for sync methods)
        """
        super().__init__(self, asynchronous=asynchronous, loop=loop, **storage_options)
        self.timeout = storage_options.get("timeout", XRootDFileSystem.default_timeout)
        self.hostid = hostid
        self._myclient = client.FileSystem("root://" + hostid)
        if not self._myclient.url.is_valid():
            raise ValueError(f"Invalid hostid: {hostid!r}")
        storage_options.setdefault("listing_expiry_time", 0)
        self.storage_options = storage_options
        self._readonly_filehandle_cache = ReadonlyFileHandleCache(
            self.loop,
            max_items=storage_options.get("filehandle_cache_size", 256),
            ttl=storage_options.get("filehandle_cache_ttl", 30),
        )

    def invalidate_cache(self, path: str | None = None) -> None:
        if path is None:
            self.dircache.clear()
            self._readonly_filehandle_cache.close_all()
        else:
            try:
                del self.dircache[path]
            except KeyError:
                pass
            self._readonly_filehandle_cache.close(
                self.unstrip_protocol(path), self.timeout
            )

    @staticmethod
    def _get_kwargs_from_urls(u: str) -> dict[Any, Any]:
        url = client.URL(u)
        # The hostid encapsulates user,pass,host,port in one string
        return {"hostid": url.hostid}

    @classmethod
    def _strip_protocol(cls, path: str | list[str]) -> Any:
        if isinstance(path, str):
            if path.startswith(cls.protocol):
                return client.URL(path).path.rstrip("/") or cls.root_marker
            # assume already stripped
            return path.rstrip("/") or cls.root_marker
        elif isinstance(path, list):
            return [cls._strip_protocol(item) for item in path]
        else:
            raise ValueError("Strip protocol not given string or list")

    def unstrip_protocol(self, name: str) -> str:
        prefix = f"{self.protocol}://{self.hostid}/"
        if name.startswith(prefix):
            return name
        return prefix + name

    async def _mkdir(
        self, path: str, create_parents: bool = True, **kwargs: Any
    ) -> None:
        if create_parents:
            status, _ = await _async_wrap(self._myclient.mkdir)(
                path, flags=MkDirFlags.MAKEPATH, timeout=self.timeout
            )
        else:
            status, _ = await _async_wrap(self._myclient.mkdir)(
                path, timeout=self.timeout
            )
        if not status.ok:
            raise OSError(f"Directory not made properly: {status.message}")

    async def _makedirs(self, path: str, exist_ok: bool = False) -> None:
        if not exist_ok:
            if await self._exists(path):
                raise OSError(
                    "Location already exists and exist_ok arg was set to false"
                )
        status, _ = await _async_wrap(self._myclient.mkdir)(
            path, MkDirFlags.MAKEPATH, timeout=self.timeout
        )
        if not status.ok and not (status.code == ErrorCodes.INVALID_PATH and exist_ok):
            raise OSError(f"Directory not made properly: {status.message}")

    async def _rm(
        self,
        path: str,
        recursive: bool = False,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> Any:
        # TODO: ensure the coros run in order, pyxrootd won't work otherwise
        # TODO: implement on_error
        batch_size = batch_size or self.batch_size
        path = await self._expand_path(path, recursive=recursive)
        return await _run_coros_in_chunks(
            [self._rm_file(p, **kwargs) for p in reversed(path)],
            batch_size=batch_size,
            nofiles=True,
        )

    async def _rmdir(self, path: str) -> None:
        status, _ = await _async_wrap(self._myclient.rmdir)(path, self.timeout)
        if not status.ok:
            raise OSError(f"Directory not removed properly: {status.message}")

    rmdir = sync_wrapper(_rmdir)

    async def _rm_file(self, path: str, **kwargs: Any) -> None:
        status, _ = await _async_wrap(self._myclient.rm)(path, self.timeout)
        if not status.ok:
            raise OSError(f"File not removed properly: {status.message}")

    async def _touch(self, path: str, truncate: bool = False, **kwargs: Any) -> None:
        if truncate or not await self._exists(path):
            status, _ = await _async_wrap(self._myclient.truncate)(
                path, size=0, timeout=self.timeout
            )
            if not status.ok:
                raise OSError(f"File not touched properly: {status.message}")
        else:
            len = await self._info(path)
            status, _ = await _async_wrap(self._myclient.truncate)(
                path,
                size=len.get("size"),
                timeout=self.timeout,
            )
            if not status.ok:
                raise OSError(f"File not touched properly: {status.message}")

    touch = sync_wrapper(_touch)

    async def _modified(self, path: str) -> Any:
        status, statInfo = await _async_wrap(self._myclient.stat)(path, self.timeout)  # type: ignore[var-annotated]
        return statInfo.modtime

    modified = sync_wrapper(_modified)

    async def _exists(self, path: str, **kwargs: Any) -> bool:
        if path in self.dircache:
            return True
        else:
            status, _ = await _async_wrap(self._myclient.stat)(path, self.timeout)
            if status.code == ErrorCodes.INVALID_PATH:
                return False
            elif not status.ok:
                raise OSError(f"status check failed with message: {status.message}")
            return True

    async def _info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        spath = os.path.split(path)
        deet = self._ls_from_cache(spath[0])
        if deet is not None and len(deet) != 0:
            for item in deet:
                if item["name"] == path:
                    return {
                        "name": path,
                        "size": item["size"],
                        "type": item["type"],
                    }
            raise OSError("_ls_from_cache() failed to function")
        else:
            status, deet = await _async_wrap(self._myclient.stat)(path, self.timeout)
            if not status.ok:
                raise OSError(f"File stat request failed: {status.message}")
            if deet.flags & StatInfoFlags.IS_DIR:
                ret = {
                    "name": path,
                    "size": deet.size,
                    "type": "directory",
                }
            elif deet.flags & StatInfoFlags.OTHER:
                ret = {
                    "name": path,
                    "size": deet.size,
                    "type": "other",
                }
            else:
                ret = {
                    "name": path,
                    "size": deet.size,
                    "type": "file",
                }
            return ret

    async def _ls(self, path: str, detail: bool = True, **kwargs: Any) -> list[Any]:
        listing = []
        if path in self.dircache and not kwargs.get("force_update", False):
            if detail:
                listing = self._ls_from_cache(path)
                return listing
            else:
                return [
                    os.path.basename(item["name"]) for item in self._ls_from_cache(path)
                ]
        else:
            status, deets = await _async_wrap(self._myclient.dirlist)(  # type: ignore[var-annotated]
                path, DirListFlags.STAT, self.timeout
            )
            if not status.ok:
                raise OSError(
                    f"Server failed to provide directory info: {status.message}"
                )
            for item in deets:
                if item.statinfo.flags & StatInfoFlags.IS_DIR:
                    listing.append(
                        {
                            "name": path + "/" + item.name,
                            "size": item.statinfo.size,
                            "type": "directory",
                        }
                    )
                elif item.statinfo.flags & StatInfoFlags.OTHER:
                    listing.append(
                        {
                            "name": path + "/" + item.name,
                            "size": item.statinfo.size,
                            "type": "other",
                        }
                    )
                else:
                    listing.append(
                        {
                            "name": path + "/" + item.name,
                            "size": item.statinfo.size,
                            "type": "file",
                        }
                    )
            self.dircache[path] = listing
            if detail:
                return listing
            else:
                return [os.path.basename(item["name"].rstrip("/")) for item in listing]

    async def _cat_file(
        self, path: str, start: int | None, end: int | None, **kwargs: Any
    ) -> Any:
        _myFile = await self._readonly_filehandle_cache._open(
            self.unstrip_protocol(path),
            self.timeout,
        )
        n_bytes = end
        if start is not None and end is not None:
            n_bytes = end - start

        status, data = await _async_wrap(_myFile.read)(  # type: ignore[var-annotated]
            start or 0,
            n_bytes or 0,
            self.timeout,
        )
        if not status.ok:
            raise OSError(f"Bytes failed to read from open file: {status.message}")
        return data

    async def _get_file(
        self, rpath: str, lpath: str, chunk_size: int = 262_144, **kwargs: Any
    ) -> None:
        # Open the remote file for reading
        remote_file = await self._readonly_filehandle_cache._open(
            self.unstrip_protocol(rpath),
            self.timeout,
        )

        with open(lpath, "wb") as local_file:
            start: int = 0
            while True:
                # Read a chunk of content from the remote file
                status, chunk = await _async_wrap(remote_file.read)(  # type: ignore[var-annotated]
                    start, chunk_size, self.timeout
                )
                start += chunk_size

                if not status.ok:
                    raise OSError(f"Remote file failed to read: {status.message}")

                # Break if there is no more content
                if not chunk:
                    break

                # Write the chunk to the local file
                local_file.write(chunk)

    @classmethod
    async def _get_max_chunk_info(cls, file: Any) -> tuple[int, int]:
        """Queries the XRootD server for info required for pyxrootd vector_read() function.
        Queries for maximum number of chunks and the maximum chunk size allowed by the server.

        Parameters
        ----------
        file: xrootd client.File() object

        Returns
        -------
        Tuple of max chunk size and max number of chunks. Both ints.
        """
        data_server = file.get_property("DataServer")
        if data_server == "":
            return cls.default_max_num_chunks, cls.default_max_chunk_size
        # Normalize to URL
        data_server = client.URL(data_server)
        data_server = f"{data_server.protocol}://{data_server.hostid}/"
        if data_server not in cls._dataserver_info_cache:
            fs = client.FileSystem(data_server)
            status, result = await _async_wrap(fs.query)(  # type: ignore[var-annotated]
                QueryCode.CONFIG, "readv_iov_max readv_ior_max"
            )
            if not status.ok:
                raise OSError(
                    f"Server query for vector read info failed: {status.message}"
                )
            try:
                max_num_chunks, max_chunk_size = map(int, result.split(b"\n", 1))
            except ValueError:
                raise OSError(
                    f"Server query for vector read info failed: could not parse {result!r}"
                ) from None
            cls._dataserver_info_cache[data_server] = {
                "max_num_chunks": max_num_chunks,
                "max_chunk_size": max_chunk_size,
            }
        info = cls._dataserver_info_cache[data_server]
        return (info["max_num_chunks"], info["max_chunk_size"])

    async def _cat_vector_read(
        self,
        path: str,
        chunks: list[tuple[int, int]],
        batch_size: int,
    ) -> tuple[str, list[bytes]]:
        """Called by _cat_ranges() to vector read a file.

        Parameters
        ----------
        path: str, file path
        chunks: list of tuples, each in the form (start, end)
        batch_size: int, upper limit on simultainious vector reads

        Returns
        -------
        Tuple containing path name and a list of returned
        bytes in the same order as requested.
        """
        _myFile = await self._readonly_filehandle_cache._open(
            self.unstrip_protocol(path),
            self.timeout,
        )

        max_num_chunks, max_chunk_size = await self._get_max_chunk_info(_myFile)
        vectors = _chunks_to_vectors(chunks, max_num_chunks, max_chunk_size)

        coros = [_async_wrap(_myFile.vector_read)(v, self.timeout) for v in vectors]  # type: ignore[var-annotated]

        results = await _run_coros_in_chunks(coros, batch_size=batch_size, nofiles=True)
        result_bufs = []
        for status, buffers in results:
            if not status.ok:
                raise OSError(f"File did not vector_read properly: {status.message}")
            result_bufs.append(buffers)
        deets = _vectors_to_chunks(chunks, result_bufs)

        return (path, deets)

    async def _cat_ranges(
        self,
        paths: list[str],
        starts: list[int],
        ends: list[int],
        max_gap: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[bytes]:
        # TODO: on_error
        if max_gap is not None:
            # use utils.merge_offset_ranges
            raise NotImplementedError
        if not isinstance(paths, list):
            raise TypeError
        if not isinstance(starts, list):
            raise TypeError
        if not isinstance(ends, list):
            raise TypeError
        if len(starts) != len(paths) or len(ends) != len(paths):
            raise ValueError

        uniquePaths = defaultdict(list)

        for path, start, end in zip(paths, starts, ends):
            uniquePaths[path].append((start, end))

        batch_size = batch_size or self.batch_size

        coros = [
            self._cat_vector_read(key, uniquePaths[key], batch_size)
            for key in uniquePaths.keys()
        ]

        results = await _run_coros_in_chunks(coros, batch_size=batch_size, nofiles=True)

        resDict = dict(results)

        deets = [resDict[path].pop(0) for path in paths]

        return deets

    async def open_async(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        if "b" not in mode or kwargs.get("compression"):
            raise NotImplementedError
        return self.open(path, mode, kwargs=kwargs)

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | str | None = None,
        autocommit: bool = True,
        cache_options: dict[Any, Any] | None = None,
        **kwargs: Any,
    ) -> XRootDFile:
        return XRootDFile(
            self,
            path,
            mode,
            block_size,
            autocommit,
            cache_options=cache_options,
            **kwargs,
        )

    def open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | int | None = None,
        cache_options: dict[Any, Any] | None = None,
        compression: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Returns text wrapper or XRootDFile."""

        path = self._strip_protocol(path)
        if "b" not in mode:
            mode = mode.replace("t", "") + "b"

            text_kwargs = {
                k: kwargs.pop(k)
                for k in ["encoding", "errors", "newline"]
                if k in kwargs
            }
            return io.TextIOWrapper(
                self.open(
                    path,
                    mode,
                    block_size=block_size,
                    cache_options=cache_options,
                    compression=compression,
                    **kwargs,
                ),
                **text_kwargs,
            )
        else:
            ac = kwargs.pop("autocommit", not self._intrans)
            f = self._open(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=ac,
                cache_options=cache_options,
                **kwargs,
            )
            if compression is not None:
                from fsspec.compression import compr
                from fsspec.core import get_compression

                compression = get_compression(path, compression)
                compress = compr[compression]
                f = compress(f, mode=mode[0])

            if not ac and "r" not in mode:
                self.transaction.files.append(f)
            return f


class XRootDFile(AbstractBufferedFile):  # type: ignore[misc]
    def __init__(
        self,
        fs: XRootDFileSystem,
        path: str,
        mode: str = "rb",
        block_size: int | str | None = "default",
        autocommit: bool = True,
        cache_type: str = "readahead",
        cache_options: dict[Any, Any] | None = None,
        size: int | None = None,
        **kwargs: dict[Any, Any],
    ) -> None:
        from fsspec.core import caches

        self.timeout = fs.timeout
        # by this point, mode will have a "b" in it
        # update "+" mode removed for now since seek() is read only
        if "x" in mode:
            self.mode = OpenFlags.NEW
        elif "a" in mode:
            self.mode = OpenFlags.UPDATE
        elif "w" in mode:
            self.mode = OpenFlags.DELETE
        elif "r" in mode:
            self.mode = OpenFlags.READ
        else:
            raise NotImplementedError

        if not isinstance(path, str):
            raise ValueError(f"Path expected to be string, path: {path}")

        # Ensure any read-only handle is closed
        fs.invalidate_cache(path)
        self._myFile = client.File()
        status, _ = self._myFile.open(
            fs.unstrip_protocol(path),
            self.mode,
            timeout=self.timeout,
        )
        if not status.ok:
            raise OSError(f"File did not open properly: {status.message}")

        self.metaOffset = 0
        if "a" in mode:
            _stats, _deets = self._myFile.stat(timeout=self.timeout)
            self.metaOffset = _deets.size

        self.path = path
        self.fs = fs
        self.mode = mode
        self.blocksize = (
            self.DEFAULT_BLOCK_SIZE if block_size in ["default", None] else block_size
        )
        self.loc = 0
        self.autocommit = autocommit
        self.end = None
        self.start = None
        self.closed = False

        if cache_options is None:
            cache_options = {}

        if "trim" in kwargs:
            warnings.warn(
                "Passing 'trim' to control the cache behavior has been"
                " deprecated. "
                "Specify it within the 'cache_options' argument instead.",
                FutureWarning,
                stacklevel=1,
            )
            cache_options["trim"] = kwargs.pop("trim")

        self.kwargs = kwargs

        if mode not in {"ab", "rb", "wb"}:
            raise NotImplementedError("File mode not supported")
        if mode == "rb":
            if size is not None:
                self.size = size
            else:
                self.size = self.details["size"]
            self.cache = caches[cache_type](
                self.blocksize, self._fetch_range, self.size, **cache_options
            )
        else:
            self.buffer = io.BytesIO()
            self.forced = False
            self.location = None
            self.offset = 0

    def _fetch_range(self, start: int, end: int) -> Any:
        status, data = self._myFile.read(
            self.metaOffset + start, self.metaOffset + end - start, timeout=self.timeout
        )
        if not status.ok:
            raise OSError(f"File did not read properly: {status.message}")
        return data

    def flush(self, force: bool = False) -> None:
        if self.closed:
            raise ValueError("Flush on closed file")
        if force and self.forced:
            raise ValueError("Force flush cannot be called more than once")
        if force:
            self.forced = True

        if self.mode not in {"wb", "ab"}:
            # no-op to flush on read-mode
            return

        if not force and self.buffer.tell() < self.blocksize:
            # Defer write on small block
            return

        if self._upload_chunk(final=force) is not False:
            self.offset += self.buffer.seek(0, 2)
            self.buffer = io.BytesIO()

    def _upload_chunk(self, final: bool = False) -> Any:
        status, _n = self._myFile.write(
            self.buffer.getvalue(),
            self.offset + self.metaOffset,
            self.buffer.tell(),
            timeout=self.timeout,
        )
        if final:
            self.closed
            self.close()
        if not status.ok:
            raise OSError(f"File did not write properly: {status.message}")
        return status.ok

    def close(self) -> None:
        if getattr(self, "_unclosable", False):
            return
        if self.closed or not self._myFile.is_open():
            return
        if self.mode == "rb":
            self.cache = None
        else:
            if not self.forced:
                self.flush(force=True)

            if self.fs is not None:
                self.fs.invalidate_cache(self.path)
                self.fs.invalidate_cache(self.fs._parent(self.path))
        status, _n = self._myFile.close(timeout=self.timeout)
        if not status.ok:
            raise OSError(f"File did not close properly: {status.message}")
        self.closed = True
