from __future__ import annotations

import io
import os.path
import warnings
from enum import IntEnum
from typing import Any

from fsspec.dircache import DirCache  # type: ignore[import]
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem  # type: ignore[import]
from XRootD import client  # type: ignore[import]
from XRootD.client.flags import (  # type: ignore[import]
    DirListFlags,
    MkDirFlags,
    OpenFlags,
    StatInfoFlags,
)


class ErrorCodes(IntEnum):
    INVALID_PATH = 400


class XRootDFileSystem(AbstractFileSystem):  # type: ignore[misc]

    protocol = "root"
    root_marker = "/"
    default_timeout = 60

    def __init__(self, *args: list[Any], **storage_options: Any) -> None:
        self.timeout = storage_options.get("timeout", XRootDFileSystem.default_timeout)
        self._path = storage_options["path"]
        self._myclient = client.FileSystem(
            storage_options["protocol"] + "://" + storage_options["hostid"]
        )
        status, _n = self._myclient.ping(15)
        if not status.ok:
            raise OSError(f"Could not connect to server {storage_options['hostid']}")
        self._intrans = False
        self.dircache = DirCache(
            use_listings_cache=True,
            listings_expiry_time=storage_options.get("listings_expiry_time", 0),
        )
        self.storage_options = storage_options

    def invalidate_cache(self, path: str | None = None) -> None:
        if path is None:
            self.dircache.clear()
        else:
            try:
                del self.dircache[path]
            except KeyError:
                pass

    @staticmethod
    def _get_kwargs_from_urls(u: str) -> dict[Any, Any]:

        url = client.URL(u)

        return {
            "hostid": url.hostid,
            "protocol": url.protocol,
            "username": url.username,
            "password": url.password,
            "hostname": url.hostname,
            "port": url.port,
            "path": url.path,
            "path_with_params": url.path_with_params,
        }

    @classmethod
    def _strip_protocol(cls, path: str | list[str]) -> Any:
        if type(path) == str:
            return client.URL(path).path
        elif type(path) == list:
            return [client.URL(item).path for item in path]
        else:
            raise ValueError("Strip protocol not given string or list")

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        if create_parents:
            status, n = self._myclient.mkdir(
                path, MkDirFlags.MAKEPATH, timeout=self.timeout
            )
        else:
            status, n = self._myclient.mkdir(path, timeout=self.timeout)
        if not status.ok:
            raise OSError(f"Directory not made properly: {status.message}")

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        if not exist_ok:
            if self.exists(path):
                raise OSError(
                    "Location already exists and exist_ok arg was set to false"
                )
        status, n = self._myclient.mkdir(
            path, MkDirFlags.MAKEPATH, timeout=self.timeout
        )
        if not status.ok and not (status.code == ErrorCodes.INVALID_PATH and exist_ok):
            raise OSError(f"Directory not made properly: {status.message}")

    def rmdir(self, path: str) -> None:
        status, n = self._myclient.rmdir(path, timeout=self.timeout)
        if not status.ok:
            raise OSError(f"Directory not removed properly: {status.message}")

    def _rm(self, path: str) -> None:
        status, n = self._myclient.rm(path, timeout=self.timeout)
        if not status.ok:
            raise OSError(f"File not removed properly: {status.message}")

    def touch(self, path: str, truncate: bool = False, **kwargs: Any) -> None:
        if truncate or not self.exists(path):
            status, _ = self._myclient.truncate(path, 0, timeout=self.timeout)
            if not status.ok:
                raise OSError(f"File not touched properly: {status.message}")
        else:
            status, _ = self._myclient.truncate(
                path, self.info(path).get("size"), timeout=self.timeout
            )
            if not status.ok:
                raise OSError(f"File not touched properly: {status.message}")

    def modified(self, path: str) -> Any:
        status, statInfo = self._myclient.stat(path, timeout=self.timeout)
        return statInfo.modtime

    def exists(self, path: str, **kwargs: Any) -> bool:
        if path in self.dircache:
            return True
        else:
            status, _ = self._myclient.stat(path, timeout=self.timeout)
            if status.code == ErrorCodes.INVALID_PATH:
                return False
            elif not status.ok:
                raise OSError(f"status check failed with message: {status.message}")
            return True

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        spath = os.path.split(path)
        deet = self._ls_from_cache(spath[0])
        if deet is not None:
            for item in deet:
                if item["name"] == path:
                    return {
                        "name": path,
                        "size": item["size"],
                        "type": item["type"],
                    }
            raise OSError("_ls_from_cache() failed to function")
        else:
            status, deet = self._myclient.stat(path, timeout=self.timeout)
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

    def ls(self, path: str, detail: bool = True, **kwargs: Any) -> list[Any]:
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
            status, deets = self._myclient.dirlist(
                path, DirListFlags.STAT, timeout=self.timeout
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
                from fsspec.compression import compr  # type: ignore[import]
                from fsspec.core import get_compression  # type: ignore[import]

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
        self._myFile = client.File()
        status, _n = self._myFile.open(
            fs.storage_options["protocol"]
            + "://"
            + fs.storage_options["hostid"]
            + "/"
            + path,
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
