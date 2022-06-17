# Implementations of AbstractFileSystem and AbstractBufferedFile

from fsspec.spec import AbstractFileSystem, AbstractBufferedFile
from XRootD import client
from XRootD.client.flags import OpenFlags, StatInfoFlags, DirListFlags
import warnings
import io


class XRootDFileSystem(AbstractFileSystem):

    # unpack storage_options as necessary
    def __init__(self, *args, **storage_options):
        self._path = storage_options["path"]
        self._myclient = client.FileSystem(storage_options["protocol"] + "://"
                                           + storage_options["hostid"])
        self.storage_options = storage_options
        self._intrans = False

    @staticmethod
    def _get_kwargs_from_urls(u):

        url = client.URL(u)

        return {"hostid": url.hostid, "protocol": url.protocol,
                "username": url.username, "password": url.password,
                "hostname": url.hostname, "port": url.port,
                "path": url.path, "path_with_params": url.path_with_params}

    # Implement a new _strip_protocol?

    def ls(self, path, detail=True, **kwargs):

        stats, deets = self._myclient.dirlist(path, DirListFlags.STAT)

        listing = []

        if detail:
            for item in deets:
                t = ""
                if item.statinfo.flags and StatInfoFlags.IS_DIR:
                    t = "directory"
                elif item.statinfo.flags and StatInfoFlags.OTHER:
                    t = "other"
                else:
                    t = "file"

                listing.append({"name": path+"/"+item.name,
                                "size": item.statinfo.size, "type": t})
        else:
            for item in deets:
                listing.append(item.name)

        return listing

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):

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
        path,
        mode="rb",
        block_size=None,
        cache_options=None,
        compression=None,
        **kwargs,
    ):

        import io

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


class XRootDFile(AbstractBufferedFile):

    def __init__(
        self,
        fs,
        path,
        mode="rb",
        block_size="default",
        autocommit=True,
        cache_type="readahead",
        cache_options=None,
        size=None,
        **kwargs,
    ):
        from fsspec.core import caches

        # by this point, mode will have a "b" in it
        if "x" in mode:
            self.mode = OpenFlags.NEW
        elif "a" in mode:
            self.mode = OpenFlags.APPEND
        elif "+" in mode:
            self.mode = OpenFlags.UPDATE
        elif "w" in mode:
            self.mode = OpenFlags.DELETE
        elif "r" in mode:
            self.mode = OpenFlags.READ
        else:
            raise NotImplementedError

        self._myFile = client.File()
        stat, _n = self._myFile.open(fs.storage_options["protocol"]
                                     + "://" + fs.storage_options["hostid"]
                                     + "/" + path, self.mode)

        if not stat.ok:
            print(stat.message)
            raise IOError

        self.path = path
        self.fs = fs
        self.mode = mode
        self.blocksize = (
            self.DEFAULT_BLOCK_SIZE if block_size in ["default", None]
            else block_size
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
            self.offset = None
            self.forced = False
            self.location = None

    def _fetch_range(self, start, end):
        status, data = self._myFile.read(start, end-start)
        return data

    def close(self):
        print("Closed!")
        if getattr(self, "_unclosable", False):
            return
        if self.closed:
            return
        if self.mode == "rb":
            self.cache = None
        else:
            if not self.forced:
                self.flush(force=True)

            if self.fs is not None:
                self.fs.invalidate_cache(self.path)
                self.fs.invalidate_cache(self.fs._parent(self.path))
        self._myFile.close()
        self.closed = True
