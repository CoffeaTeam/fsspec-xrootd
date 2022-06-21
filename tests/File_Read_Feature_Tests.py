from __future__ import annotations

import xrootd as xrtd
from XRootD import client

url = "PLACE HOLDER"  # use url of git hub xrootd server

kw = xrtd.XRootDFileSystem._get_kwargs_from_urls(url)

url = client.URL(url)

refDict = {  # dict to check kw against
    "hostid": url.hostid,
    "protocol": url.protocol,
    "username": url.username,
    "password": url.password,
    "hostname": url.hostname,
    "port": url.port,
    "path": url.path,
    "path_with_params": url.path_with_params,
}

if not (refDict == kw):
    raise Exception("kwargs from url error")

fs = xrtd.XRootDFileSystem(**kw)

# adjust seek and read positions for final tests
s = 0
r = 10

refDat = "PLACE HOLDER"

with fs.open(kw["path"], "rt") as f:
    f.seek(s)
    dat = f.read(r)
    if not (refDat == dat):
        raise Exception("Read data doesn't match expected")
    f.seek(-1)
    assert f.read() == ""
