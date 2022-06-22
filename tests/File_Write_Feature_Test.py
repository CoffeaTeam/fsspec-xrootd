from __future__ import annotations

import xrootd as xrtd
from XRootD import client

url = "root://anon@PortaVac.localdomain:1094//tmp/TEST3/eggs5"

kw = xrtd.XRootDFileSystem._get_kwargs_from_urls(url)

fs = xrtd.XRootDFileSystem(**kw)

with fs.open(kw["path"], "rt") as f:
    dat = f.read(4)
    print(dat)

with fs.open(kw["path"], "wt") as f:
    len = f.write("lego my ego")

with fs.open(kw["path"], "rt") as f:
    dat = f.read(10)
    print(dat)
