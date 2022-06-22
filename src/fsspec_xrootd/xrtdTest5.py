import xrootd as xrtd

url = "root://anon@PortaVac.localdomain:1094//tmp/WriteTests/file1"

kw = xrtd.XRootDFileSystem._get_kwargs_from_urls(url)

fs = xrtd.XRootDFileSystem(**kw)

#print(kw["path"])

with fs.open(kw["path"], "rt") as f:
    dat = f.read()
    print(dat)

with fs.open(kw["path"], "wt", 5) as f:
    len = f.write("Write test")

with fs.open(kw["path"], "rt") as f:
    dat = f.read()
    print(dat)

with fs.open(kw["path"], "at", 50) as f:
    len = f.write("Append test")

with fs.open(kw["path"], "rt") as f:
    dat = f.read()
    print(dat)