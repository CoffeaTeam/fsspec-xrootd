import xrootd as xrtd

url = 'root://localhost//tmp/pytest-of-scott/pytest-5/srv0'

kw = xrtd.XRootDFileSystem._get_kwargs_from_urls(url)

print(kw["path"])