"""
Copyright (c) 2022 Scott Demarest. All rights reserved.

fsspec-xrootd: xrootd implementation for fsspec
"""


from __future__ import annotations

from ._version import version as __version__
from .xrootd import XRootDFile, XRootDFileSystem

__all__ = ("__version__", "XRootDFileSystem", "XRootDFile")

try:
    import XRootD
    import XRootD.client

except ModuleNotFoundError:
    raise ModuleNotFoundError(
        """Install XRootD python bindings with:
conda install -c conda-forge xrootd
(or download from http://xrootd.org/dload.html and manually compile with """
        """cmake; setting PYTHONPATH and LD_LIBRARY_PATH appropriately)."""
    )
