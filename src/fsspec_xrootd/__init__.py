"""
Copyright (c) 2022 Scott Demarest. All rights reserved.

fsspec-xrootd: xrootd implementation for fsspec
"""


from __future__ import annotations

from ._version import version as __version__
from .xrootd import XRootDFile, XRootDFileSystem

__all__ = ("__version__", "XRootDFileSystem", "XRootDFile")
