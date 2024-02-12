"""fsspec-xrootd

An xrootd implementation for fsspec.

BSD 3-Clause License; see https://github.com/CoffeaTeam/fsspec-xrootd/blob/main/LICENSE
"""

from __future__ import annotations

from ._version import version as __version__
from .xrootd import XRootDFile, XRootDFileSystem

__all__ = ("__version__", "XRootDFileSystem", "XRootDFile")
