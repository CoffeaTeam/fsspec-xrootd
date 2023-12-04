from __future__ import annotations

import os
import shutil
import subprocess
import time
import pytest


@pytest.fixture(scope="module")
def available_port():
    """
    Get an available port on localhost.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind to port 0 to let the OS choose an available port
        s.bind(('localhost', 0))
        # Get the selected port
        _, port = s.getsockname()
    return port


@pytest.fixture(scope="module")
def server(tmpdir_factory, available_port):
    srvdir = tmpdir_factory.mktemp("srv")
    tmp_path = os.path.join(srvdir, "Folder")
    os.mkdir(tmp_path)
    xrootd_executable = shutil.which("xrootd")
    proc = subprocess.Popen([xrootd_executable, "-p", str(available_port), str(srvdir)])
    time.sleep(2)  # give it some startup
    yield f"root://localhost:{available_port}/{tmp_path}", tmp_path
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture()
def clear_server(server):
    url, path = server
    shutil.rmtree(path)
    os.mkdir(path)
    yield
