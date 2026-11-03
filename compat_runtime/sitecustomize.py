"""Normalize this host's non-PEP-440 Python patch version for dependencies."""

import platform


_python_version = platform.python_version
if _python_version().endswith("+"):
    platform.python_version = lambda: _python_version().rstrip("+")
