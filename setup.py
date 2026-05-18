from __future__ import annotations

import os
import sys
from setuptools import setup, Extension

try:
    import pybind11
except ImportError:
    pybind11 = None


def _eigen_include() -> str:
    env = os.environ.get("EIGEN_INCLUDE_DIR")
    if env:
        return env
    candidates = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/homebrew/include/eigen3",
        "C:/vcpkg/installed/x64-windows/include/eigen3",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    sys.stderr.write(
        "WARNING: Eigen headers not found. Set EIGEN_INCLUDE_DIR to your "
        "Eigen prefix (the directory containing 'Eigen/Dense').\n"
    )
    return ""


ext_modules = []
if pybind11 is not None:
    ext_modules.append(Extension(
        "rpo._ekf_cpp",
        sources=["cpp/src/pybindings.cpp"],
        include_dirs=[
            "cpp/include",
            pybind11.get_include(),
            _eigen_include(),
        ],
        language="c++",
        extra_compile_args=(
            ["/std:c++17", "/O2"]
            if sys.platform == "win32"
            else ["-std=c++17", "-O2", "-Wall"]
        ),
    ))

setup(ext_modules=ext_modules)
