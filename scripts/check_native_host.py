"""Bounded real host-wrapper syntax/API checks; NOT a CUDA extension build.

Use the Python3.11 CPU validation environment. g++ and torch headers are needed.
CUDA SDK, linking the six native modules and GPU operator tests are separate gates.
"""
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile

import torch
from torch.utils.cpp_extension import include_paths

ROOT = Path(__file__).resolve().parents[1]
compiler = os.environ.get("CXX", "g++")
common = [compiler, "-std=c++17", "-O0",
          "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
          *["-I" + path for path in include_paths()],
          "-I" + sysconfig.get_path("include")]
for name in ("channelnorm", "resample2d"):
    source = ROOT / "src/imaginaire/third_party" / name / "src" / (name + "_cuda.cc")
    subprocess.run(common + ["-fsyntax-only", "-DTORCH_API_INCLUDE_EXTENSION_H",
                             "-DTORCH_EXTENSION_NAME=" + name + "_cuda", str(source)],
                   check=True, timeout=150)
    print("Host syntax passed:", name, flush=True)
with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / "native-api-smoke"
    library_dir = Path(torch.__file__).parent / "lib"
    subprocess.run(common + [str(ROOT / "tests/native_api_smoke.cpp"),
                             "-L" + str(library_dir), "-ltorch_cpu", "-lc10",
                             "-Wl,-rpath," + str(library_dir), "-o", str(output)],
                   check=True, timeout=150)
    subprocess.run([str(output)], check=True, timeout=10)
    print("CPU scalar_type/data_ptr API probe passed; no CUDA claim.", flush=True)
