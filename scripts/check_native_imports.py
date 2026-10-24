"""Import all six compiled modules; does not run GPU parity or fetch models."""
import importlib
from pathlib import Path
import sys

# Load libtorch before its compiled extensions; this does not execute GPU ops.
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src/imaginaire/model_utils/gancraft/voxlib"))
for directory in ("correlation", "channelnorm", "resample2d", "bias_act", "upfirdn2d"):
    sys.path.insert(0, str(root / "src/imaginaire/third_party" / directory))
for name in ("correlation_cuda", "channelnorm_cuda", "resample2d_cuda",
             "bias_act_cuda", "upfirdn2d_cuda", "voxlib"):
    importlib.import_module(name)
    print("Imported", name)
