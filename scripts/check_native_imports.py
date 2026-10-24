"""Import all six compiled modules; does not run GPU parity or fetch models."""
import importlib
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src/imaginaire/model_utils/gancraft/voxlib"))
for name in ("correlation_cuda", "channelnorm_cuda", "resample2d_cuda",
             "bias_act_cuda", "upfirdn2d_cuda", "voxlib"):
    importlib.import_module(name)
    print("Imported", name)
