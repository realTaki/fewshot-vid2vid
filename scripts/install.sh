#!/bin/sh
# CUDA/Python migration candidate. Run only in the matching isolated container.
set -eu
CURRENT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$CURRENT"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
command -v nvcc >/dev/null
nvcc --version | grep -q 'release 11.8,'
export CUDA_VERSION=11.8
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6+PTX}"
export MAX_JOBS="${MAX_JOBS:-2}"

# Exact local-version torch pins + hashes prevent a CPU/CUDA backend substitution.
python -m pip install --require-hashes \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -r requirements-python311-cu118.lock
python -m pip check
python -c 'import torch, torchvision, torchaudio; assert torch.version.cuda == "11.8"; assert torch.__version__ == "2.7.1+cu118"'

for p in correlation channelnorm resample2d bias_act upfirdn2d; do
    (cd "src/imaginaire/third_party/$p" && python setup.py build_ext --inplace && python setup.py install)
done
(cd src/imaginaire/model_utils/gancraft/voxlib && make all)
python scripts/check_native_imports.py
