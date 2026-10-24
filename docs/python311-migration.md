# Isolated Python 3.11 / CUDA 11.8 migration candidate

**Draft only: do not merge, deploy, replace the existing environment, or infer
GPU readiness from the CPU results below.** This branch starts at
`cddcda13f2fb3770a3eb96df1a71dd30ced83416`, after the separately approved
Pillow and Requests/tqdm compatibility PRs were normally merged. The unrelated
SciPy Dependabot PR #3 is not merged or reused as a security fix.

## Finite security scope and actual alert accounting

The authenticated 2026-10-24 pre-merge snapshot contained 39 actual alerts in
root `requirements.txt`: 34 Pillow, four Requests, one tqdm. The merged Python
3.8 pins avoid 16 reported ranges (#2–8, #11, #12, #14–16, #9, #18–20).
The remaining **23** ranges are the reason for this coordinated runtime branch:

| Package / actual alert IDs | First patched version | Python floor |
| --- | --- | --- |
| Pillow #22–23 | 12.2.0 | >=3.10 |
| Pillow #24–35 | 12.3.0 | >=3.10 |
| Pillow #36–43 | 13.0.0 | >=3.11 |
| Requests #21 | 2.33.0 | >=3.10 |

The candidate pins Pillow 13.0.0, Requests 2.33.0 and preserves tqdm 4.66.3.
They avoid all 39 **snapshot version ranges**, not a general repository or
transitive-dependency security audit. No exploitability conclusion is inferred.
A draft branch closes no default-branch alerts. Even after the first two merges,
the actual endpoint still reported 39 open at 21:49 UTC; the dynamic Dependabot
graph-update run completed successfully and submitted the correct 10.4.0 /
2.32.4 / 4.66.3 versions. **No alert is counted fixed without a fixed-state
receipt.** Scanner lag is not silently substituted for such a receipt.

## Coordinated candidate and primary support evidence

| Component | Candidate |
| --- | --- |
| Python | 3.11.17 |
| Container | CUDA 11.8.0 devel / Ubuntu 22.04; GCC/G++ 11 |
| torch / torchvision / torchaudio | 2.7.1 / 0.22.1 / 2.7.1, all cu118 |
| Pillow / Requests / tqdm | 13.0.0 / 2.33.0 / 4.66.3 |
| NumPy / SciPy | 1.26.4 / 1.15.3 |
| Albumentations / OpenCV | 1.3.1 / opencv-python-headless 4.11.0.86 |
| scikit-image / imageio | 0.25.2 / 2.37.0 |

- [PyTorch's official tuple/index commands](https://pytorch.org/get-started/previous-versions/#v271)
  include cu118 and CPU variants; the [official 2.7.1 builder](https://github.com/pytorch/pytorch/blob/v2.7.1/.ci/manywheel/build_cuda.sh)
  includes sm70 for cu118. This is necessary metadata, not a V100 execution test.
- [Pillow 13 metadata](https://pypi.org/pypi/pillow/13.0.0/json),
  [Requests 2.33 metadata](https://pypi.org/pypi/requests/2.33.0/json) and
  [SciPy 1.15.3 runtime requirements](https://github.com/scipy/scipy/blob/v1.15.3/pyproject.toml)
  support the chosen Python/NumPy anchors. Albumentations 1.3.1 retains the
  actual legacy augmentation signatures; unpinned 2.x does not.
- Python is built from the [official PSF 3.11.17 source](https://www.python.org/downloads/release/python-31117/),
  SHA256 `bfb74ad39efae27cda510f134ab408e00f9992c56851cfc0b1cdb5646da11599`.
  The candidate does not replace Ubuntu's system Python or add a PPA.
- The historical README driver 460.119.04 is **not a full CUDA 11.8 validation**.
  [NVIDIA's 11.8 release notes](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html)
  list 520.61.05 for the corresponding Linux toolkit driver. The lower
  450.80.02 minor-compatibility floor has restrictions and is not a blanket
  full-feature guarantee. No host driver is changed by this branch.

## Necessary runtime/source changes

- Five extension setups and voxlib use C++17. Correlation/channelnorm dispatch
  uses `scalar_type()`, the three FlowNet kernels use `data_ptr<T>()`, and
  voxlib uses the current `C10_CUDA_CHECK` declaration. These are source
  adaptations awaiting six real CUDA builds, not proof those builds pass.
- NumPy aliases become explicit `int64`, `float64` and `bool_`; FlowNet uses
  `imageio.v2.imread`, `inspect.getfullargspec` and `time.perf_counter`.
- All seven source checkpoint loaders explicitly use `weights_only=True`.
  There is no unrestricted-pickle fallback. Synthetic state/tensor formats
  pass; existing external checkpoints must still be verified individually.
- The resolved closure exposed overlapping `cv2` providers. The root now has
  just `opencv-python-headless`; the unused `face-alignment` root was removed
  because its dependency reinstated `opencv-python`. No import/use was found
  in tracked repository source. Users of external face-alignment scripts need
  a separately isolated/validated environment; do not co-install cv2 variants.
  Likewise the duplicate `pynvml`/old `nvidia-ml-py3` providers are replaced by
  the single maintained NVIDIA `nvidia-ml-py` distribution, which exposes the
  same `pynvml` module. These changes are closure compatibility work, not a
  claim of a new independently mapped security fix.

## Locks: what was and was not established

`requirements-python311-cu118.lock` resolves **192** packages for Linux x86-64 /
Python 3.11, records published artifact hashes, and pins the three `+cu118`
versions. It was generated from the complete root requirements with uv's
explicit per-package PyTorch backend routing. The complete CUDA closure was
**not downloaded or installed**. Source distributions (for example dlib) still
need real native builds; build-backend/system packages and the mutable CUDA
image tag are not an independently verified fully reproducible image.

The CUDA lock also passed an explicit-backend, hash-required **dry-run** resolve;
it would install 157 packages, which were not downloaded/installed.

`requirements-cpu-test.lock` is a separate **46-package**, hash-checked finite
validation environment, not the full application's requirements. It was synced
and `uv pip check` passed. Both locks contain exactly one OpenCV distribution.
Do not substitute CPU wheels into the CUDA lock or pass a bare extra index to
uv: its first-index protection can correctly reject unrelated pinned packages.
Use the explicit backend selector below; no unsafe index strategy is needed.

## Reproduce bounded CPU verification (no models or GPU)

With a Python 3.11 interpreter and uv available:

```sh
uv venv --python python3.11 .venv-cpu
uv pip sync --python .venv-cpu/bin/python --require-hashes \
  --torch-backend cpu --index-url https://pypi.org/simple requirements-cpu-test.lock
uv pip check --python .venv-cpu/bin/python
mkdir -p .cache/matplotlib .cache/xdg
MPLCONFIGDIR=$PWD/.cache/matplotlib XDG_CACHE_HOME=$PWD/.cache/xdg \
  OMP_NUM_THREADS=2 .venv-cpu/bin/python -m unittest discover -s tests -v
.venv-cpu/bin/python scripts/check_native_host.py
```

Recorded on CPython **3.11.17** (isolated uv-managed Astral build, not the PSF
source-build Docker image) and the official **CPU** torch tuple:

- **31 tracked tests passed**: the retained 13 image tests and seven
  Requests/tqdm tests, plus 11 migration tests. Real torch/vision/audio imports,
  real tensor image helpers, torchvision CPU NMS and audio resampling ran.
  Real Albumentations image/mask/keypoint crop paths and FlowNet's timer,
  introspection and RGB/RGBA/array readers ran without model downloads.
- The actual trainer save/load **method bodies** were AST-isolated, with the
  distributed `master_only` decorator replaced by an identity in the harness;
  real tiny torch networks, Adam state and StepLR state round-tripped. This
  is not a full trainer import/startup or training test. An inert unsupported
  object is rejected in restricted mode.
- The seven HTTP tests also passed separately with urllib3 **1.26.20** and
  **2.2.3**, then the exact CPU lock was restored and all 31 tests re-passed.
  The old urllib3 versions are test points, not recommended runtime pins.
- Both actual channelnorm/resample host wrappers passed C++17 syntax checks
  with local GCC 14.2 and CPU torch headers. A tiny linked CPU probe exercised
  `scalar_type()` and `data_ptr()`. This is **not** CUDA compilation/linking or
  a GCC14/CUDA11.8 support claim; the candidate container selects GCC11.
- Python3.11 AST parsing passed for all 168 source/test/script files checked
  before publication. Shell syntax and diff
  whitespace checks pass on the final candidate.

Failures were retained, not relabeled as passes: the first unchanged tests
failed the old Requests pin and removed `ImageMath.eval`; the initial adaptation
also used the old positional environment signature and omitted the isolated
trainer decorator. Final tests use the correct keyword-only `unsafe_eval` API only with fixed,
trusted literal expressions; it must never accept external expression strings
and is not a safe evaluator for untrusted input. The trainer harness boundary
is explicit. An all-wheel-only resolve failed on the legacy
source-only NVIDIA provider; a metadata-capable solve and single-provider
closure repaired that issue. The first CPU lock sync correctly failed on
ambiguous extra-index routing; explicit `--torch-backend cpu` succeeded.
Albumentations still emits its existing ReplayMode warning for target-dependent
random crop; the synthetic crop passes, but broader replay parity is unproven.

## Required next gates — no automatic merge or deployment

1. In an **existing authorized** CUDA11.8/GCC11 environment, build the candidate
   Docker image and its source distributions; verify complete `pip check`,
   real runtime imports and all six native extension imports. Docker/nvcc/GPU
   were absent from this worker, so this gate is unrun, not failed or passed.
2. Execute small CUDA operator forward/backward/reference parity fixtures for
   all six modules, including V100/sm70 if that host remains the target. Check
   the actual driver/toolkit/ABI rather than the historical README numbers.
3. Validate the provenance and restricted-load compatibility of existing,
   trusted checkpoints without downloading new ones automatically. Compare
   inference pixels/flow on consented fixtures and test applicable training
   startup/optimizer resume. Historical 23GB VRAM/60GB RAM figures are not new
   resource measurements or a promise that a 24GB GPU suffices.
4. Review source-distribution builds, remaining unpinned-source/API behavior,
   the image digest and external script needs. A metadata solve and CPU subset
   do not establish full application compatibility or future security.
5. Keep this PR **draft**. Any later ready/merge/deploy step needs its own
   approval; after a permitted merge, read actual default-branch alerts before
   counting closures. No force push, settings change, release/tag, deployment,
   new login, paid GPU, model download or host-driver change was performed.
