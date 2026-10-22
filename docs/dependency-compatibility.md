# Requests and tqdm compatibility patch

This change retains the README's tested host environment (Python 3.8.10 and
CUDA 11.4) and the separate container/install recipe (CUDA 11.1.1,
PyTorch 1.9.1+cu111, torchvision 0.10.1+cu111, and torchaudio 0.9.1). It does not
change either configuration, upgrade the ML stack, or claim to make the whole
historical environment secure or reproducible.

## Selected pins

- `requests==2.32.4` supports Python >=3.8 and includes the fixes for
  [HTTPS proxy credential forwarding](https://github.com/psf/requests/security/advisories/GHSA-j8r2-6x86-q33q),
  [TLS verification reuse](https://github.com/psf/requests/security/advisories/GHSA-9wx4-h78v-vm56),
  and [malicious-URL .netrc disclosure](https://github.com/psf/requests/security/advisories/GHSA-9hjg-9r4m-mvj7).
  This extends the existing Requests Dependabot PR rather than replacing its
  original commit. [Release metadata](https://pypi.org/pypi/requests/2.32.4/json).
- `tqdm==4.66.3` supports Python >=3.7 and fixes
  [CLI argument evaluation](https://github.com/tqdm/tqdm/security/advisories/GHSA-g7vv-2v7x-gj9p).
  [Release metadata](https://pypi.org/pypi/tqdm/4.66.3/json).

## Known residuals and separate decisions

- Requests 2.32.4 is still listed for CVE-2026-25645 /
  [GHSA-gc5v-m9x4-r6x2](https://github.com/psf/requests/security/advisories/GHSA-gc5v-m9x4-r6x2),
  involving direct `extract_zipped_paths()` calls and predictable temporary
  files. No direct call was found in this repository's current source. This is
  an exposure observation, not an all-clear. The fixed Requests 2.33.0 requires
  Python >=3.10; Requests 2.32.5 already requires >=3.9. Raising the project runtime
  needs a separate compatibility decision, not an incidental dependency bump.
- Requests' transitive dependencies are not locked here. The isolated smoke-test
  closure is not a deployment lock or a clean security audit; in particular the
  Python-3.8-compatible urllib3 line needs separate security/runtime review.
- Pillow and SciPy remain under their existing separate Dependabot PRs. Many
  other requirements are unpinned; the complete CUDA build, inference, training,
  model checkpoints, and datasets were not validated by this change.
- Current GitHub Dependabot alert counts could not be re-read through the known
  authentication gate. Historical counts must not be reported as current or zero.

## Bounded offline validation

Run `python -m unittest discover -s tests -v` with the selected dependencies.
The tests use in-memory responses and synthetic credentials only; sockets and
DNS are blocked. They exercise the repository's real streaming/confirmation
helpers without importing torch, plus Requests hostname/proxy/TLS behavior and
tqdm iteration/CLI parsing. They never fetch checkpoints or media.

The recorded local run used Python 3.12.14. Python 3.8 compatibility was checked
through release metadata, a Python-3.8-targeted wheel resolution, and Python 3.8
grammar parsing; no Python 3.8 interpreter execution or full application pass is
claimed. All newly changed Python code must retain Python 3.8 syntax.
