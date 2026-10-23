# Pillow security patch within the existing Python 3.8 image stack

This PR changes only the root `requirements.txt` Pillow pin from the bot's
9.3.0 to 10.4.0, plus focused tests and this note. The default branch still pins
8.2. Python, CUDA, torch 1.9.1+cu111, torchvision 0.10.1+cu111 and the separate
SciPy/Requests PRs are not changed. No deployment lockfile existed or is invented.

## Live alert mapping (2026-10-23 UTC)

The authenticated Dependabot endpoint returned 31 open alerts, 26 for Pillow,
all mapped to `requirements.txt` and the default-branch `= 8.2` declaration.
Those are alert counts, not evidence of exploitability or successful closure.

Compared with this PR's original 9.3.0, 10.4.0 additionally avoids these four
reported vulnerable ranges:

| Alert | Advisory | First fixed |
| --- | --- | --- |
| #12 | [GHSA-j7hp-h8jx-5ppr](https://github.com/advisories/GHSA-j7hp-h8jx-5ppr), bundled libwebp | 10.0.1 |
| #14 | [GHSA-8ghj-p4vj-mr35](https://github.com/advisories/GHSA-8ghj-p4vj-mr35), font string allocation | 10.0.0 |
| #15 | [GHSA-3f63-hfp8-52jq](https://github.com/advisories/GHSA-3f63-hfp8-52jq), ImageMath environment | 10.2.0 |
| #16 | [GHSA-44wm-f244-xhp3](https://github.com/advisories/GHSA-44wm-f244-xhp3), ImageCms buffer overflow | 10.3.0 |

The bot's 9.3.0 already avoided #2/#3/#4/#5/#6/#7/#8/#11. This candidate thus
avoids 12 of the 26 observed Pillow ranges; **14 Pillow alerts (#22-#35) remain
applicable**, requiring 12.2.0 or 12.3.0. Both releases require Python >=3.10;
installing them blindly would break the documented Python 3.8 baseline. Resolving
those remaining alerts requires a separately authorized supported runtime/ML
migration. Five non-Pillow alerts are outside this PR. None is claimed closed:
this remains a draft and the default branch has not changed.

## Compatibility evidence and limits

[Pillow's support table](https://pillow.readthedocs.io/en/stable/installation/python-support.html)
lists 10.4 as the last series supporting Python 3.8. Its official wheel metadata
requires >=3.8. The exact official PyTorch `0.10.1+cu111` cp38 wheel declares
`pillow (>=5.3.0)`, without an upper bound; 10.4 satisfies that declaration.
This metadata is necessary, not proof of whole-application compatibility.

The official cp38 Linux x86_64 Pillow and NumPy wheels were verified against
PyPI SHA256, and the exact torchvision+cu111 wheel against its official index
SHA256. With native CPython 3.8.20 and NumPy 1.22.3:

- The 13 tracked offline tests pass with Pillow 10.4.0. The same tests on 9.3.0
  fail exactly three guards: inert ImageMath environment names, a five-character
  font bound, and bundled libwebp version (old 1.2.4 versus fixed >=1.3.2).
  No exploit commands or unbounded resource probes are run. The ImageCms case
  checks normal RGB conversion, not a memory-corruption reproducer.
- Image tests execute the unchanged `tensor2pilimage` and
  `save_pilimage_in_jpeg` function bodies, RGB channel order, normalization,
  bicubic resizing, nested JPEG output, checked-in JPEG/PNG frames, WebP lossless
  roundtrip and a small decompression-bomb boundary. AST-loading deliberately
  skips ML module imports; its NumPy-backed tensor protocol fixture is **not** a
  real torch.Tensor.
- Nine external checks of the exact torchvision+cu111 pure-PIL function bodies
  pass on both versions (resize, flips, crop/pad, affine/perspective,
  rotation/grayscale, color and invalid-type boundaries). Only its `import torch`
  and Torch JIT decorators are omitted. This is not a full torchvision import.

Run the tracked tests in an isolated environment with the selected Pillow and
NumPy installed:

```sh
python -m unittest discover -s tests -p test_pillow_compat.py -v
```

The reused Python 3.8.20 interpreter is uv-managed Astral standalone 20241002,
not the README's exact 3.8.10 or a PSF build; its independent archive checksum
remains unverified after an earlier HTTP403, which was not retried here.
Full ML dependency resolution (including legacy SciPy 1.3.3), torch/torchvision
module import, CUDA extensions, training/inference, model/data downloads,
Windows and complete CI are not validated. Non-wheel/system codec builds need
their own libwebp validation. No direct application use of ImageMath or the
font/CMS vulnerable APIs is established, and no complete safety claim is made.
