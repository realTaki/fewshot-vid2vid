"""Offline Pillow image-path checks; no torch/GPU or model inference is claimed.

Run: python -m unittest discover -s tests -p test_pillow_compat.py -v
Only the named, unchanged repo helper functions are AST-loaded so their image
operations can be checked without importing the ML/CUDA stack. ArrayTensor is
a deliberately narrow NumPy-backed protocol fixture, not a real torch.Tensor.
"""
import ast
import io
import os
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
from PIL import Image, ImageCms, ImageFont, ImageMath, features

ROOT = Path(__file__).resolve().parents[1]


def repo_function(relative, name):
    source = ROOT / relative
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"np": np, "Image": Image, "os": os}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


def image_eval(expression, **environment):
    # Test-only fixed trusted expressions; never accept external expression strings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ImageMath.unsafe_eval(expression, **environment)


class ArrayTensor:
    """The subset consumed by tensor2pilimage, intentionally not a torch mock."""
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def size(self, dimension=None):
        return self.array.shape if dimension is None else self.array.shape[dimension]

    def detach(self):
        return self

    def cpu(self):
        return self

    def squeeze(self):
        return ArrayTensor(self.array.squeeze())

    def numpy(self):
        return self.array

    def __add__(self, number):
        return ArrayTensor(self.array + number)

    def __mul__(self, number):
        return ArrayTensor(self.array * number)


class PillowImagePaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.convert = staticmethod(repo_function(
            "src/imaginaire/utils/visualization/common.py", "tensor2pilimage"))
        cls.save_jpeg = staticmethod(repo_function(
            "src/imaginaire/utils/io.py", "save_pilimage_in_jpeg"))

    def test_repo_rgb_channel_order_and_pixels(self):
        tensor = ArrayTensor(np.stack([np.ones((3, 4)), np.zeros((3, 4)), np.zeros((3, 4))]))
        image = self.convert(tensor)
        self.assertEqual((image.mode, image.size), ("RGB", (4, 3)))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_repo_bicubic_resize(self):
        image = self.convert(ArrayTensor(np.full((3, 3, 4), 0.5)), width=8, height=6)
        self.assertEqual(image.size, (8, 6))
        self.assertEqual(image.getpixel((0, 0)), (127, 127, 127))

    def test_repo_normalized_range(self):
        image = self.convert(ArrayTensor(np.full((3, 3, 4), -1)), minus1to1_normalized=True)
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))

    def test_repo_rejects_invalid_dimensions_and_channels(self):
        for shape in [(1, 3, 3, 4), (4, 3, 4)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.convert(ArrayTensor(np.zeros(shape)))

    def test_repo_nested_jpeg_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "frame.jpg"
            self.save_jpeg(str(output), Image.new("RGB", (4, 3), "red"))
            with Image.open(output) as image:
                self.assertEqual((image.format, image.mode, image.size), ("JPEG", "RGB", (4, 3)))
                self.assertGreater(image.getpixel((0, 0))[0], 250)

    def test_checked_in_input_frames_decode(self):
        base = ROOT / "src/dataset/unit_test/raw/fs_vid2vid"
        samples = [next((base / "face/images").rglob("*.jpg")),
                   next((base / "pose/human_instance_maps").rglob("*.png"))]
        for sample in samples:
            with self.subTest(sample=sample.name), Image.open(sample) as image:
                image.load()
                self.assertGreater(min(image.size), 0)
                self.assertEqual(image.convert("RGB").mode, "RGB")

    def test_webp_lossless_roundtrip(self):
        original = Image.fromarray(np.arange(48, dtype=np.uint8).reshape(4, 4, 3))
        buffer = io.BytesIO()
        original.save(buffer, "WEBP", lossless=True)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            np.testing.assert_array_equal(np.asarray(decoded), np.asarray(original))

    def test_webp_library_includes_2023_fix(self):
        # GHSA-j7hp-h8jx-5ppr: wheel codec evidence, not a memory-corruption PoC.
        version = features.version("webp")
        self.assertIsNotNone(version)
        self.assertGreaterEqual(tuple(int(x) for x in version.split(".")), (1, 3, 2))

    def test_imagemath_rejects_dunder_environment(self):
        # GHSA-3f63-hfp8-52jq: inert keys only; no command or payload is executed.
        with self.assertRaises(ValueError):
            image_eval("a", a=1, __class__=1)

    def test_imagemath_preserves_normal_image_expression(self):
        image = Image.new("I", (2, 2), 3)
        result = image_eval("a + 2", a=image)
        self.assertEqual(result.getpixel((0, 0)), 5)

    def test_font_length_guard_with_bounded_input(self):
        # GHSA-8ghj-p4vj-mr35: five characters, not a resource-exhaustion probe.
        sentinel = object()
        previous = getattr(ImageFont, "MAX_STRING_LENGTH", sentinel)
        try:
            ImageFont.MAX_STRING_LENGTH = 4
            font = ImageFont.load_default()
            with self.assertRaises(ValueError):
                font.getlength("abcde")
        finally:
            if previous is sentinel:
                del ImageFont.MAX_STRING_LENGTH
            else:
                ImageFont.MAX_STRING_LENGTH = previous

    def test_icc_normal_rgb_transform(self):
        profile = ImageCms.createProfile("sRGB")
        transform = ImageCms.buildTransform(profile, profile, "RGB", "RGB")
        output = ImageCms.applyTransform(Image.new("RGB", (2, 2), (12, 34, 56)), transform)
        self.assertEqual(output.getpixel((0, 0)), (12, 34, 56))

    def test_bounded_decompression_guard_remains_enabled(self):
        previous = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = 16
            buffer = io.BytesIO()
            Image.new("RGB", (8, 8)).save(buffer, "PNG")
            buffer.seek(0)
            with self.assertRaises(Image.DecompressionBombError):
                Image.open(buffer)
        finally:
            Image.MAX_IMAGE_PIXELS = previous


if __name__ == "__main__":
    unittest.main()
