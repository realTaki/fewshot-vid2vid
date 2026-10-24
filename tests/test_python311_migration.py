"""Finite CPU migration tests, never training, CUDA operators or model downloads.

Run in the documented Python3.11 CPU tuple. Real CPU-safe modules are imported;
only the large trainer's save/load methods are AST-isolated (no trainer init).
"""
import argparse
import ast
import contextlib
import importlib
import io
import os
import pickle
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import scipy
import torch
import torchaudio
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def trainer_function(name):
    path = ROOT / "src/imaginaire/trainers/base.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    namespace = {"os": os, "torch": torch, "master_only": lambda function: function}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class Python311Migration(unittest.TestCase):
    def setUp(self):
        for name in ("create_connection", "getaddrinfo", "socket"):
            guard = mock.patch.object(socket, name,
                                      side_effect=AssertionError("network forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_exact_cpu_validation_anchors(self):
        self.assertEqual(sys.version_info[:2], (3, 11))
        self.assertEqual(torch.__version__.split("+")[0], "2.7.1")
        self.assertEqual(torchvision.__version__.split("+")[0], "0.22.1")
        self.assertEqual(torchaudio.__version__.split("+")[0], "2.7.1")
        self.assertEqual(np.__version__, "1.26.4")
        self.assertEqual(scipy.__version__, "1.15.3")

    def test_real_torch_image_helpers_and_torchvision_ops(self):
        from imaginaire.utils.visualization.common import tensor2pilimage
        from imaginaire.utils.io import save_intermediate_training_results
        image = tensor2pilimage(torch.ones(3, 3, 4), width=8, height=6)
        self.assertEqual((image.mode, image.size), ("RGB", (8, 6)))
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        with tempfile.TemporaryDirectory() as directory:
            save_intermediate_training_results(torch.zeros(1, 3, 4, 4), directory, 1, 2)
            output = Path(directory) / "images/epoch_00001iteration000000002.jpg"
            self.assertTrue(output.is_file())
        selected = torchvision.ops.nms(torch.tensor([[0., 0., 2., 2.],
                                                      [0., 0., 2., 2.]]),
                                       torch.tensor([.9, .5]), .5)
        self.assertEqual(selected.tolist(), [0])

    def test_torchaudio_cpu_resample(self):
        output = torchaudio.functional.resample(torch.zeros(1, 16), 16000, 8000)
        self.assertEqual(tuple(output.shape), (1, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_flownet_python311_timer_and_argument_introspection(self):
        from imaginaire.third_party.flow_net.flownet2.utils import tools
        class Candidate:
            def __init__(self, width=4):
                self.width = width
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with tools.TimerBlock("synthetic") as timer:
                timer.log("bounded")
        self.assertGreaterEqual(timer.interval, 0)
        self.assertIn("Operation finished", output.getvalue())
        parser = argparse.ArgumentParser()
        with mock.patch.object(sys, "argv", ["synthetic"]):
            tools.add_arguments_for_module(parser, SimpleNamespace(Candidate=Candidate),
                                           "candidate", "Candidate")
            args = parser.parse_args(["--candidate_width", "7"])
        self.assertEqual(args.candidate_width, 7)

    def test_flownet_frame_reader_rgb_rgba_and_array(self):
        from imaginaire.third_party.flow_net.flownet2.utils import frame_utils
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("RGB", "RGBA"):
                path = Path(directory) / (mode + ".png")
                Image.new(mode, (3, 2), (11, 22, 33) if mode == "RGB"
                          else (11, 22, 33, 255)).save(path)
                pixels = frame_utils.read_gen(str(path))
                self.assertEqual(pixels.shape, (2, 3, 3))
                np.testing.assert_array_equal(pixels[0, 0], [11, 22, 33])
            path = Path(directory) / "synthetic.raw"
            with path.open("wb") as stream:
                np.save(stream, np.zeros((2, 3), dtype=np.float32))
            self.assertEqual(frame_utils.read_gen(str(path)).shape, (2, 3))

    def test_real_legacy_augmentation_with_image_mask_keypoint(self):
        from imaginaire.utils.data import Augmentor
        augmentor = Augmentor({"resize_h_w": "8,8", "center_crop_h_w": "4,4"},
                              {}, ["images", "mask"],
                              {"images": False, "mask": True}, ["points"],
                              "INTER_LINEAR")
        inputs = {"images": [Image.new("RGB", (8, 8), "red")],
                  "mask": [Image.new("L", (8, 8), 1)],
                  "points": [np.array([[3., 3.]])]}
        result, flipped = augmentor.perform_augmentation(
            inputs, True, augmentor.augment_ops)
        self.assertEqual(result["images"][0].shape[:2], (4, 4))
        self.assertEqual(result["mask"][0].shape[:2], (4, 4))
        np.testing.assert_array_equal(result["points"][0], [[1., 1.]])
        self.assertFalse(flipped)

    def test_legacy_crop_and_compression_apis_construct(self):
        from imaginaire.utils.data import Augmentor
        augmentor = Augmentor({"random_resize_h_w_aspect": "4,4(1,1)",
                               "compression": {"quality_lower": 80, "p": 1}},
                              {}, ["image"], {"image": False}, [], "INTER_LINEAR")
        result, _ = augmentor.perform_augmentation(
            {"image": [Image.new("RGB", (8, 8), "red")]},
            True, augmentor.augment_ops)
        self.assertEqual(result["image"][0].shape, (4, 4, 3))

    def test_real_trainer_state_optimizer_scheduler_checkpoint_roundtrip(self):
        save = trainer_function("_save_checkpoint")
        load = trainer_function("load_checkpoint")
        with tempfile.TemporaryDirectory() as directory:
            cfg = SimpleNamespace(logdir=directory,
                                  trainer=SimpleNamespace(strict_resume=True))
            networks = [torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)]
            optimizers = [torch.optim.Adam(n.parameters()) for n in networks]
            schedulers = [torch.optim.lr_scheduler.StepLR(o, 1) for o in optimizers]
            for network, optimizer in zip(networks, optimizers):
                network(torch.ones(1, 2)).sum().backward()
                optimizer.step()
            expected = networks[0].weight.detach().clone()
            path = save(cfg, *networks, *optimizers, *schedulers, 2, 7)
            with torch.no_grad():
                networks[0].weight.zero_()
            trainer = SimpleNamespace(cfg=cfg, net_G=networks[0], net_D=networks[1],
                                      opt_G=optimizers[0], opt_D=optimizers[1],
                                      sch_G=schedulers[0], sch_D=schedulers[1],
                                      is_inference=False)
            result = load(trainer, cfg, path, resume=True)
            self.assertEqual(result, (True, 2, 7))
            torch.testing.assert_close(networks[0].weight, expected)
            self.assertTrue(optimizers[0].state)

    def test_tensor_checkpoint_and_reject_unallowlisted_object(self):
        with tempfile.TemporaryDirectory() as directory:
            tensor_path = Path(directory) / "tensor.pt"
            torch.save(torch.ones(2, 2), tensor_path)
            torch.testing.assert_close(torch.load(tensor_path, weights_only=True),
                                       torch.ones(2, 2))
            object_path = Path(directory) / "unsupported.pt"
            # A plain inert object, not a payload and never unpickled unrestricted.
            torch.save(SimpleNamespace(value=1), object_path)
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(object_path, weights_only=True)

    def test_all_source_checkpoint_loads_explicitly_keep_restricted_mode(self):
        count = 0
        for path in (ROOT / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "torch" and node.func.attr == "load"):
                    count += 1
                    keywords = {k.arg: k.value for k in node.keywords}
                    self.assertIn("weights_only", keywords, str(path))
                    self.assertIsInstance(keywords["weights_only"], ast.Constant)
                    self.assertIs(keywords["weights_only"].value, True)
        self.assertEqual(count, 7)

    def test_native_source_contracts_only_not_a_cuda_compile(self):
        for name in ("correlation", "channelnorm", "resample2d", "bias_act", "upfirdn2d"):
            setup = (ROOT / "src/imaginaire/third_party" / name / "setup.py").read_text()
            self.assertIn("-std=c++17", setup)
            self.assertNotIn("-std=c++14", setup)
        for name in ("correlation", "channelnorm", "resample2d"):
            for path in (ROOT / "src/imaginaire/third_party" / name / "src").glob("*.cu"):
                self.assertNotIn(".type()", path.read_text())
                self.assertNotIn(".data<", path.read_text())
        for name in ("positional_encoding_kernel.cu", "sp_trilinear_worldcoord_kernel.cu"):
            source = (ROOT / "src/imaginaire/model_utils/gancraft/voxlib" / name).read_text()
            self.assertNotIn("THCudaCheck", source)
            self.assertIn("C10_CUDA_CHECK", source)


if __name__ == "__main__":
    unittest.main()
