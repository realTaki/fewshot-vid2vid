"""Finite free-CPU-runner CUDA compile dependency setup; no driver/GPU execution.

--check-plan is offline. Payload setup runs only inside GitHub Actions; every
package is downloaded, hash-checked, installed/extracted, and deleted before
the next payload download. No cache or artifact upload is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def wheel_requirements(path):
    blocks = []
    current = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("--index-url"):
            if line != "--index-url https://pypi.org/simple":
                raise ValueError("unexpected index")
            continue
        if not line[0].isspace():
            if current:
                blocks.append("\n".join(current) + "\n")
            current = [line]
        else:
            if not current or not line.strip().startswith("--hash=sha256:"):
                raise ValueError("unexpected requirement continuation")
            current.append(line)
    if current:
        blocks.append("\n".join(current) + "\n")
    if len(blocks) != 24 or any("--hash=sha256:" not in b for b in blocks):
        raise ValueError("expected complete hash-required24-package closure")
    if sum(b.startswith("torch @ https://download-r2.pytorch.org/whl/cu118/")
           for b in blocks) != 1:
        raise ValueError("missing exact official torch backend URL")
    return blocks


def load_plan():
    plan = json.loads((ROOT / "scripts/cuda118-native-sdk.json").read_text())
    if plan["base_url"] != "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/":
        raise ValueError("unexpected SDK origin")
    rows = plan["nvidia_dependency_closure"]
    if len(rows) != 13 or not plan["no_driver_package"]:
        raise ValueError("unexpected SDK closure")
    for row in rows:
        name = row["Package"]
        if (not name.endswith("-11-8") or name.startswith("nvidia-driver")
                or name in ("cuda-11-8", "cuda-runtime-11-8", "cuda-drivers")):
            raise ValueError("driver/meta-package is outside compile-only scope")
        if not re.fullmatch(r"[a-f0-9]{64}", row["SHA256"]):
            raise ValueError("invalid SDK hash")
        if not re.fullmatch(r"\./[a-zA-Z0-9_.+~-]+\.deb", row["Filename"]):
            raise ValueError("invalid SDK filename")
    return plan, wheel_requirements(ROOT / "requirements-native-cu118.lock")


def require_free(path, minimum):
    free = shutil.disk_usage(path).free
    print(f"Free bytes: {free}; required: {minimum}", flush=True)
    if free < minimum:
        raise RuntimeError("disk resource gate; no cleanup/larger-runner fallback")


def setup(plan, blocks):
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("payload setup is CI-only; use --check-plan locally")
    if sys.version_info[:3] != (3, 11, 17):
        raise RuntimeError("expected pinned CPython3.11.17")
    temp = Path(os.environ["RUNNER_TEMP"]) / "fewshot-native"
    temp.mkdir(exist_ok=True)
    require_free(temp, 12 * GIB)
    sdk = temp / "sdk"
    sdk.mkdir(exist_ok=True)
    for row in plan["nvidia_dependency_closure"]:
        destination = temp / Path(row["Filename"]).name
        url = plan["base_url"] + row["Filename"].removeprefix("./")
        digest = hashlib.sha256()
        received = 0
        with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
        if received != int(row["Size"]) or digest.hexdigest() != row["SHA256"]:
            raise RuntimeError("SDK size/hash mismatch: " + row["Package"])
        subprocess.run(["dpkg-deb", "--extract", str(destination), str(sdk)],
                       check=True, timeout=180)
        destination.unlink()
        print("Verified/extracted/deleted SDK package", row["Package"], flush=True)
    venv = temp / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=120)
    python = venv / "bin/python"
    wheel_dir = temp / "one-wheel"
    wheel_dir.mkdir(exist_ok=True)
    one = temp / "one-requirement.lock"
    for block in blocks:
        one.write_text("--index-url https://pypi.org/simple\n" + block)
        if list(wheel_dir.iterdir()):
            raise RuntimeError("previous wheel was not removed")
        subprocess.run([str(python), "-m", "pip", "download", "--no-deps",
                        "--no-cache-dir", "--only-binary=:all:", "--require-hashes",
                        "--retries", "2", "--timeout", "60", "-r", str(one),
                        "--dest", str(wheel_dir)], check=True, timeout=360)
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("expected one verified wheel")
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps",
                        "--no-index", "--no-cache-dir", str(wheels[0])],
                       check=True, timeout=180)
        wheels[0].unlink()
        print("Installed/deleted verified wheel", block.splitlines()[0], flush=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True, timeout=60)
    home = sdk / "usr/local/cuda-11.8"
    subprocess.run([str(home / "bin/nvcc"), "--version"], check=True, timeout=30)
    require_free(temp, 2 * GIB)
    with Path(os.environ["GITHUB_ENV"]).open("a") as output:
        output.write(f"CUDA_HOME={home}\nNATIVE_PYTHON={python}\n")
    with Path(os.environ["GITHUB_PATH"]).open("a") as output:
        output.write(str(home / "bin") + "\n" + str(venv / "bin") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-plan", action="store_true")
    args = parser.parse_args()
    plan, blocks = load_plan()
    print("Validated SDK13 package hashes and native24 package hashlock.", flush=True)
    if not args.check_plan:
        setup(plan, blocks)
