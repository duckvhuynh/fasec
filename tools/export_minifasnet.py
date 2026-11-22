"""Export the MiniFASNet liveness models to ONNX from the official upstream PyTorch weights.

Upstream: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing (Apache-2.0), pinned commit below.
Pinned versions (see tools/requirements-export.txt): Python 3.12.10, torch 2.15.0+cpu, onnx 1.23.2,
onnxscript 0.7.2, onnxruntime 1.32.0, numpy 2.5.5. models/manifest.toml pins the SHA-256 of this export.

Usage: python tools/export_minifasnet.py [--out models]

Writes minifasnet_v2.onnx and minifasnet_v1se.onnx: input `input` float32 [1,3,80,80] BGR 0..255,
output `output` [1,3] raw logits (label 1 = real), opset 17.
"""

import argparse
import hashlib
import io
import sys
import tarfile
import tempfile
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

COMMIT = "b6d5f04ad78778917853b25c778acef6d5626d15"
TARBALL = f"https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/archive/{COMMIT}.tar.gz"
OPSET = 17
# Files we load or execute from the tarball, pinned so a changed archive cannot slip through.
PINNED = {
    "src/model_lib/MiniFASNet.py": "e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7",
    "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth":
        "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0",
    "resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth":
        "84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990",
}
MODELS = [  # (output file, upstream .pth, model factory name)
    ("minifasnet_v2.onnx", "2.7_80x80_MiniFASNetV2.pth", "MiniFASNetV2"),
    ("minifasnet_v1se.onnx", "4_0_0_80x80_MiniFASNetV1SE.pth", "MiniFASNetV1SE"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_upstream(tmp: Path) -> Path:
    print(f"downloading {TARBALL}")
    data = urllib.request.urlopen(TARBALL, timeout=120).read()
    root = tmp / "upstream"
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        # Extract only the pinned files, each verified before it is written.
        for rel, want in PINNED.items():
            blob = tar.extractfile(f"Silent-Face-Anti-Spoofing-{COMMIT}/{rel}").read()
            got = hashlib.sha256(blob).hexdigest()
            if got != want:
                sys.exit(f"sha256 mismatch for {rel}: expected {want}, got {got}")
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(blob)
    sys.path.insert(0, str(root))
    return root


def load_model(root: Path, pth: str, factory: str) -> torch.nn.Module:
    from src.model_lib import MiniFASNet  # upstream model definitions (hash checked above)

    model = getattr(MiniFASNet, factory)(conv6_kernel=(5, 5))  # upstream get_kernel(80, 80)
    state = torch.load(root / "resources/anti_spoof_models" / pth, map_location="cpu", weights_only=True)
    # Upstream (AntiSpoofPredict._load_model) strips a DataParallel `module.` prefix.
    if next(iter(state)).find("module.") >= 0:
        state = OrderedDict((k[7:], v) for k, v in state.items())
    model.load_state_dict(state)
    return model.eval()


def check_parity(model: torch.nn.Module, onnx_path: Path) -> float:
    import onnx

    onnx.checker.check_model(str(onnx_path))
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed: skipping ONNX-vs-PyTorch parity")
        return float("nan")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(8):
        x = rng.integers(0, 256, (1, 3, 80, 80)).astype(np.float32)  # BGR 0..255 like fasec
        with torch.no_grad():
            want = model(torch.from_numpy(x)).numpy()
        got = sess.run(["output"], {"input": x})[0]
        worst = max(worst, float(np.abs(got - want).max()))
    if worst > 1e-3:
        sys.exit(f"{onnx_path.name}: ONNX logits differ from PyTorch by {worst}")
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = fetch_upstream(Path(tmp))
        for name, pth, factory in MODELS:
            model = load_model(root, pth, factory)
            path = out / name
            # dynamo=False: the TorchScript exporter gives a byte-reproducible single file
            # (its deprecation warning is expected with the pinned torch).
            torch.onnx.export(
                model, (torch.zeros(1, 3, 80, 80),), str(path), input_names=["input"],
                output_names=["output"], opset_version=OPSET, dynamo=False,
            )
            diff = check_parity(model, path)
            print(f"{name}: max |logit diff| {diff:.2e}")
            print(f"  sha256 = \"{sha256(path)}\"\n  size = {path.stat().st_size}")


if __name__ == "__main__":
    main()
