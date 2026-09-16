"""
ABREngine — PyTorch → ONNX export
=================================

Exports the trained ActorCritic checkpoint (``checkpoints/*.pt``) to a single
ONNX graph that can be executed by onnxruntime-web (browser) and
onnxruntime-node (Node.js).

IMPORTANT — semantics are preserved exactly:

  * The ONNX graph input is the **same (S_INFO=6, S_LEN=8) state tensor** that
    ``src/model.py::ActorCritic.forward`` consumes. No preprocessing is folded
    into the graph, no normalisation is changed, no rows are reordered.
  * The ONNX graph outputs are the **same two tensors** the PyTorch module
    returns: ``action_probs`` (softmax over A_DIM=6) and ``state_value``.
  * Action semantics are untouched: index i -> BITRATES[i] with
    BITRATES = [300, 750, 1200, 1850, 2850, 4300] kbps (src/env.py).

The only structural difference from ``ActorCritic.forward`` is that the export
wrapper is a thin ``nn.Module`` around it, so that the traced graph has stable
named inputs/outputs and a dynamic batch axis.

Usage
-----
    python export/export_onnx.py \
        --checkpoint checkpoints/abrengine_final.pt \
        --out packages/abrengine/models/ac3-controller.onnx

Also writes a sidecar ``<out>.json`` manifest describing the input/output
schema, the normalisation constants copied from ``src/train.py``, and the
bitrate ladder copied from ``src/env.py``, so the JavaScript package never
hard-codes these numbers independently of the research code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make ``src/`` importable without modifying it.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import ActorCritic, S_INFO, S_LEN, A_DIM, HIDDEN  # noqa: E402
from env import (  # noqa: E402
    BITRATES,
    NUM_BITRATES,
    NUM_CHUNKS,
    VIDEO_CHUNK_LEN,
    BUFFER_THRESH,
)
from train import BUFFER_NORM, CHUNK_NORM, THROUGHPUT_NORM  # noqa: E402

OPSET = 17  # onnxruntime-web 1.x supports opset 17 comfortably


class AbrPolicyExport(nn.Module):
    """Thin export wrapper. Delegates 1:1 to the trained ActorCritic."""

    def __init__(self, net: ActorCritic):
        super().__init__()
        self.net = net

    def forward(self, state: torch.Tensor):
        probs, value = self.net(state)
        return probs, value


def _git_rev(root: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_net(checkpoint: Path) -> ActorCritic:
    net = ActorCritic()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    net.load_state_dict(state)
    net.eval()
    return net


def export(checkpoint: Path, out: Path, opset: int = OPSET) -> Path:
    net = load_net(checkpoint)
    wrapper = AbrPolicyExport(net).eval()

    dummy = torch.zeros(1, S_INFO, S_LEN, dtype=torch.float32)

    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["action_probs", "state_value"],
        dynamic_axes={
            "state": {0: "batch"},
            "action_probs": {0: "batch"},
            "state_value": {0: "batch"},
        },
        dynamo=False,  # TorchScript exporter: stable, no extra deps
    )

    # Structural check
    import onnx

    m = onnx.load(str(out))
    onnx.checker.check_model(m)

    ops = sorted({n.op_type for n in m.graph.node})

    manifest = {
        "name": "ac3-controller",
        "description": (
            "ABREngine A3C actor-critic policy (Pensieve-style) exported from "
            "PyTorch to ONNX. Graph input is the raw (6, 8) state tensor built "
            "by src/train.py::StateBuilder; graph outputs are the softmax policy "
            "and the critic value."
        ),
        "createdUtc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "checkpoint": checkpoint.name,
            "checkpointSha256": _sha256(checkpoint),
            "gitRev": _git_rev(REPO_ROOT),
            "torch": torch.__version__,
            "opset": opset,
        },
        "graph": {
            "opTypes": ops,
            "inputs": [
                {
                    "name": "state",
                    "type": "float32",
                    "shape": ["batch", S_INFO, S_LEN],
                    "layout": {
                        "0": "throughput history, Mbps, oldest..newest (÷ THROUGHPUT_NORM)",
                        "1": "download-time history, seconds, oldest..newest",
                        "2": "next-chunk sizes for each of A_DIM bitrates, bytes ÷ CHUNK_NORM, "
                        "stored at indices 0..A_DIM-1; indices A_DIM..S_LEN-1 unused (0)",
                        "3": "buffer level, seconds ÷ BUFFER_NORM, stored at index S_LEN-1 only",
                        "4": "remaining chunks ÷ NUM_CHUNKS, stored at index S_LEN-1 only",
                        "5": "last bitrate index ÷ (A_DIM-1), stored at index S_LEN-1 only",
                    },
                }
            ],
            "outputs": [
                {"name": "action_probs", "type": "float32", "shape": ["batch", A_DIM]},
                {"name": "state_value", "type": "float32", "shape": ["batch", 1]},
            ],
        },
        "dims": {
            "S_INFO": S_INFO,
            "S_LEN": S_LEN,
            "A_DIM": A_DIM,
            "HIDDEN": HIDDEN,
        },
        # Copied verbatim from the research code so JS never re-derives them.
        "normalisation": {
            "BUFFER_NORM": BUFFER_NORM,
            "CHUNK_NORM": CHUNK_NORM,
            "THROUGHPUT_NORM": THROUGHPUT_NORM,
            "NUM_CHUNKS": NUM_CHUNKS,
        },
        "ladder": {
            "bitratesKbps": list(BITRATES),
            "numBitrates": NUM_BITRATES,
            "chunkDurationSeconds": VIDEO_CHUNK_LEN,
            "bufferThresholdSeconds": BUFFER_THRESH,
        },
        "actionSemantics": (
            "argmax over action_probs -> index i in [0, A_DIM); the trained policy "
            "means 'download the next chunk at ladder.bitratesKbps[i]'. Index 0 is the "
            "lowest quality, index A_DIM-1 the highest."
        ),
        "modelSha256": _sha256(out),
        "modelBytes": out.stat().st_size,
    }

    manifest_path = out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"✓ ONNX model    → {out}  ({out.stat().st_size / 1024:.1f} KiB)")
    print(f"✓ Manifest      → {manifest_path}")
    print(f"  op types: {', '.join(ops)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export ABREngine .pt → .onnx")
    ap.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "checkpoints" / "abrengine_final.pt"),
        help="Path to the trained .pt checkpoint",
    )
    ap.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "packages" / "abrengine" / "models" / "ac3-controller.onnx"
        ),
        help="Output .onnx path",
    )
    ap.add_argument("--opset", type=int, default=OPSET)
    args = ap.parse_args()

    export(Path(args.checkpoint), Path(args.out), args.opset)


if __name__ == "__main__":
    main()
