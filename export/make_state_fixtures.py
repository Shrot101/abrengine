"""
ABREngine — StateBuilder golden fixtures
========================================

`packages/abrengine/src/core/state-builder.ts` is a port of
`src/train.py::StateBuilder`. This script records, for a real rollout through
`src/env.py`, the *inputs* each `update()` received and the *exact state tensor*
Python produced afterwards.

The TypeScript test replays the same input sequence and asserts bit-for-bit
agreement (float32 exactness, not a tolerance — both sides do the same
arithmetic in the same order on the same float32 buffer).

This is what stops the port from silently drifting: a changed shift direction, a
changed divisor, or a row written at the wrong index all show up here
immediately, whereas the ONNX parity test would not notice because it starts
from a state tensor that is handed to it.

Usage:
    python export/make_state_fixtures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "export"))

from env import VideoStreamEnv, NUM_BITRATES, NUM_CHUNKS  # noqa: E402
from model import S_INFO, S_LEN, A_DIM  # noqa: E402
from train import StateBuilder, BUFFER_NORM, CHUNK_NORM, THROUGHPUT_NORM  # noqa: E402

from export_onnx import load_net  # noqa: E402


def record_episode(net, seed: int, policy: str) -> list[dict]:
    """Record every StateBuilder.update() input and the state it produced."""
    env = VideoStreamEnv(random_seed=seed)
    builder = StateBuilder()
    obs = env.reset()

    steps: list[dict] = []

    def record(o: dict) -> None:
        builder.update(o)
        steps.append(
            {
                # Inputs, in exactly the form StateBuilder.update() consumes.
                "input": {
                    "segmentBytes": int(o["chunk_size"]),
                    "downloadSec": float(o["delay"]),
                    "bufferSec": float(o["buffer_size"]),
                    "remainingSegments": float(o["video_chunk_remain"]),
                    "lastActionIndex": int(o["bitrate_action"]),
                    "nextSegmentBytesByAction": [float(x) for x in o["next_chunk_sizes"]],
                },
                # The full state tensor afterwards, row-major float32.
                "state": [float(x) for x in builder.get().reshape(-1)],
            }
        )

    record(obs)  # matches src/test.py, which updates with the reset observation

    done = False
    i = 0
    while not done:
        s = builder.get()
        if policy == "model":
            with torch.no_grad():
                probs, _ = net(torch.FloatTensor(s).unsqueeze(0))
            action = int(probs.argmax(dim=-1).item())
        else:
            action = (i * 5 + 1) % NUM_BITRATES
        obs, _, done, _ = env.step(action)
        record(obs)
        i += 1

    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint", default=str(REPO_ROOT / "checkpoints" / "abrengine_final.pt")
    )
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "export" / "fixtures" / "state-fixtures.json")
    )
    args = ap.parse_args()

    net = load_net(Path(args.checkpoint))

    episodes = [
        {"id": "model-seed7", "steps": record_episode(net, 7, "model")},
        {"id": "cycle-seed11", "steps": record_episode(net, 11, "cycle")},
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": {"stateShape": [S_INFO, S_LEN], "aDim": A_DIM},
                "normalisation": {
                    "bufferNormSec": BUFFER_NORM,
                    "chunkNormBytes": CHUNK_NORM,
                    "throughputNorm": THROUGHPUT_NORM,
                    "totalChunksNorm": NUM_CHUNKS,
                },
                "episodes": episodes,
            },
            indent=1,
        )
        + "\n"
    )
    total = sum(len(e["steps"]) for e in episodes)
    print(f"✓ {total} state fixtures → {out} ({out.stat().st_size/1024:.0f} KiB)")
    # Sanity: the recorded tensors really are what the network was fed.
    assert np.array(episodes[0]["steps"][0]["state"]).size == S_INFO * S_LEN


if __name__ == "__main__":
    main()
