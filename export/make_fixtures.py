"""
ABREngine — golden fixture generator
====================================

Produces a deterministic set of (state, expected_output) pairs from the
*research* code path, so that the JavaScript runtime can be verified against
PyTorch without needing PyTorch at test time.

States come from three sources so the fixtures cover realistic *and*
adversarial inputs:

  1. Real rollouts through ``src/env.py`` driven by the trained policy itself
     (argmax, exactly as ``src/test.py::policy_abrengine`` does).
  2. Rollouts driven by the buffer-based heuristic, so the state distribution
     is not limited to states the policy likes.
  3. Hand-built boundary states (all zeros, saturated buffer, zero throughput,
     max bitrate index, huge chunk sizes, ...).

For each state we record the PyTorch ``action_probs``, ``state_value`` and the
argmax action. Written to ``export/fixtures/parity-fixtures.json``.

Usage:
    python export/make_fixtures.py
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

from env import VideoStreamEnv, NUM_BITRATES  # noqa: E402
from model import S_INFO, S_LEN, A_DIM  # noqa: E402
from train import StateBuilder  # noqa: E402

from export_onnx import load_net  # noqa: E402


def _rollout_states(net, seed: int, policy: str) -> list[np.ndarray]:
    """One episode; returns the state observed *before* each decision."""
    env = VideoStreamEnv(random_seed=seed)
    builder = StateBuilder()
    obs = env.reset()
    builder.update(obs)  # matches src/test.py::evaluate_policy

    states: list[np.ndarray] = []
    done = False
    while not done:
        s = builder.get()
        states.append(s.copy())

        if policy == "model":
            with torch.no_grad():
                probs, _ = net(torch.FloatTensor(s).unsqueeze(0))
            action = int(probs.argmax(dim=-1).item())
        elif policy == "buffer":
            buf = obs["buffer_size"]
            if buf < 5.0:
                action = 0
            elif buf > 40.0:
                action = NUM_BITRATES - 1
            else:
                action = min(
                    int((buf - 5.0) / 35.0 * (NUM_BITRATES - 1)), NUM_BITRATES - 1
                )
        else:  # deterministic pseudo-random walk over the ladder
            action = (len(states) * 7) % NUM_BITRATES

        obs, _, done, _ = env.step(action)
        builder.update(obs)
    return states


def _boundary_states() -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []

    z = np.zeros((S_INFO, S_LEN), dtype=np.float32)
    out.append(("all-zeros", z.copy()))

    s = np.zeros((S_INFO, S_LEN), dtype=np.float32)
    s[0, :] = 8.0  # very fast link, 8 Mbps sustained
    s[1, :] = 0.5
    s[2, :A_DIM] = np.array([0.15, 0.375, 0.6, 0.925, 1.425, 2.15], dtype=np.float32)
    s[3, -1] = 6.0  # 60 s buffer (BUFFER_THRESH) / BUFFER_NORM=10
    s[4, -1] = 1.0
    s[5, -1] = 1.0
    out.append(("fast-link-full-buffer", s.copy()))

    s = np.zeros((S_INFO, S_LEN), dtype=np.float32)
    s[0, :] = 0.05  # near-dead link
    s[1, :] = 20.0
    s[2, :A_DIM] = np.array([0.15, 0.375, 0.6, 0.925, 1.425, 2.15], dtype=np.float32)
    s[3, -1] = 0.0  # empty buffer -> rebuffering
    s[4, -1] = 0.02
    s[5, -1] = 1.0
    out.append(("dead-link-empty-buffer", s.copy()))

    s = np.zeros((S_INFO, S_LEN), dtype=np.float32)
    s[0, :] = np.linspace(0.3, 9.0, S_LEN, dtype=np.float32)  # rising bandwidth
    s[1, :] = np.linspace(4.0, 0.3, S_LEN, dtype=np.float32)
    s[2, :A_DIM] = np.array([0.15, 0.375, 0.6, 0.925, 1.425, 2.15], dtype=np.float32)
    s[3, -1] = 1.5
    s[4, -1] = 0.5
    s[5, -1] = 0.4
    out.append(("rising-bandwidth", s.copy()))

    s = np.zeros((S_INFO, S_LEN), dtype=np.float32)
    s[0, :] = np.linspace(9.0, 0.3, S_LEN, dtype=np.float32)  # collapsing bandwidth
    s[1, :] = np.linspace(0.3, 4.0, S_LEN, dtype=np.float32)
    s[2, :A_DIM] = np.array([0.15, 0.375, 0.6, 0.925, 1.425, 2.15], dtype=np.float32)
    s[3, -1] = 0.3
    s[4, -1] = 0.8
    s[5, -1] = 1.0
    out.append(("collapsing-bandwidth", s.copy()))

    # Out-of-distribution magnitudes: verifies numerics, not policy quality.
    s = np.full((S_INFO, S_LEN), 1e3, dtype=np.float32)
    out.append(("large-magnitude", s.copy()))

    s = np.full((S_INFO, S_LEN), -1e3, dtype=np.float32)
    out.append(("negative-magnitude", s.copy()))

    s = np.full((S_INFO, S_LEN), 1e-7, dtype=np.float32)
    out.append(("tiny-magnitude", s.copy()))

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint", default=str(REPO_ROOT / "checkpoints" / "abrengine_final.pt")
    )
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "export" / "fixtures" / "parity-fixtures.json")
    )
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()

    net = load_net(Path(args.checkpoint))

    cases: list[dict] = []

    for ep in range(args.episodes):
        for policy in ("model", "buffer", "cycle"):
            for i, s in enumerate(_rollout_states(net, 20250101 + ep, policy)):
                cases.append({"id": f"rollout-{policy}-ep{ep}-step{i}", "state": s})

    for name, s in _boundary_states():
        cases.append({"id": f"boundary-{name}", "state": s})

    # Batch through PyTorch in one shot (also exercises the dynamic batch axis).
    batch = torch.from_numpy(np.stack([c["state"] for c in cases]).astype(np.float32))
    with torch.no_grad():
        probs, value = net(batch)
    probs_np = probs.numpy()
    value_np = value.numpy()

    records = []
    for i, c in enumerate(cases):
        records.append(
            {
                "id": c["id"],
                "state": c["state"].astype(np.float32).reshape(-1).tolist(),
                "expected": {
                    "actionProbs": [float(x) for x in probs_np[i]],
                    "stateValue": float(value_np[i][0]),
                    "action": int(np.argmax(probs_np[i])),
                },
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema": {
                    "stateShape": [S_INFO, S_LEN],
                    "stateOrder": "row-major (C order): index = row * S_LEN + col",
                    "aDim": A_DIM,
                },
                "checkpoint": Path(args.checkpoint).name,
                "torch": torch.__version__,
                "count": len(records),
                "cases": records,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"✓ {len(records)} fixtures → {out_path} ({out_path.stat().st_size/1024:.0f} KiB)")


if __name__ == "__main__":
    main()
