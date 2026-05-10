"""
ABREngine Evaluation Script
============================
Loads a trained checkpoint and evaluates it against three baselines:
  1. Random policy
  2. Lowest bitrate always (safe)
  3. Highest bitrate always (greedy)
  4. Buffer-based heuristic (simple rule-based ABR)

Usage:
    python test.py --model checkpoints/abrengine_final.pt
    python test.py --model checkpoints/abrengine_final.pt --episodes 100
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from env   import VideoStreamEnv, NUM_BITRATES, NUM_CHUNKS, BITRATES
from model import ActorCritic
from train import StateBuilder

EVAL_EPISODES = 50


# ── Baseline policies ────────────────────────────────────────────────────────

def policy_random(obs, state_np, **_):
    return np.random.randint(NUM_BITRATES)

def policy_lowest(obs, state_np, **_):
    return 0

def policy_highest(obs, state_np, **_):
    return NUM_BITRATES - 1

def policy_buffer_based(obs, state_np, **_):
    """
    Simple buffer-based ABR heuristic (similar to BBA):
      - buffer < 5s  → lowest bitrate
      - buffer > 40s → highest bitrate
      - otherwise    → linearly scale
    """
    buf = obs["buffer_size"]
    if buf < 5.0:
        return 0
    elif buf > 40.0:
        return NUM_BITRATES - 1
    else:
        idx = int((buf - 5.0) / (40.0 - 5.0) * (NUM_BITRATES - 1))
        return min(idx, NUM_BITRATES - 1)

def policy_abrengine(obs, state_np, net, device, **_):
    state_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
    with torch.no_grad():
        probs, _ = net(state_t)
    return probs.argmax(dim=-1).item()


# ── Evaluation runner ─────────────────────────────────────────────────────────

def evaluate_policy(policy_fn, n_episodes=EVAL_EPISODES, seed_offset=10000, **kwargs):
    """
    Runs a policy for n_episodes and returns arrays of episode metrics.
    """
    rewards, avg_bitrates, rebuf_totals, quality_smooth = [], [], [], []

    for ep in range(n_episodes):
        env     = VideoStreamEnv(random_seed=seed_offset + ep)
        builder = StateBuilder()
        obs     = env.reset()
        builder.update(obs)

        ep_reward = 0.0
        ep_rebuf  = 0.0
        ep_br     = []
        prev_br   = 0
        done      = False

        while not done:
            state_np = builder.get()
            action   = policy_fn(obs=obs, state_np=state_np, **kwargs)
            obs, reward, done, _ = env.step(action)
            builder.update(obs)

            ep_reward += reward
            ep_rebuf  += obs["rebuf"]
            ep_br.append(BITRATES[action])
            prev_br = action

        rewards.append(ep_reward)
        rebuf_totals.append(ep_rebuf)
        avg_bitrates.append(np.mean(ep_br))

    return {
        "reward":      np.array(rewards),
        "rebuf":       np.array(rebuf_totals),
        "avg_bitrate": np.array(avg_bitrates),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(results: dict, save_path="eval_comparison.png"):
    policies = list(results.keys())
    colors   = ["#F44336", "#FF9800", "#9C27B0", "#2196F3", "#4CAF50"][:len(policies)]

    metrics = ["reward", "rebuf", "avg_bitrate"]
    titles  = ["QoE Reward (higher = better)",
               "Total Rebuffering / episode (lower = better)",
               "Average Bitrate kbps (higher = better)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ABREngine vs Baselines", fontsize=15, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        means = [results[p][metric].mean() for p in policies]
        stds  = [results[p][metric].std()  for p in policies]
        bars  = ax.bar(policies, means, color=colors, alpha=0.85, width=0.5)
        ax.errorbar(policies, means, yerr=stds, fmt="none",
                    ecolor="black", capsize=5, linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels(policies, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Annotate bar values
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * abs(m),
                    f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Comparison chart → {save_path}")
    plt.close()


def plot_episode_trace(net, device, save_path="episode_trace.png"):
    """Visualise one episode: bitrate choices, buffer, throughput, rewards."""
    env     = VideoStreamEnv(random_seed=99999)
    builder = StateBuilder()
    obs     = env.reset()
    builder.update(obs)

    bitrate_log, buffer_log, reward_log, rebuf_log = [], [], [], []
    done = False
    while not done:
        state_t = torch.FloatTensor(builder.get()).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, _ = net(state_t)
        action = probs.argmax(dim=-1).item()
        obs, reward, done, _ = env.step(action)
        builder.update(obs)
        bitrate_log.append(BITRATES[action])
        buffer_log.append(obs["buffer_size"])
        reward_log.append(reward)
        rebuf_log.append(obs["rebuf"])

    chunks = range(len(bitrate_log))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("ABREngine — Single Episode Trace", fontsize=13)

    axes[0].step(chunks, bitrate_log, color="#2196F3", where="post")
    axes[0].set_ylabel("Bitrate (kbps)")
    axes[0].set_ylim(0, max(BITRATES) * 1.1)
    for br in BITRATES:
        axes[0].axhline(br, color="grey", linestyle="--", alpha=0.25, linewidth=0.8)

    axes[1].plot(chunks, buffer_log, color="#4CAF50")
    axes[1].axhline(5.0,  color="orange", linestyle="--", alpha=0.5, label="5s (low)")
    axes[1].axhline(40.0, color="red",    linestyle="--", alpha=0.5, label="40s (high)")
    axes[1].set_ylabel("Buffer (s)")
    axes[1].legend(fontsize=8)

    axes[2].bar(chunks, reward_log, color=["#F44336" if r < 0 else "#4CAF50"
                                            for r in reward_log], alpha=0.8)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Step Reward")
    axes[2].set_xlabel("Chunk index")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✓ Episode trace   → {save_path}")
    plt.close()


# ── Entry point ─────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    net = ActorCritic().to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    print(f"✓ Loaded model from {args.model}")

    n = args.episodes
    print(f"\nEvaluating {n} episodes per policy …\n")

    results = {}
    results["Random"]       = evaluate_policy(policy_random,       n)
    results["Lowest-BR"]    = evaluate_policy(policy_lowest,        n)
    results["Highest-BR"]   = evaluate_policy(policy_highest,       n)
    results["Buffer-Based"] = evaluate_policy(policy_buffer_based,  n)
    results["ABREngine"]     = evaluate_policy(policy_abrengine, n,
                                              net=net, device=device)

    # Print summary table
    print(f"\n{'Policy':<15} {'Reward':>10} {'Rebuf(s)':>10} {'AvgBR kbps':>12}")
    print("-" * 50)
    for name, m in results.items():
        print(f"{name:<15} {m['reward'].mean():>10.3f} "
              f"{m['rebuf'].mean():>10.3f} {m['avg_bitrate'].mean():>12.0f}")

    import os
    out_dir = os.path.dirname(args.model) or "."
    plot_comparison(results,  save_path=os.path.join(out_dir, "eval_comparison.png"))
    plot_episode_trace(net, device, save_path=os.path.join(out_dir, "episode_trace.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES,
                        help="Episodes per policy for evaluation")
    args = parser.parse_args()
    main(args)
