"""
ABREngine A3C Training Loop (PyTorch)
======================================
Single-process A3C with rollout-based updates (on-policy).
For a resume project this is cleaner than full async multi-process A3C.
You can extend it to multiprocessing later if needed.

Usage:
    python train.py                    # default settings
    python train.py --episodes 5000    # longer run
    python train.py --trace_dir /path/to/cooked_traces
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from env   import VideoStreamEnv, NUM_BITRATES, NUM_CHUNKS, BITRATES
from model import ActorCritic, S_INFO, S_LEN

# ── Hyperparameters ────────────────────────────────────────────────────────
GAMMA          = 0.99     # discount factor
ENTROPY_WEIGHT = 0.5      # entropy regularisation coefficient (annealed later)
VALUE_LOSS_COEF= 0.5      # critic loss weighting
MAX_GRAD_NORM  = 0.5      # gradient clipping
LR             = 1e-4     # Adam learning rate
ROLLOUT_LEN    = 8        # steps per rollout before updating
LOG_INTERVAL   = 100      # episodes between console logs
SAVE_INTERVAL  = 500      # episodes between checkpoint saves
BUFFER_NORM    = 10.0     # normalisation constant for buffer level (seconds)
CHUNK_NORM     = 1e6      # normalise chunk sizes (bytes → MB-ish)
THROUGHPUT_NORM= 1.0      # Mbps values already reasonable


# ── State builder ──────────────────────────────────────────────────────────

class StateBuilder:
    """
    Maintains a rolling state tensor of shape (S_INFO, S_LEN).
    Call .update(obs) after every env.step(); call .get() to retrieve it.
    """
    def __init__(self):
        self.state = np.zeros((S_INFO, S_LEN), dtype=np.float32)
        self.throughput_hist = []
        self.download_hist   = []

    def reset(self):
        self.state[:] = 0
        self.throughput_hist = []
        self.download_hist   = []

    def update(self, obs: dict):
        delay_s   = obs["delay"]                  # seconds
        chunk_b   = obs["chunk_size"]             # bytes
        buffer_s  = obs["buffer_size"]            # seconds
        remain    = obs["video_chunk_remain"]
        bitrate   = obs["bitrate_action"]
        next_cs   = obs["next_chunk_sizes"]       # list of NUM_BITRATES bytes

        # Throughput = chunk_size / download_time (Mbps)
        tp = (chunk_b * 8.0 / 1e6) / max(delay_s, 1e-6)
        self.throughput_hist.append(tp)
        self.download_hist.append(delay_s)

        # Roll history window
        self.state[0, :-1] = self.state[0, 1:]
        self.state[0,  -1] = tp / THROUGHPUT_NORM

        self.state[1, :-1] = self.state[1, 1:]
        self.state[1,  -1] = delay_s

        # Next chunk sizes (one value per bitrate, normalised)
        for i in range(NUM_BITRATES):
            self.state[2, i] = next_cs[i] / CHUNK_NORM

        # Scalar inputs stored at last position of their row
        self.state[3, -1] = buffer_s  / BUFFER_NORM
        self.state[4, -1] = remain    / NUM_CHUNKS
        self.state[5, -1] = bitrate   / (NUM_BITRATES - 1)

    def get(self) -> np.ndarray:
        return self.state.copy()


# ── Training loop ──────────────────────────────────────────────────────────

def compute_returns(rewards, last_value, gamma=GAMMA):
    """Discounted returns from a rollout (used as targets for critic)."""
    R = last_value
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Setup ──────────────────────────────────────────────────────────────
    net       = ActorCritic().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)

    # Collect trace files if a directory is given
    trace_files = [None]   # None → synthetic trace
    if args.trace_dir and os.path.isdir(args.trace_dir):
        trace_files = [os.path.join(args.trace_dir, f)
                       for f in os.listdir(args.trace_dir)
                       if not f.startswith(".")]
        print(f"Loaded {len(trace_files)} trace files from {args.trace_dir}")

    env     = VideoStreamEnv()
    builder = StateBuilder()

    # ── Tracking ───────────────────────────────────────────────────────────
    ep_rewards     = []
    ep_avg_bitrate = []
    ep_rebuf       = []
    total_steps    = 0

    entropy_weight = ENTROPY_WEIGHT

    # ── Episode loop ────────────────────────────────────────────────────────
    for episode in range(1, args.episodes + 1):

        # Pick a random trace each episode (curriculum-free)
        trace = trace_files[np.random.randint(len(trace_files))]
        env   = VideoStreamEnv(trace_file=trace, random_seed=episode)
        builder.reset()

        obs         = env.reset()
        ep_reward   = 0.0
        ep_rb       = 0.0
        ep_br_sum   = 0.0
        step_count  = 0

        # Rollout buffers
        states, actions, rewards, log_probs, values, entropies = [], [], [], [], [], []

        done = False
        while not done:
            state_np = builder.get()

            action, log_prob, entropy, value = net.act(state_np, device)

            obs, reward, done, _ = env.step(action)
            builder.update(obs)

            # Track stats
            ep_reward  += reward
            ep_rb      += obs["rebuf"]
            ep_br_sum  += BITRATES[action]
            step_count += 1
            total_steps += 1

            states.append(state_np)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)

            # ── Update every ROLLOUT_LEN steps (or at episode end) ─────────
            if len(rewards) == ROLLOUT_LEN or done:
                # Bootstrap value for last state
                if done:
                    last_val = 0.0
                else:
                    with torch.no_grad():
                        s_t = torch.FloatTensor(builder.get()).unsqueeze(0).to(device)
                        _, last_val_t = net(s_t)
                        last_val = last_val_t.item()

                returns = compute_returns(rewards, last_val)
                returns_t = torch.FloatTensor(returns).to(device)

                states_t  = torch.FloatTensor(np.array(states)).to(device)
                actions_t = torch.LongTensor(actions).to(device)

                log_probs_t, vals_t, ents_t = net.evaluate(states_t, actions_t)

                # Advantage
                advantages = (returns_t - vals_t.detach())

                # Normalise advantages (reduces variance)
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Losses
                actor_loss  = -(log_probs_t * advantages).mean()
                critic_loss = VALUE_LOSS_COEF * (returns_t - vals_t).pow(2).mean()
                entropy_loss = -entropy_weight * ents_t.mean()

                loss = actor_loss + critic_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                # Clear rollout buffers
                states, actions, rewards, log_probs, values, entropies = [], [], [], [], [], []

        # ── Episode tracking ────────────────────────────────────────────────
        ep_rewards.append(ep_reward)
        ep_avg_bitrate.append(ep_br_sum / max(step_count, 1))
        ep_rebuf.append(ep_rb)

        # Anneal entropy weight (encourages exploitation over time)
        entropy_weight = max(0.1, ENTROPY_WEIGHT * (0.9995 ** episode))

        # ── Logging ─────────────────────────────────────────────────────────
        if episode % LOG_INTERVAL == 0:
            window = min(LOG_INTERVAL, len(ep_rewards))
            avg_r  = np.mean(ep_rewards[-window:])
            avg_br = np.mean(ep_avg_bitrate[-window:])
            avg_rb = np.mean(ep_rebuf[-window:])
            print(f"Ep {episode:5d} | Reward: {avg_r:7.3f} | "
                  f"Avg BR: {avg_br:7.0f} kbps | Rebuf: {avg_rb:.3f}s | "
                  f"Entropy W: {entropy_weight:.4f}")

        # ── Checkpoint ──────────────────────────────────────────────────────
        if episode % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(args.save_dir, f"abrengine_ep{episode}.pt")
            torch.save({
                "episode": episode,
                "model_state": net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "ep_rewards": ep_rewards,
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint → {ckpt_path}")

    # ── Save final model ────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, "abrengine_final.pt")
    torch.save({"model_state": net.state_dict(), "ep_rewards": ep_rewards}, final_path)
    print(f"\n✓ Training complete. Final model → {final_path}")

    # ── Plot training curves ─────────────────────────────────────────────────
    _plot_training(ep_rewards, ep_avg_bitrate, ep_rebuf, args.save_dir)

    return net, ep_rewards


def _plot_training(rewards, bitrates, rebuf, save_dir):
    window = 100
    def smooth(x):
        return np.convolve(x, np.ones(window)/window, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("ABREngine Training (PyTorch A3C)", fontsize=14)

    axes[0].plot(smooth(rewards),  color="#2196F3")
    axes[0].set_ylabel("Episode Reward (QoE)")
    axes[0].set_title("QoE Reward over Training")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(smooth(bitrates), color="#4CAF50")
    axes[1].set_ylabel("Avg Bitrate (kbps)")
    axes[1].set_title("Average Bitrate Selected")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(smooth(rebuf),    color="#F44336")
    axes[2].set_ylabel("Total Rebuffer (s)")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Rebuffering Events")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Training curves → {plot_path}")
    plt.close()


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ABREngine (PyTorch A3C)")
    parser.add_argument("--episodes",  type=int,   default=3000,
                        help="Total training episodes (default 3000)")
    parser.add_argument("--lr",        type=float, default=LR,
                        help="Adam learning rate")
    parser.add_argument("--trace_dir", type=str,   default=None,
                        help="Path to directory of cooked network traces")
    parser.add_argument("--save_dir",  type=str,   default="checkpoints",
                        help="Where to save model checkpoints")
    args = parser.parse_args()
    train(args)
