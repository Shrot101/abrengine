# ABREngine — Adaptive Bitrate Streaming with Deep Reinforcement Learning

> A PyTorch implementation of a reinforcement learning–based adaptive bitrate (ABR) controller for HTTP video streaming, inspired by the Pensieve system (Mao et al., SIGCOMM '17). Built from scratch as an end-to-end RL research reproduction and systems engineering project.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![A3C](https://img.shields.io/badge/Algorithm-A3C-purple)]()
[![Status](https://img.shields.io/badge/Status-Trained%20%26%20Evaluated-brightgreen)]()

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Why Reinforcement Learning?](#why-reinforcement-learning)
- [Architecture](#architecture)
- [Environment](#environment)
- [Reward Function](#reward-function)
- [State Representation](#state-representation)
- [A3C Network](#a3c-network)
- [Training](#training)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Research Inspiration](#research-inspiration)

---

## Overview

ABREngine trains a neural network agent to make real-time bitrate selection decisions during video streaming. Every few seconds, when a video player must decide which quality chunk to download next, ABREngine's policy network observes the current network conditions, buffer state, and video context — then outputs the optimal bitrate choice to maximise user Quality of Experience (QoE).

The project reproduces and experiments with the core RL pipeline from the Pensieve paper, reimplementing the environment simulation, actor-critic network architecture, and A3C training loop entirely in PyTorch. It is evaluated against four baseline ABR policies with quantitative QoE metrics.

**Key results at a glance:**

| Policy | QoE Reward | Rebuffer (s) | Avg Bitrate (kbps) |
|---|---|---|---|
| Random | 12.70 | 2.99 | 1889 |
| Lowest-BR (safe) | −1.84 | 0.43 | 300 |
| Highest-BR (greedy) | −302.69 | 99.50 | 4300 |
| Buffer-Based heuristic | 87.14 | 0.43 | 2359 |
| **ABREngine (ours)** | **95.45** | **1.89** | **2797** |

ABREngine achieves **+9.5% higher QoE** than the best heuristic baseline (Buffer-Based) while selecting **+18.6% higher average bitrate**, at the cost of a moderate increase in rebuffering.

---

## Problem Statement

HTTP Adaptive Bitrate (ABR) streaming powers platforms like YouTube, Netflix, and Twitch. The player must repeatedly decide: *given the current network, buffer, and video state — which quality level should the next chunk be downloaded at?*

This is a sequential decision-making problem with three competing objectives:

**1. Maximise video quality** — users prefer high-resolution video.  
**2. Minimise rebuffering** — stalls are the most damaging event to perceived quality.  
**3. Minimise quality switches** — frequent resolution changes are perceptually jarring.

Traditional ABR algorithms use hand-crafted heuristics: buffer-based rules (BBA), throughput-based rules (BOLA), or model-predictive control (MPC). These require careful manual tuning and fail to generalise across diverse network conditions. A learned policy can, in principle, balance all three objectives simultaneously and adapt to patterns that fixed rules cannot capture.

---

## Why Reinforcement Learning?

ABR selection maps naturally to a Markov Decision Process:

- **State** `sₜ` — observable network and buffer conditions at decision time `t`
- **Action** `aₜ` — bitrate level chosen for the next chunk (discrete, 6 options)
- **Reward** `rₜ` — QoE signal after chunk download completes
- **Episode** — one complete video playback session (48 chunks, ~3 minutes)

Conventional supervised learning cannot be applied directly: there is no ground-truth "correct" bitrate at each step — only a delayed, cumulative quality signal. RL is the natural fit because it learns to maximise long-horizon reward through trial and error, without requiring labelled decisions.

The A3C (Asynchronous Advantage Actor-Critic) algorithm was chosen for its sample efficiency and stability compared to vanilla policy gradient methods, and its demonstrated effectiveness in the original Pensieve work.

---

## Architecture

```
Network Input: State tensor  (6 streams × 8 timesteps)
                │
    ┌───────────┼───────────────────────────┐
    │           │                           │
    ▼           ▼                           ▼
Conv1D(k=4)  Conv1D(k=4)             Conv1D(k=4)
128 filters  128 filters             128 filters
Throughput   Download Time           Next Chunk Sizes
history      history                 (6 bitrate levels)
    │           │                           │
    └───────────┴──────────┬────────────────┘
                           │
             ┌─────────────┼──────────────┐
             │             │              │
           FC(1→128)    FC(1→128)     FC(1→128)
           Buffer        Remaining     Last Bitrate
           Level         Chunks
             │             │              │
             └─────────────┴──────────────┘
                                │
                    Concatenate all 6 branches
                                │
                        FC(merged → 128)
                             ReLU
                    ┌───────────────────┐
                    │                   │
              Actor Head            Critic Head
            FC(128 → 6)           FC(128 → 1)
              Softmax
                    │                   │
           Bitrate probabilities    State value V(s)
           π(a | s)                 (for advantage)
```

The parallel branch design allows each input modality to be processed independently before fusion, which is important because throughput histories, download times, and chunk sizes have very different statistical properties and should not share early-layer weights.

---

## Environment

`env.py` simulates a complete video streaming session as a gym-style environment. No external video files or media servers are required.

**Simulation mechanics:**

Given a bitrate action `aₜ`, the environment:

1. Looks up the chunk size in bytes for that bitrate: `chunk_bytes = CHUNK_SIZES[aₜ][chunk_idx]`
2. Walks through the bandwidth trace segment by segment, computing how long download takes given the available throughput at each moment
3. Simultaneously drains the playback buffer as time passes during the download
4. Tracks any rebuffering (time the buffer was empty while downloading)
5. Caps the buffer at 60 seconds (the player sleeps if the buffer is full)
6. Computes the QoE reward and returns the new observation

**Bandwidth traces:**

Each episode uses a network trace — a time series of `(timestamp_ms, bandwidth_Mbps)` pairs. When real traces are available (e.g. FCC or HSDPA), they are loaded directly. The synthetic trace generator samples bandwidth from a clipped normal distribution, producing realistic variability over a 300-second horizon.

**Video model:**

- 6 bitrate levels: 300, 750, 1200, 1850, 2850, 4300 kbps
- 48 chunks per video, 4 seconds each (~3-minute video)
- Chunk sizes are generated proportional to bitrate with 10% variance

---

## Reward Function

ABREngine uses the log-scale QoE formulation from the Pensieve paper:

```
rₜ = log(BRₜ / BR_min)  −  λ · Tᵣₑbᵤf  −  μ · |log(BRₜ/BR_min) − log(BRₜ₋₁/BR_min)|

where:
  BRₜ       = bitrate chosen for chunk t  (kbps)
  BR_min    = 300 kbps  (lowest available)
  Tᵣₑbᵤf   = rebuffering duration in seconds
  λ = 4.3   (rebuffer penalty weight)
  μ = 1.0   (smoothness penalty weight)
```

**Why log-scale quality?** The perceptual difference between 300 kbps and 750 kbps is much larger than the difference between 2850 kbps and 4300 kbps. Logarithmic quality reflects this diminishing returns property. Using raw kbps as the quality term would make λ = 4.3 far too weak, causing the agent to tolerate excessive rebuffering in exchange for marginal bitrate gains.

**Component ranges with log quality:**
- Quality term `log(BRₜ / 300)` ∈ [0, 2.66]
- Rebuffer penalty: 1 second of stall costs −4.3, exceeding the maximum quality gain
- This ensures the agent cannot rationally tolerate stalls just for a bitrate upgrade

---

## State Representation

The agent observes a `(6, 8)` state matrix at each decision step. Each row represents one information stream:

| Row | Content | Normalisation |
|---|---|---|
| 0 | Past 8 throughput measurements | Raw Mbps |
| 1 | Past 8 chunk download times | Raw seconds |
| 2 | Next chunk sizes at all 6 bitrates | ÷ 1×10⁶ |
| 3 | Current buffer level (scalar at `[-1]`) | ÷ 10.0 seconds |
| 4 | Remaining chunks (scalar at `[-1]`) | ÷ 48 |
| 5 | Last bitrate index (scalar at `[-1]`) | ÷ 5 |

The history window (rows 0–1) gives the agent temporal context about bandwidth trends. The next chunk size vector (row 2) allows the agent to reason about the cost of each bitrate option before committing. The scalar inputs (rows 3–5) provide current playback context.

---

## A3C Network

**Why Conv1D for time-series inputs?**

The throughput and download-time histories are short sequences (length 8). 1D convolutions with kernel size 4 extract local temporal patterns — such as a recent bandwidth drop or a sustained high-throughput period — without requiring the full sequence length. This is computationally efficient and avoids the vanishing gradient issues of RNNs for sequences this short.

**Actor-Critic design:**

- The **actor** outputs `π(aₜ | sₜ)` — a probability distribution over 6 bitrate actions. During training, actions are sampled stochastically. At inference, the argmax is taken.
- The **critic** estimates `V(sₜ)` — the expected cumulative discounted reward from state `sₜ`. This baseline is subtracted from Monte Carlo returns to form the advantage `Aₜ = Rₜ − V(sₜ)`, reducing gradient variance substantially.
- Both heads share the first six branches and the merged FC layer, encouraging shared representation learning.

**Weight initialisation:**

All layers use orthogonal initialisation with gain `√2`. The actor head uses a much smaller gain (0.01) to ensure the initial policy is nearly uniform — the agent should explore all bitrate options at the start of training rather than committing to a mode immediately.

---

## Training

**Algorithm:** Rollout-based A3C (single process, on-policy)

At each step:
1. The current state `sₜ` is passed through the network to get `π(aₜ | sₜ)` and `V(sₜ)`
2. Action `aₜ` is sampled from `π`; the environment returns `rₜ` and `sₜ₊₁`
3. After every 8 steps (or episode end), a rollout is complete

The update computes three loss terms:

```
L_actor  = −E[log π(aₜ|sₜ) · Aₜ]          # policy gradient
L_critic = 0.5 · E[(Rₜ − V(sₜ))²]          # value function MSE
L_entropy= −β · H[π(·|sₜ)]                  # entropy regularisation

L_total  = L_actor + L_critic + L_entropy
```

where `Aₜ = Rₜ − V(sₜ)` is the advantage and `Rₜ` is the discounted Monte Carlo return.

**Entropy annealing:** The entropy weight `β` starts at 0.5 and decays multiplicatively (factor 0.9995 per episode, floor 0.1). Early in training this encourages broad exploration across all bitrate levels; later it allows the policy to sharpen around high-value actions.

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 1×10⁻⁴ | Conservative; stable with Adam |
| Discount γ | 0.99 | Long-horizon QoE matters |
| Rollout length | 8 | One video segment worth of decisions |
| Entropy weight β | 0.5 → 0.1 | Exploration to exploitation schedule |
| Grad clip | 0.5 | Prevents policy collapse on bad traces |
| Value loss coef | 0.5 | Standard A3C weighting |

---

## Results

### Training Curves

![Training Curves](training_curves.png)

**What the plot shows:** Three panels tracking 5,000 training episodes (smoothed with a 100-episode window): QoE reward (top), average bitrate selected (middle), and total rebuffering per episode (bottom).

**Observations:**
- **Reward** grows monotonically from ~15 to ~90, demonstrating stable convergence. The dip around episodes 1,200–2,000 is a known A3C exploration phase where the entropy annealing causes the policy to temporarily commit to lower-bitrate, safer choices before recovering.
- **Bitrate** oscillates during early exploration (1,200–1,800 kbps range) as the agent tests conservative strategies, then stabilises around 2,700 kbps — a middle-high level that balances quality and safety.
- **Rebuffering** peaks above 15 seconds in the first 200 episodes (random-like policy), then declines steadily to under 2 seconds by episode 5,000. The secondary spike around episode 2,500 corresponds to the agent briefly over-selecting high bitrates before the critic catches up.

The training trajectory indicates the agent is genuinely learning the QoE tradeoff, not simply memorising traces.

---

### Baseline Comparison

![Evaluation Comparison](eval_comparison.png)

**What the plot shows:** Bar charts comparing five policies across QoE reward, total rebuffering per episode, and average bitrate selected, evaluated over 100 episodes on held-out random seeds.

**Policy breakdown:**
- **Random** — selects bitrate uniformly at random. Moderate reward due to occasional high-bitrate selections, but high variance.
- **Lowest-BR** — always selects 300 kbps. Zero rebuffering but very low quality; negative reward due to log-scale quality being near 0.
- **Highest-BR** — always selects 4300 kbps. Catastrophic: 99.5 seconds of rebuffering per episode on synthetic traces that regularly drop below the bitrate's download requirement.
- **Buffer-Based** — linearly maps buffer level to bitrate. Very low rebuffering (0.43s) and reasonable bitrate, representing a well-tuned heuristic.
- **ABREngine** — achieves the highest reward (95.45) with moderate rebuffering (1.89s) and the second-highest average bitrate (2797 kbps).

**Quantitative analysis:**

*vs. Buffer-Based (best heuristic):*
- QoE improvement: **(95.45 − 87.14) / 87.14 = +9.5%**
- Bitrate improvement: **(2797 − 2359) / 2359 = +18.6%**
- Rebuffering increase: **(1.89 − 0.43) / 0.43 = +339%** (from 0.43s to 1.89s absolute)

**Interpretation:** ABREngine trades a moderate increase in rebuffering (1.46 additional seconds per 3-minute video, or ~0.8% of playback time) for substantially higher video quality. Whether this tradeoff is desirable depends on the deployment context. In the original Pensieve paper, similar tradeoffs were observed; the log-QoE reward function explicitly encodes that higher bitrate is worth some rebuffering up to a threshold. With a higher λ (rebuffer penalty), the policy can be made more conservative.

---

### Single Episode Trace

![Episode Trace](episode_trace.png)

**What the plot shows:** A single evaluation episode, showing bitrate decisions (top), buffer level over time (middle), and per-step reward (bottom).

**Observations:**
- The agent starts conservatively (300 kbps, chunk 0) to avoid rebuffering from a cold start, then immediately jumps to 2850 kbps once a throughput estimate is established.
- The buffer stabilises just above the 5-second low threshold and stays there for most of the episode — the agent is actively managing buffer risk without building unnecessary reserve.
- Chunks 1–2 produce negative rewards (the cold-start penalty and initial rebuffering from the bitrate jump); all subsequent chunks are positive.
- The bitrate remains locked at 2850 kbps for the entire episode — no switching penalties are incurred after the initial ramp-up.

This behaviour reflects a learned strategy: start safe, ramp up quickly, then hold a steady middle-high bitrate rather than constantly switching.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/abrengine-rl.git
cd abrengine-rl

# Install dependencies
pip install torch numpy matplotlib

# Python 3.9+ required
```

No GPU required. Training on CPU completes in approximately 20–40 minutes for 3,000 episodes.

---

## Usage

### Train

```bash
# Train with synthetic traces (default, no data needed)
python train.py

# Train longer
python train.py --episodes 5000

# Train with real network traces
python train.py --trace_dir ./traces --episodes 5000

# Custom learning rate and checkpoint directory
python train.py --lr 5e-5 --save_dir ./my_checkpoints
```

Checkpoints are saved every 500 episodes to `checkpoints/abrengine_ep{N}.pt`.  
Training curves are saved to `checkpoints/training_curves.png`.

### Evaluate

```bash
# Evaluate against all baselines
python test.py --model checkpoints/abrengine_final.pt

# Evaluate over more episodes for statistical significance
python test.py --model checkpoints/abrengine_final.pt --episodes 200
```

Outputs: `eval_comparison.png`, `episode_trace.png`, and a console results table.

### Using Real Network Traces (Recommended)

Real FCC and HSDPA traces are available in the original Pensieve repository:

```bash
git clone https://github.com/hongzimao/pensieve.git
cp -r pensieve/sim/cooked_traces ./traces
python train.py --trace_dir ./traces --episodes 5000
```

Each trace file contains two columns: `timestamp_ms  bandwidth_Mbps`.

---

## Project Structure

```
abrengine-rl/
├── env.py            ← Streaming environment (pure NumPy, no ML dependency)
├── model.py          ← A3C actor-critic network (PyTorch)
├── train.py          ← Training loop + state builder + checkpointing
├── test.py           ← Evaluation against baselines + plotting
├── README.md
├── checkpoints/
│   ├── abrengine_ep500.pt
│   ├── abrengine_ep1000.pt
│   ├── abrengine_final.pt
│   └── training_curves.png
└── results/
    ├── eval_comparison.png
    └── episode_trace.png
```

---

## Research Inspiration

This project is an independent engineering implementation and reproduction inspired by:

> **Hongzi Mao, Ravi Netravali, Mohammad Alizadeh.**  
> *Neural Adaptive Video Streaming with Pensieve.*  
> ACM SIGCOMM 2017. [https://dl.acm.org/doi/10.1145/3098822.3098843](https://dl.acm.org/doi/10.1145/3098822.3098843)

Original source repository: [https://github.com/hongzimao/pensieve](https://github.com/hongzimao/pensieve)

ABREngine re-implements the core ideas from scratch in modern PyTorch (the original uses TensorFlow 1.x / TFLearn), makes the architecture and training loop more readable, replaces raw-kbps quality with the correct log-scale QoE formulation, and adds structured baseline evaluation. It does not reproduce the full production system (web server, Selenium integration, real video files) as those are not necessary for the RL research component.

---

## Technologies

- **Python 3.9+** — core language
- **PyTorch 2.0+** — neural network, automatic differentiation, optimiser
- **NumPy** — environment simulation, state computation
- **Matplotlib** — training curves and evaluation plots

---
