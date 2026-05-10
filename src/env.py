"""
ABREngine Simulation Environment
=====================================
Simulates a video streaming session over a real network trace.
The agent picks a bitrate for each chunk; the env returns what happened
(delay, rebuffering, buffer level, etc.) and whether the video ended.

No RL framework dependency — pure Python + NumPy.
"""

import math

import numpy as np
import os

# ── Video settings ──────────────────────
VIDEO_CHUNK_LEN   = 4.0          # seconds per chunk
BITRATES          = [300, 750, 1200, 1850, 2850, 4300]   # kbps
NUM_BITRATES      = len(BITRATES)
BUFFER_THRESH     = 60.0         # seconds — if buffer exceeds this, we sleep
DRAIN_BUFFER_SLEEP_TIME = 500.0  # ms
PACKET_PAYLOAD_PORTION = 0.95    # fraction of bytes that are payload
LINK_RTT          = 80           # ms baseline RTT
PACKET_SIZE       = 1500         # bytes
NUM_CHUNKS        = 48           # chunks per video (≈ 3-min video at 4s each)


def _make_chunk_sizes():
    rng = np.random.default_rng(42)
    sizes = {}
    for i, br in enumerate(BITRATES):
        # base size = bitrate * chunk_len / 8  (bits→bytes), add 10% variance
        base = br * 1000 * VIDEO_CHUNK_LEN / 8
        sizes[i] = (base * (1 + 0.1 * rng.standard_normal(NUM_CHUNKS))).astype(int)
        sizes[i] = np.maximum(sizes[i], 1)
    return sizes

CHUNK_SIZES = _make_chunk_sizes()   # dict: bitrate_idx → array[NUM_CHUNKS]


class VideoStreamEnv:
    """
    Single streaming session environment.

    Args:
        trace_file  : path to a cooked bandwidth trace (two columns: time_ms  bw_mbps)
                      If None, a synthetic trace is generated automatically.
        random_seed : for reproducibility
    """

    def __init__(self, trace_file=None, random_seed=42):
        self.rng = np.random.default_rng(random_seed)
        self.trace_file = trace_file

        # Load or synthesise bandwidth trace
        self.cooked_time, self.cooked_bw = self._load_trace(trace_file)

        self.reset()


    def reset(self):
        """Start a new episode. Returns the initial (zero) state."""
        self.chunk_idx      = 0
        self.buffer_size    = 0.0      # seconds of video buffered
        self.video_chunk_remain = NUM_CHUNKS

        # Pick a random starting point in the trace
        self.trace_ptr = self.rng.integers(0, len(self.cooked_time))
        self.last_mb_used = 0.0

        # Last chosen bitrate (for smoothness penalty)
        self.last_bitrate = 0

        return self._zero_obs()

    def step(self, bitrate_action: int):
        """
        Download the next chunk at the chosen bitrate.

        Returns
        -------
        obs   : dict with keys matching what the A3C agent needs
        reward: float   (QoE signal)
        done  : bool    (True when all chunks have been downloaded)
        info  : dict    (extra diagnostics)
        """
        assert 0 <= bitrate_action < NUM_BITRATES

        # ── Simulate downloading one chunk ─────────────────────────────────
        chunk_size_bytes = CHUNK_SIZES[bitrate_action][self.chunk_idx]
        delay, rebuf = self._simulate_download(chunk_size_bytes)

        # Update buffer
        self.buffer_size = max(0.0, self.buffer_size - rebuf)  # drained during download
        self.buffer_size += VIDEO_CHUNK_LEN                    # new chunk added

        # Cap buffer (player won't buffer more than BUFFER_THRESH)
        sleep_time = 0.0
        if self.buffer_size > BUFFER_THRESH:
            sleep_time = self.buffer_size - BUFFER_THRESH
            self.buffer_size = BUFFER_THRESH

        # ── Reward (QoE) ───────────────────────────────────────────────────
        # linear QoE = quality - rebuf_penalty - smoothness_penalty
        q_t      = math.log(BITRATES[bitrate_action] / BITRATES[0])   # log(BR / 300)
        q_prev   = math.log(BITRATES[self.last_bitrate] / BITRATES[0])
        reward = q_t  -  4.3 * rebuf  -  1.0 * abs(q_t - q_prev)

        self.last_bitrate = bitrate_action

        # ── Advance state ──────────────────────────────────────────────────
        self.chunk_idx += 1
        self.video_chunk_remain -= 1
        done = (self.chunk_idx >= NUM_CHUNKS)

        # Sizes of next chunk across all bitrates (needed as agent input)
        if not done:
            next_chunk_sizes = [CHUNK_SIZES[b][self.chunk_idx] for b in range(NUM_BITRATES)]
        else:
            next_chunk_sizes = [0] * NUM_BITRATES

        obs = {
            "delay":             delay / 1000.0,              # seconds
            "sleep_time":        sleep_time / 1000.0,
            "buffer_size":       self.buffer_size,
            "rebuf":             rebuf,
            "chunk_size":        chunk_size_bytes,
            "next_chunk_sizes":  next_chunk_sizes,
            "video_chunk_remain": self.video_chunk_remain,
            "bitrate_action":    bitrate_action,
        }

        info = {"chunk_idx": self.chunk_idx - 1, "trace_ptr": self.trace_ptr}
        return obs, reward, done, info

    # ── internals ──────────────────────────────────────────────────────────

    def _simulate_download(self, chunk_size_bytes):
        """
        Walk through the bandwidth trace to figure out how long it takes
        to download chunk_size_bytes. Returns (delay_ms, rebuf_seconds).
        """
        payload_bytes = chunk_size_bytes / PACKET_PAYLOAD_PORTION
        delay    = 0.0   # ms
        rebuf    = 0.0   # seconds

        while payload_bytes > 0:
            bw_bytes_per_ms = self.cooked_bw[self.trace_ptr] * 1e6 / 8.0 / 1000.0
            duration_ms = (self.cooked_time[self.trace_ptr]
                           - self.cooked_time[self.trace_ptr - 1
                                              if self.trace_ptr > 0 else 0])
            duration_ms = max(duration_ms, 1.0)

            bits_avail = bw_bytes_per_ms * duration_ms
            bytes_sent = min(bits_avail, payload_bytes)
            actual_ms  = duration_ms * bytes_sent / max(bits_avail, 1e-9)

            delay         += actual_ms
            payload_bytes -= bytes_sent

            # Buffer drains while we're downloading
            buffer_drain = actual_ms / 1000.0       # convert ms → seconds
            if self.buffer_size > buffer_drain:
                self.buffer_size -= buffer_drain
            else:
                rebuf += buffer_drain - self.buffer_size
                self.buffer_size = 0.0

            self.trace_ptr = (self.trace_ptr + 1) % len(self.cooked_time)

        delay += LINK_RTT
        return delay, rebuf

    def _load_trace(self, trace_file):
        """Load a cooked trace or generate a synthetic one."""
        if trace_file and os.path.exists(trace_file):
            times, bws = [], []
            with open(trace_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        times.append(float(parts[0]))
                        bws.append(float(parts[1]))
            return np.array(times), np.array(bws)
        else:
            # Synthetic 300-second trace with random bandwidth (0.5–8 Mbps)
            t = np.cumsum(self.rng.uniform(0.5, 2.0, 300))
            bw = np.abs(self.rng.normal(3.0, 1.5, 300)).clip(0.3, 10.0)
            return t, bw

    def _zero_obs(self):
        return {
            "delay": 0.0, "sleep_time": 0.0, "buffer_size": 0.0,
            "rebuf": 0.0, "chunk_size": 0, "next_chunk_sizes": [0]*NUM_BITRATES,
            "video_chunk_remain": NUM_CHUNKS, "bitrate_action": 0,
        }
