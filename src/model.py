"""
ABREngine A3C Network (PyTorch)
================================
Exact architecture from the Pensieve paper (SIGCOMM '17, §5.3):

  • One 1D-CNN per time-series input (throughput history, download-time history)
  • One 1D-CNN for the next-chunk size vector
  • Three FC layers for scalar inputs (buffer, remaining chunks, last bitrate)
  • All streams are concatenated → shared FC → actor head + critic head

State tensor shape: (S_INFO=6, S_LEN=8)
  Row 0 : past k throughput measurements          (Mbps)
  Row 1 : past k download times                   (seconds)
  Row 2 : next chunk sizes across A_DIM bitrates  (normalised)
  Row 3 : current buffer level                    (normalised, scalar at [-1])
  Row 4 : remaining chunks                        (normalised, scalar at [-1])
  Row 5 : last bitrate index                      (one-hot-ish, scalar at [-1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Dimensions (matching original repo constants) ──────────────────────────
S_INFO   = 6    # number of input feature streams
S_LEN    = 8    # history window length
A_DIM    = 6    # number of bitrate levels (actions)
HIDDEN   = 128  # width of the shared FC layer


class ActorCritic(nn.Module):
    """
    Single network with shared body, actor head, and critic head.
    Using a shared body is standard A3C practice and produces better
    feature representations than two separate networks.
    """

    def __init__(self, s_len=S_LEN, a_dim=A_DIM, hidden=HIDDEN):
        super().__init__()
        self.s_len = s_len
        self.a_dim = a_dim

        # ── Time-series streams (1D CNN → flatten) ─────────────────────────
        # Throughput history  (row 0)
        self.cnn_throughput = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Download time history  (row 1)
        self.cnn_download = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Next chunk sizes  (row 2, length = A_DIM = 6)
        self.cnn_chunk = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
        )

        # ── Scalar inputs — three separate FC branches ────────────────────
        self.fc_buffer   = nn.Linear(1, 128)
        self.fc_remain   = nn.Linear(1, 128)
        self.fc_bitrate  = nn.Linear(1, 128)

        # ── Compute merged size dynamically ───────────────────────────────
        dummy_ts  = torch.zeros(1, 1, s_len)
        dummy_cs  = torch.zeros(1, 1, a_dim)
        cnn_ts_out = self.cnn_throughput(dummy_ts).shape[1]   # 128*(s_len-3)
        cnn_cs_out = self.cnn_chunk(dummy_cs).shape[1]        # 128*(a_dim-3)
        merged = cnn_ts_out + cnn_ts_out + cnn_cs_out + 128 * 3

        # ── Shared FC ─────────────────────────────────────────────────────
        self.fc_shared = nn.Sequential(
            nn.Linear(merged, hidden),
            nn.ReLU(),
        )

        # ── Actor head (policy) ───────────────────────────────────────────
        self.actor_head = nn.Linear(hidden, a_dim)

        # ── Critic head (value) ───────────────────────────────────────────
        self.critic_head = nn.Linear(hidden, 1)

        self._init_weights()

    def forward(self, state: torch.Tensor):
        """
        Args:
            state : (batch, S_INFO, S_LEN)  — see module docstring for layout

        Returns:
            action_probs : (batch, A_DIM)   — softmax policy
            state_value  : (batch, 1)       — critic estimate
        """
        B = state.shape[0]

        # Split streams
        tp   = state[:, 0, :].unsqueeze(1)          # (B, 1, S_LEN)
        dl   = state[:, 1, :].unsqueeze(1)          # (B, 1, S_LEN)
        cs   = state[:, 2, :self.a_dim].unsqueeze(1)  # (B, 1, A_DIM)
        buf  = state[:, 3, -1].unsqueeze(1)         # (B, 1)  scalar
        rem  = state[:, 4, -1].unsqueeze(1)         # (B, 1)  scalar
        bit  = state[:, 5, -1].unsqueeze(1)         # (B, 1)  scalar

        # CNN branches
        x_tp  = self.cnn_throughput(tp)
        x_dl  = self.cnn_download(dl)
        x_cs  = self.cnn_chunk(cs)

        # FC scalar branches
        x_buf = F.relu(self.fc_buffer(buf))
        x_rem = F.relu(self.fc_remain(rem))
        x_bit = F.relu(self.fc_bitrate(bit))

        # Merge
        x = torch.cat([x_tp, x_dl, x_cs, x_buf, x_rem, x_bit], dim=1)
        x = self.fc_shared(x)

        action_probs = F.softmax(self.actor_head(x), dim=-1)
        state_value  = self.critic_head(x)

        return action_probs, state_value

    def act(self, state_np: np.ndarray, device="cpu"):
        """
        Convenience: takes a raw numpy state (S_INFO, S_LEN),
        returns (action_int, log_prob, entropy, value).
        Used by the training loop.
        """
        state_t = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        probs, value = self.forward(state_t)
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        return (action.item(),
                dist.log_prob(action),
                dist.entropy(),
                value)

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Used in the update step: evaluate a batch of (state, action) pairs.
        Returns log_probs, state_values, entropy.
        """
        probs, values = self.forward(states)
        dist      = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values.squeeze(-1), entropy

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Actor head: smaller gain for initial near-uniform policy
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
