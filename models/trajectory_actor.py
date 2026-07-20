import torch
import torch.nn as nn
from torch.distributions import Normal


class TrajectoryActor(nn.Module):
    """Minimal Gaussian actor over relative waypoint parameters."""

    def __init__(
        self,
        state_dim,
        n_uavs,
        hidden_dim=128,
        w_scale=1.0,
        log_std_min=-2.0,
        log_std_max=1.0,
    ):
        super().__init__()

        self.state_dim = int(state_dim)
        self.n_uavs = int(n_uavs)
        self.hidden_dim = int(hidden_dim)
        self.w_scale = float(w_scale)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.scheduled_log_std_max = None
        self.output_dim = self.n_uavs * 2

        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.n_uavs <= 0:
            raise ValueError("n_uavs must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.w_scale < 0:
            raise ValueError("w_scale must be non-negative.")
        if self.log_std_min > self.log_std_max:
            raise ValueError("log_std_min must be <= log_std_max.")

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(self.hidden_dim, self.output_dim)
        self.log_std_head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, state):
        """
        Args:
            state: Tensor or array with shape [state_dim] or [batch_size, state_dim].

        Returns:
            mean: Tensor with shape [batch_size, n_uavs * 2].
            log_std: Tensor with shape [batch_size, n_uavs * 2].
        """
        state = self._prepare_state(state)
        hidden = self.net(state)
        mean = self.mean_head(hidden)
        log_std_max = self._effective_log_std_max()
        log_std = self.log_std_head(hidden).clamp(self.log_std_min, log_std_max)
        return mean, log_std

    def sample(self, state, std_scale=1.0):
        """
        Sample bounded relative waypoint parameters with reparameterization.

        Returns:
            w: Tensor with shape [batch_size, n_uavs, 2].
            mean: Tensor with shape [batch_size, n_uavs * 2].
            log_std: Tensor with shape [batch_size, n_uavs * 2].
            raw_w: Tensor with shape [batch_size, n_uavs * 2].
        """
        mean, log_std = self.forward(state)
        std_scale = float(std_scale)
        if std_scale < 0.0:
            raise ValueError("std_scale must be non-negative.")

        # log_std is clamped in forward; scaling is an eval/training-time variance control.
        dist = Normal(mean, log_std.exp() * std_scale)
        raw_w = dist.rsample()
        bounded_w = torch.tanh(raw_w) * self.w_scale
        w = bounded_w.reshape(mean.shape[0], self.n_uavs, 2)
        return w, mean, log_std, raw_w

    def deterministic(self, state):
        """Return bounded mean trajectory parameters without sampling noise."""
        mean, log_std = self.forward(state)
        w = (torch.tanh(mean) * self.w_scale).reshape(mean.shape[0], self.n_uavs, 2)
        return w, mean, log_std

    def set_log_std_schedule(self, scheduled_log_std_max=None):
        """Set a temporary upper clamp for log_std; None restores the constructor bound."""
        if scheduled_log_std_max is None:
            self.scheduled_log_std_max = None
            return
        scheduled_log_std_max = float(scheduled_log_std_max)
        self.scheduled_log_std_max = max(self.log_std_min, min(self.log_std_max, scheduled_log_std_max))

    def _effective_log_std_max(self):
        if self.scheduled_log_std_max is None:
            return self.log_std_max
        return max(self.log_std_min, min(self.log_std_max, self.scheduled_log_std_max))

    def _prepare_state(self, state):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32)

        device = next(self.parameters()).device
        state = state.to(device=device, dtype=torch.float32)

        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"state must have shape [{self.state_dim}] or [B, {self.state_dim}].")
        return state
