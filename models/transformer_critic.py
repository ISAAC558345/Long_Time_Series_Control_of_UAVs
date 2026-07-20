import torch
import torch.nn as nn


class TrajectoryTransformerCritic(nn.Module):
    """Minimal Transformer critic for fixed-length UAV action segments."""

    def __init__(
        self,
        state_dim,
        n_uavs,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        max_segment_length=16,
        normalize_q=True,
        smooth_prefix_q=True,
        prefix_ema_alpha=0.6,
        q_norm_eps=1e-6,
    ):
        super().__init__()

        self.state_dim = int(state_dim)
        self.n_uavs = int(n_uavs)
        self.hidden_dim = int(hidden_dim)
        self.max_segment_length = int(max_segment_length)
        self.normalize_q = bool(normalize_q)
        self.smooth_prefix_q = bool(smooth_prefix_q)
        self.prefix_ema_alpha = float(prefix_ema_alpha)
        self.q_norm_eps = float(q_norm_eps)

        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.n_uavs <= 0:
            raise ValueError("n_uavs must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.max_segment_length <= 0:
            raise ValueError("max_segment_length must be positive.")
        if not 0.0 < self.prefix_ema_alpha <= 1.0:
            raise ValueError("prefix_ema_alpha must be in (0, 1].")
        if self.q_norm_eps <= 0.0:
            raise ValueError("q_norm_eps must be positive.")

        self.state_encoder = nn.Linear(self.state_dim, self.hidden_dim)
        self.action_encoder = nn.Linear(self.n_uavs * 2, self.hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_segment_length + 1, self.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        self.q_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, start_state, action_subseq):
        """
        Args:
            start_state: Tensor with shape [batch_size, state_dim].
            action_subseq: Tensor with shape [batch_size, segment_length, n_uavs, 2].

        Returns:
            state_value: Tensor with shape [batch_size].
            prefix_q_values: Tensor with shape [batch_size, segment_length].
        """
        if start_state.ndim != 2 or start_state.shape[-1] != self.state_dim:
            raise ValueError(f"start_state must have shape [B, {self.state_dim}].")
        if action_subseq.ndim != 4 or action_subseq.shape[2:] != (self.n_uavs, 2):
            raise ValueError(f"action_subseq must have shape [B, L, {self.n_uavs}, 2].")
        if start_state.shape[0] != action_subseq.shape[0]:
            raise ValueError("start_state and action_subseq must have the same batch size.")

        batch_size, segment_length = action_subseq.shape[:2]
        if segment_length > self.max_segment_length:
            raise ValueError("segment_length exceeds max_segment_length.")

        state_token = self.state_encoder(start_state).unsqueeze(1)
        flat_actions = action_subseq.reshape(batch_size, segment_length, self.n_uavs * 2)
        action_tokens = self.action_encoder(flat_actions)
        tokens = torch.cat([state_token, action_tokens], dim=1)
        tokens = tokens + self.pos_embedding[:, : segment_length + 1]

        causal_mask = self._causal_mask(segment_length + 1, tokens.device)
        hidden = self.transformer(tokens, mask=causal_mask)

        state_value = self.value_head(hidden[:, 0]).squeeze(-1)
        raw_prefix_q_values = self.q_head(hidden[:, 1:]).squeeze(-1)
        prefix_q_values = self.postprocess_prefix_q(raw_prefix_q_values)
        return state_value, prefix_q_values

    def postprocess_prefix_q(self, prefix_q_values):
        """Normalize and smooth prefix Q values for more stable trajectory ranking."""
        if self.normalize_q:
            mean = prefix_q_values.mean()
            std = prefix_q_values.std(unbiased=False)
            prefix_q_values = (prefix_q_values - mean) / (std + self.q_norm_eps)

        if self.smooth_prefix_q and prefix_q_values.shape[1] > 1:
            smoothed = [prefix_q_values[:, 0]]
            prev = prefix_q_values[:, 0]
            alpha = self.prefix_ema_alpha
            for idx in range(1, prefix_q_values.shape[1]):
                prev = alpha * prefix_q_values[:, idx] + (1.0 - alpha) * prev
                smoothed.append(prev)
            prefix_q_values = torch.stack(smoothed, dim=1)

        return prefix_q_values

    def value(self, states):
        """Estimate V(s) for states with shape [B, state_dim] or [B, L, state_dim]."""
        if states.ndim == 2:
            if states.shape[-1] != self.state_dim:
                raise ValueError(f"states must have shape [B, {self.state_dim}].")
            return self.value_head(self.state_encoder(states)).squeeze(-1)
        if states.ndim == 3:
            if states.shape[-1] != self.state_dim:
                raise ValueError(f"states must have shape [B, L, {self.state_dim}].")
            return self.value_head(self.state_encoder(states)).squeeze(-1)
        raise ValueError("states must have shape [B, state_dim] or [B, L, state_dim].")

    @staticmethod
    def _causal_mask(seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
