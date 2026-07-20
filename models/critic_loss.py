import torch
import torch.nn.functional as F


VALID_TARGET_MODES = {"bootstrapped_n_step", "no_bootstrap", "weighted_bootstrap"}
VALID_REWARD_NORMALIZATIONS = {"none", "mean_abs", "running_mean_abs"}


class RunningRewardNormalizer:
    """Running mean-absolute reward scaler used only before critic target construction."""

    def __init__(self, momentum=0.99, eps=1e-6, clip_value=5.0):
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.clip_value = float(clip_value)
        self.running_abs = None

        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")
        if self.clip_value <= 0.0:
            raise ValueError("clip_value must be positive.")

    def normalize(self, rewards, update=True):
        with torch.no_grad():
            batch_abs = float(rewards.detach().abs().mean().clamp_min(self.eps).item())
            if update or self.running_abs is None:
                if self.running_abs is None:
                    self.running_abs = batch_abs
                else:
                    self.running_abs = self.momentum * self.running_abs + (1.0 - self.momentum) * batch_abs
            scale = self.running_abs if self.running_abs is not None else batch_abs
        return torch.clamp(rewards / (scale + self.eps), -self.clip_value, self.clip_value)


def _normalize_prefix_values(values, eps=1e-6):
    mean = values.mean()
    std = values.std(unbiased=False)
    return (values - mean) / (std + eps)


def _ema_prefix_values(values, alpha=0.6):
    if values.shape[1] <= 1:
        return values
    smoothed = [values[:, 0]]
    prev = values[:, 0]
    for idx in range(1, values.shape[1]):
        prev = alpha * values[:, idx] + (1.0 - alpha) * prev
        smoothed.append(prev)
    return torch.stack(smoothed, dim=1)


def _postprocess_targets_like_critic(targets, critic):
    processed = targets
    if getattr(critic, "normalize_q", False):
        processed = _normalize_prefix_values(processed, eps=getattr(critic, "q_norm_eps", 1e-6))
    if getattr(critic, "smooth_prefix_q", False):
        processed = _ema_prefix_values(processed, alpha=getattr(critic, "prefix_ema_alpha", 0.6))
    return processed


def compute_n_step_targets(
    reward_subseq,
    done_subseq,
    bootstrap_values,
    gamma=0.99,
    target_mode="bootstrapped_n_step",
    bootstrap_weight=1.0,
    reward_normalization="mean_abs",
    reward_norm_eps=1e-6,
    reward_normalizer=None,
    update_reward_normalizer=True,
    reward_clip=5.0,
):
    """Compute prefix N-step targets for Q(s, a_0:N)."""
    if reward_subseq.ndim != 3:
        raise ValueError("reward_subseq must have shape [B, L, n_uavs].")
    if done_subseq.ndim != 2:
        raise ValueError("done_subseq must have shape [B, L].")
    if bootstrap_values.ndim != 2:
        raise ValueError("bootstrap_values must have shape [B, L].")
    if target_mode not in VALID_TARGET_MODES:
        raise ValueError(f"target_mode must be one of {sorted(VALID_TARGET_MODES)}.")
    if reward_normalization not in VALID_REWARD_NORMALIZATIONS:
        raise ValueError(f"reward_normalization must be one of {sorted(VALID_REWARD_NORMALIZATIONS)}.")
    if bootstrap_weight < 0.0:
        raise ValueError("bootstrap_weight must be non-negative.")
    if reward_norm_eps <= 0.0:
        raise ValueError("reward_norm_eps must be positive.")

    batch_size, segment_length = done_subseq.shape
    if reward_subseq.shape[:2] != (batch_size, segment_length):
        raise ValueError("reward_subseq and done_subseq must share [B, L].")
    if bootstrap_values.shape != (batch_size, segment_length):
        raise ValueError("bootstrap_values must have shape [B, L].")

    scalar_reward_seq = reward_subseq.mean(dim=-1)
    if reward_normalization == "running_mean_abs":
        if reward_normalizer is None:
            reward_scale = scalar_reward_seq.abs().mean().clamp_min(reward_norm_eps)
            scalar_reward_seq = torch.clamp(scalar_reward_seq / reward_scale, -float(reward_clip), float(reward_clip))
        else:
            scalar_reward_seq = reward_normalizer.normalize(scalar_reward_seq, update=update_reward_normalizer)
    elif reward_normalization == "mean_abs":
        reward_scale = scalar_reward_seq.abs().mean(dim=1, keepdim=True).clamp_min(reward_norm_eps)
        scalar_reward_seq = torch.clamp(scalar_reward_seq / reward_scale, -float(reward_clip), float(reward_clip))

    discounts = gamma ** torch.arange(segment_length, device=reward_subseq.device, dtype=reward_subseq.dtype)
    discounted_rewards = scalar_reward_seq * discounts.unsqueeze(0)
    discounted_prefix_returns = torch.cumsum(discounted_rewards, dim=1)

    done_prefix = torch.cumsum(done_subseq.to(reward_subseq.dtype), dim=1).clamp(max=1.0)
    bootstrap_mask = 1.0 - done_prefix
    bootstrap_discounts = gamma ** torch.arange(
        1,
        segment_length + 1,
        device=reward_subseq.device,
        dtype=reward_subseq.dtype,
    )
    if target_mode == "bootstrapped_n_step":
        effective_bootstrap_weight = 1.0
    elif target_mode == "no_bootstrap":
        effective_bootstrap_weight = 0.0
    else:
        effective_bootstrap_weight = float(bootstrap_weight)

    bootstrap_term = (
        effective_bootstrap_weight
        * bootstrap_mask
        * bootstrap_discounts.unsqueeze(0)
        * bootstrap_values
    )
    n_step_targets = discounted_prefix_returns + bootstrap_term
    return n_step_targets.detach(), scalar_reward_seq


def compute_critic_loss(
    critic,
    target_critic,
    start_state,
    action_subseq,
    reward_subseq,
    bootstrap_state,
    done_subseq,
    state_subseq=None,
    gamma=0.99,
    target_mode="bootstrapped_n_step",
    bootstrap_weight=1.0,
    reward_normalization="running_mean_abs",
    reward_normalizer=None,
    reward_clip=5.0,
    monotonicity_weight=1e-2,
    monotonicity_epsilon=0.0,
    ranking_weight=5e-2,
    ranking_margin=0.05,
):
    state_value, prefix_q_values = critic(start_state, action_subseq)

    with torch.no_grad():
        if state_subseq is None:
            raise ValueError("state_subseq is required for strict prefix-specific N-step targets.")
        if state_subseq.ndim != 3 or state_subseq.shape[:2] != (reward_subseq.shape[0], reward_subseq.shape[1] + 1):
            raise ValueError("state_subseq must have shape [B, L+1, state_dim].")
        bootstrap_values = target_critic.value(state_subseq[:, 1:, :])
        n_step_targets, scalar_reward_seq = compute_n_step_targets(
            reward_subseq=reward_subseq,
            done_subseq=done_subseq,
            bootstrap_values=bootstrap_values,
            gamma=gamma,
            target_mode=target_mode,
            bootstrap_weight=bootstrap_weight,
            reward_normalization=reward_normalization,
            reward_normalizer=reward_normalizer,
            update_reward_normalizer=True,
            reward_clip=reward_clip,
        )

    q_loss_targets = _postprocess_targets_like_critic(n_step_targets, critic).detach()
    prefix_q_loss = F.mse_loss(prefix_q_values, q_loss_targets)
    if prefix_q_values.shape[1] > 1:
        monotonicity_loss = F.relu(
            prefix_q_values[:, :-1] - prefix_q_values[:, 1:] - float(monotonicity_epsilon)
        ).mean()
    else:
        monotonicity_loss = prefix_q_values.new_zeros(())

    predicted_scores = prefix_q_values.mean(dim=1)
    target_scores = n_step_targets[:, -1]
    return_diff = target_scores.unsqueeze(1) - target_scores.unsqueeze(0)
    q_diff = predicted_scores.unsqueeze(1) - predicted_scores.unsqueeze(0)
    valid_pairs = return_diff > 1e-6
    if torch.any(valid_pairs):
        ranking_loss = F.relu(float(ranking_margin) - q_diff[valid_pairs]).mean()
    else:
        ranking_loss = prefix_q_values.new_zeros(())

    critic_loss = (
        prefix_q_loss
        + float(monotonicity_weight) * monotonicity_loss
        + float(ranking_weight) * ranking_loss
    )

    return critic_loss, dict(
        state_value=state_value,
        prefix_q_values=prefix_q_values,
        scalar_reward_seq=scalar_reward_seq,
        n_step_targets=n_step_targets,
        q_loss_targets=q_loss_targets,
        bootstrap_values=bootstrap_values,
        prefix_q_loss=prefix_q_loss,
        monotonicity_loss=monotonicity_loss,
        monotonicity_weight=float(monotonicity_weight),
        ranking_loss=ranking_loss,
        ranking_weight=float(ranking_weight),
        predicted_scores=predicted_scores,
        target_scores=target_scores,
        target_mode=target_mode,
        bootstrap_weight=bootstrap_weight,
        reward_normalization=reward_normalization,
        reward_clip=float(reward_clip),
    )
