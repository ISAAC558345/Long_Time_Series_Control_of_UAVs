import torch

from models.trajectory_generator import TorchLinearWaypointTrajectoryGenerator


def compute_actor_loss(
    actor,
    critic,
    start_state,
    n_uavs,
    H,
    range_pos,
    max_move_per_slot,
    smoothness_weight=0.0,
    curvature_weight=0.0,
):
    """
    Compute the minimal differentiable actor objective from critic prefix values.

    Args:
        start_state: Tensor or array with shape [B, state_dim].

    Returns:
        actor_loss: Scalar tensor.
        info: Dict of tensors used by the smoke test.
    """
    device = next(actor.parameters()).device
    if not torch.is_tensor(start_state):
        start_state = torch.as_tensor(start_state, dtype=torch.float32, device=device)
    else:
        start_state = start_state.to(device=device, dtype=torch.float32)

    if start_state.ndim != 2:
        raise ValueError("start_state must have shape [B, state_dim].")

    batch_size = start_state.shape[0]
    n_uavs = int(n_uavs)
    uav_pos_dim = n_uavs * 2
    if start_state.shape[1] < uav_pos_dim:
        raise ValueError("start_state is too short to contain normalized UAV positions.")

    current_pos_norm = start_state[:, :uav_pos_dim].reshape(batch_size, n_uavs, 2)
    current_pos = current_pos_norm * float(range_pos)

    w, mean, log_std, raw_w = actor.sample(start_state)
    generator = TorchLinearWaypointTrajectoryGenerator(
        H=H,
        n_uavs=n_uavs,
        max_move_per_slot=max_move_per_slot,
        range_pos=range_pos,
    )
    continuous_action_seq = generator.generate(current_pos=current_pos, w=w)
    smoothness_loss = generator.smoothness_penalty(continuous_action_seq)
    curvature_loss = generator.curvature_continuity_penalty(continuous_action_seq)

    critic_requires_grad = [param.requires_grad for param in critic.parameters()]
    for param in critic.parameters():
        param.requires_grad_(False)
    _, prefix_q_values = critic(start_state, continuous_action_seq)
    for param, requires_grad in zip(critic.parameters(), critic_requires_grad):
        param.requires_grad_(requires_grad)

    value_objective = -prefix_q_values.mean()
    actor_loss = value_objective + float(smoothness_weight) * smoothness_loss
    actor_loss = actor_loss + float(curvature_weight) * curvature_loss
    info = dict(
        current_pos=current_pos,
        raw_w=raw_w,
        mean=mean,
        log_std=log_std,
        w=w,
        continuous_action_seq=continuous_action_seq,
        prefix_q_values=prefix_q_values,
        value_objective=value_objective,
        smoothness_loss=smoothness_loss,
        smoothness_weight=float(smoothness_weight),
        curvature_loss=curvature_loss,
        curvature_weight=float(curvature_weight),
    )
    return actor_loss, info
