import numpy as np
import torch


class TrajectorySplinePolicy:
    """Generate C2-smooth polynomial-spline displacement sequences toward relative endpoints."""

    def __init__(self, H, n_uavs, max_move_per_slot, range_pos, boundary_margin=None, eps=1e-8):
        self.H = int(H)
        self.n_uavs = int(n_uavs)
        self.max_move_per_slot = float(max_move_per_slot)
        self.range_pos = float(range_pos)
        self.boundary_margin = (
            float(boundary_margin)
            if boundary_margin is not None
            else max(float(max_move_per_slot), 1.0)
        )
        self.eps = float(eps)

        if self.H <= 0:
            raise ValueError("H must be positive.")
        if self.n_uavs <= 0:
            raise ValueError("n_uavs must be positive.")
        if self.max_move_per_slot < 0:
            raise ValueError("max_move_per_slot must be non-negative.")
        if self.range_pos <= 0:
            raise ValueError("range_pos must be positive.")
        if self.boundary_margin <= 0:
            raise ValueError("boundary_margin must be positive.")

    def generate(self, current_pos, w):
        """
        Build a smooth continuous displacement sequence with shape [H, n_uavs, 2].

        Args:
            current_pos: Current UAV positions, shape [n_uavs, 2].
            w: Relative endpoint displacement parameters, shape [n_uavs, 2].
        """
        current_pos = self._validate_array("current_pos", current_pos)
        w = self._validate_array("w", w)

        target_pos = self._smooth_project_position(current_pos + w)
        prev_pos = current_pos.copy()
        actions = []

        for h in range(self.H):
            tau = np.float32((h + 1) / self.H)
            basis = self._basis(tau)
            desired_pos = current_pos + basis * (target_pos - current_pos)
            displacement = self._clip_displacement(desired_pos - prev_pos)
            next_pos = prev_pos + displacement
            actions.append((next_pos - prev_pos).astype(np.float32))
            prev_pos = next_pos

        return np.stack(actions, axis=0).astype(np.float32)

    def smoothness_penalty(self, action_seq):
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.ndim != 3 or action_seq.shape[1:] != (self.n_uavs, 2):
            raise ValueError(f"action_seq must have shape [H, {self.n_uavs}, 2].")
        if action_seq.shape[0] < 3:
            return 0.0
        acceleration = action_seq[2:] - 2.0 * action_seq[1:-1] + action_seq[:-2]
        return float(np.mean(np.square(acceleration)))

    def curvature_continuity_penalty(self, action_seq):
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.ndim != 3 or action_seq.shape[1:] != (self.n_uavs, 2):
            raise ValueError(f"action_seq must have shape [H, {self.n_uavs}, 2].")
        if action_seq.shape[0] < 4:
            return 0.0
        jerk = action_seq[3:] - 3.0 * action_seq[2:-1] + 3.0 * action_seq[1:-2] - action_seq[:-3]
        return float(np.mean(np.square(jerk)))

    def _validate_array(self, name, value):
        value = np.asarray(value, dtype=np.float32)
        expected_shape = (self.n_uavs, 2)
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}.")
        return value

    def _clip_displacement(self, displacement):
        norms = np.linalg.norm(displacement, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.max_move_per_slot / (norms + self.eps))
        return displacement * scale.astype(np.float32)

    @staticmethod
    def _basis(tau):
        # Quintic smootherstep: position is C2-continuous with zero endpoint velocity/acceleration.
        return np.float32(10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5)

    def _smooth_project_position(self, position):
        position = np.asarray(position, dtype=np.float32)
        margin = np.float32(self.boundary_margin)
        projected = position.copy()

        low_mask = projected < 0.0
        if np.any(low_mask):
            projected[low_mask] = margin * (1.0 - np.exp(projected[low_mask] / margin))

        high_mask = projected > self.range_pos
        if np.any(high_mask):
            overflow = projected[high_mask] - self.range_pos
            projected[high_mask] = self.range_pos - margin * (1.0 - np.exp(-overflow / margin))

        return projected.astype(np.float32)


class TrajectoryPolynomialGenerator(TrajectorySplinePolicy):
    """Backward-compatible name for the final polynomial spline policy."""


class LinearWaypointTrajectoryGenerator(TrajectorySplinePolicy):
    """Backward-compatible name for the upgraded polynomial spline policy."""


class TorchTrajectorySplinePolicy:
    """Differentiable torch version of the C2-smooth polynomial spline policy."""

    def __init__(self, H, n_uavs, max_move_per_slot, range_pos, boundary_margin=None, eps=1e-8):
        self.H = int(H)
        self.n_uavs = int(n_uavs)
        self.max_move_per_slot = float(max_move_per_slot)
        self.range_pos = float(range_pos)
        self.boundary_margin = (
            float(boundary_margin)
            if boundary_margin is not None
            else max(float(max_move_per_slot), 1.0)
        )
        self.eps = float(eps)

        if self.H <= 0:
            raise ValueError("H must be positive.")
        if self.n_uavs <= 0:
            raise ValueError("n_uavs must be positive.")
        if self.max_move_per_slot < 0:
            raise ValueError("max_move_per_slot must be non-negative.")
        if self.range_pos <= 0:
            raise ValueError("range_pos must be positive.")
        if self.boundary_margin <= 0:
            raise ValueError("boundary_margin must be positive.")

    def generate(self, current_pos, w):
        """
        Build a differentiable displacement sequence with shape [B, H, n_uavs, 2].

        Args:
            current_pos: Current UAV positions, shape [B, n_uavs, 2].
            w: Relative endpoint displacement parameters, shape [B, n_uavs, 2].
        """
        if not torch.is_tensor(current_pos):
            current_pos = torch.as_tensor(current_pos, dtype=torch.float32)
        if not torch.is_tensor(w):
            w = torch.as_tensor(w, dtype=torch.float32, device=current_pos.device)

        current_pos = current_pos.to(dtype=torch.float32)
        w = w.to(device=current_pos.device, dtype=torch.float32)
        self._validate_tensor("current_pos", current_pos)
        self._validate_tensor("w", w)
        if current_pos.shape[0] != w.shape[0]:
            raise ValueError("current_pos and w must have the same batch size.")

        target_pos = self._smooth_project_position(current_pos + w)
        prev_pos = current_pos
        actions = []

        for h in range(self.H):
            tau = torch.as_tensor((h + 1) / self.H, dtype=current_pos.dtype, device=current_pos.device)
            basis = self._basis(tau)
            desired_pos = current_pos + basis * (target_pos - current_pos)
            displacement = self._clip_displacement(desired_pos - prev_pos)
            next_pos = prev_pos + displacement
            actions.append(next_pos - prev_pos)
            prev_pos = next_pos

        return torch.stack(actions, dim=1)

    def smoothness_penalty(self, action_seq):
        if action_seq.shape[1] < 3:
            return action_seq.new_zeros(())
        acceleration = action_seq[:, 2:] - 2.0 * action_seq[:, 1:-1] + action_seq[:, :-2]
        return acceleration.pow(2).mean()

    def curvature_continuity_penalty(self, action_seq):
        if action_seq.shape[1] < 4:
            return action_seq.new_zeros(())
        jerk = action_seq[:, 3:] - 3.0 * action_seq[:, 2:-1] + 3.0 * action_seq[:, 1:-2] - action_seq[:, :-3]
        return jerk.pow(2).mean()

    def _validate_tensor(self, name, value):
        expected_tail = (self.n_uavs, 2)
        if value.ndim != 3 or tuple(value.shape[1:]) != expected_tail:
            raise ValueError(f"{name} must have shape [B, {self.n_uavs}, 2], got {tuple(value.shape)}.")

    def _clip_displacement(self, displacement):
        norm = displacement.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_move_per_slot / (norm + self.eps), max=1.0)
        return displacement * scale

    @staticmethod
    def _basis(tau):
        # Quintic smootherstep: C2 position curve with continuous velocity and acceleration.
        return 10.0 * tau.pow(3) - 15.0 * tau.pow(4) + 6.0 * tau.pow(5)

    def _smooth_project_position(self, position):
        margin = torch.as_tensor(self.boundary_margin, dtype=position.dtype, device=position.device)
        range_pos = torch.as_tensor(self.range_pos, dtype=position.dtype, device=position.device)

        low_projection = margin * (1.0 - torch.exp(position / margin))
        overflow = position - range_pos
        high_projection = range_pos - margin * (1.0 - torch.exp(-overflow / margin))

        projected = torch.where(position < 0.0, low_projection, position)
        projected = torch.where(projected > range_pos, high_projection, projected)
        return projected


class TorchTrajectoryPolynomialGenerator(TorchTrajectorySplinePolicy):
    """Backward-compatible name for the final differentiable polynomial spline policy."""


class TorchLinearWaypointTrajectoryGenerator(TorchTrajectorySplinePolicy):
    """Backward-compatible name for the upgraded differentiable polynomial spline policy."""
