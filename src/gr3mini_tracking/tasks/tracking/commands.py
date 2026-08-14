"""GR3 motion command with the extra reference channels used by DiffCritic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg


def temporal_bin_count(frame_count: int, step_dt: float, bin_duration_s: float) -> int:
    """Return the number of fixed-duration temporal bins for a reference clip."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")
    if bin_duration_s <= 0.0:
        raise ValueError(f"bin_duration_s must be positive, got {bin_duration_s}")
    return max(1, math.ceil(frame_count * step_dt / bin_duration_s))


def adaptive_sampling_probabilities(difficulty: torch.Tensor, uniform_ratio: float) -> torch.Tensor:
    """Mix a fixed uniform mass with a normalized non-negative difficulty distribution."""
    if difficulty.ndim != 1 or difficulty.numel() == 0:
        raise ValueError("difficulty must be a non-empty one-dimensional tensor")
    if not 0.0 <= uniform_ratio <= 1.0:
        raise ValueError(f"uniform_ratio must be in [0, 1], got {uniform_ratio}")

    uniform = torch.full_like(difficulty, 1.0 / difficulty.numel())
    focused = difficulty.clamp_min(0.0)
    focused_total = focused.sum()
    focused = focused / focused_total if focused_total > 0.0 else uniform
    return uniform_ratio * uniform + (1.0 - uniform_ratio) * focused


class Gr3MotionLoader:
    """Load mjlab tracking arrays plus source-local root/site channels."""

    def __init__(self, motion_file: str, body_indexes: torch.Tensor, device: str) -> None:
        data = np.load(motion_file)
        self.joint_pos = torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.as_tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.as_tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.as_tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.as_tensor(
            data["body_lin_vel_w"], dtype=torch.float32, device=device
        )
        self._body_ang_vel_w = torch.as_tensor(
            data["body_ang_vel_w"], dtype=torch.float32, device=device
        )
        self.root_lin_vel_w = torch.as_tensor(
            data["root_lin_vel_w"], dtype=torch.float32, device=device
        )
        self.root_ang_vel_b = torch.as_tensor(
            data["root_ang_vel_b"], dtype=torch.float32, device=device
        )
        self.root_projected_gravity_b = torch.as_tensor(
            data["root_projected_gravity_b"], dtype=torch.float32, device=device
        )
        self.feet_height_w = torch.as_tensor(
            data["feet_height_w"], dtype=torch.float32, device=device
        )
        self._body_indexes = body_indexes
        self.body_pos_w = self._body_pos_w[:, body_indexes]
        self.body_quat_w = self._body_quat_w[:, body_indexes]
        self.body_lin_vel_w = self._body_lin_vel_w[:, body_indexes]
        self.body_ang_vel_w = self._body_ang_vel_w[:, body_indexes]
        self.time_step_total = self.joint_pos.shape[0]


class Gr3MotionCommand(MotionCommand):
    """MotionCommand exposing indexed future frames without copying data."""

    cfg: Gr3MotionCommandCfg

    def __init__(self, cfg: Gr3MotionCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self.motion = Gr3MotionLoader(cfg.motion_file, self.body_indexes, self.device)
        if cfg.sampling_mode == "adaptive":
            self._configure_adaptive_sampling()

    def future_indexes(self, offsets: tuple[int, ...] = (1, 2, 3, 4, 5)) -> torch.Tensor:
        offset = torch.tensor(offsets, dtype=torch.long, device=self.device)
        return torch.clamp(
            self.time_steps[:, None] + offset[None, :],
            max=self.motion.time_step_total - 1,
        )

    def _configure_adaptive_sampling(self) -> None:
        cfg = self.cfg
        self.bin_count = temporal_bin_count(
            self.motion.time_step_total, self._env.step_dt, cfg.adaptive_bin_duration_s
        )
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._bin_tracking_error = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._current_bin_tracking_error = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._current_bin_visit_count = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self.metrics["sampling_uniform_mass"] = torch.zeros(self.num_envs, device=self.device)

    def _time_step_to_bin(self, time_steps: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
            0,
            self.bin_count - 1,
        )

    def _update_metrics(self) -> None:
        super()._update_metrics()
        if self.cfg.sampling_mode != "adaptive":
            return

        metrics = self.metrics
        body_linvel_error = torch.linalg.vector_norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        body_angvel_error = torch.linalg.vector_norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)
        root_pos_error = torch.linalg.vector_norm(
            self.body_pos_w[:, 0] - self.robot.data.root_link_pos_w, dim=-1
        )
        tracking_error = torch.stack(
            (
                metrics["error_body_pos"] / 0.30,
                metrics["error_body_rot"] / 0.40,
                body_linvel_error / 1.00,
                body_angvel_error / 3.14,
                root_pos_error / 0.30,
                metrics["error_anchor_rot"] / 0.40,
            ),
            dim=-1,
        ).mean(dim=-1)
        tracking_error = tracking_error.clamp(max=self.cfg.adaptive_tracking_error_clip)
        current_bins = self._time_step_to_bin(self.time_steps)
        self._current_bin_tracking_error.scatter_add_(0, current_bins, tracking_error)
        self._current_bin_visit_count.scatter_add_(0, current_bins, torch.ones_like(tracking_error))

    def _update_adaptive_scores(self) -> None:
        alpha = self.cfg.adaptive_alpha
        valid = self._current_bin_visit_count > 0.0
        if torch.any(valid):
            current_error = (
                self._current_bin_tracking_error[valid] / self._current_bin_visit_count[valid]
            )
            self._bin_tracking_error[valid] = (1.0 - alpha) * self._bin_tracking_error[
                valid
            ] + alpha * current_error
        self.bin_failed_count.mul_(1.0 - alpha).add_(self._current_bin_failed, alpha=alpha)
        self._current_bin_tracking_error.zero_()
        self._current_bin_visit_count.zero_()
        self._current_bin_failed.zero_()

    def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bins = self._time_step_to_bin(self.time_steps[env_ids][episode_failed])
            self._current_bin_failed.add_(
                torch.bincount(current_bins, minlength=self.bin_count).to(dtype=torch.float)
            )

        difficulty = self.bin_failed_count + (
            self.cfg.adaptive_tracking_error_weight * self._bin_tracking_error
        )
        sampling_probabilities = adaptive_sampling_probabilities(
            difficulty, self.cfg.adaptive_uniform_ratio
        )
        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        self.time_steps[env_ids] = (
            (sampled_bins + torch.rand(len(env_ids), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        entropy = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        normalized_entropy = entropy / math.log(self.bin_count) if self.bin_count > 1 else 1.0
        self.metrics["sampling_entropy"][:] = normalized_entropy
        top_probability, top_bin = sampling_probabilities.max(dim=0)
        self.metrics["sampling_top1_prob"][:] = top_probability
        self.metrics["sampling_top1_bin"][:] = top_bin.float() / self.bin_count
        self.metrics["sampling_uniform_mass"][:] = self.cfg.adaptive_uniform_ratio

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.time_steps += 1
        else:
            self.time_steps[env_ids] += 1
        wrap_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        if wrap_ids.numel() > 0:
            self._resample_command(wrap_ids)

        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()

        self.update_relative_body_poses()
        if env_ids is None and self.cfg.sampling_mode == "adaptive":
            self._update_adaptive_scores()

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.motion.root_lin_vel_w[self.time_steps]

    @property
    def root_ang_vel_b(self) -> torch.Tensor:
        return self.motion.root_ang_vel_b[self.time_steps]

    @property
    def root_projected_gravity_b(self) -> torch.Tensor:
        return self.motion.root_projected_gravity_b[self.time_steps]

    @property
    def feet_height_w(self) -> torch.Tensor:
        return self.motion.feet_height_w[self.time_steps]


@dataclass(kw_only=True)
class Gr3MotionCommandCfg(MotionCommandCfg):
    adaptive_bin_duration_s: float = 0.25
    """Duration of one adaptive-sampling bin in reference time."""

    adaptive_tracking_error_weight: float = 0.25
    """Relative contribution of normalized tracking error to sampling difficulty."""

    adaptive_tracking_error_clip: float = 5.0
    """Upper bound for one step's normalized tracking-error contribution."""

    def build(self, env) -> Gr3MotionCommand:
        return Gr3MotionCommand(self, env)
