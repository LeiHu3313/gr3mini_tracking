"""GR3 motion command with the extra reference channels used by DiffCritic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg


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

    def __init__(self, cfg: Gr3MotionCommandCfg, env) -> None:
        super().__init__(cfg, env)
        self.motion = Gr3MotionLoader(cfg.motion_file, self.body_indexes, self.device)

    def future_indexes(self, offsets: tuple[int, ...] = (1, 2, 3, 4, 5)) -> torch.Tensor:
        offset = torch.tensor(offsets, dtype=torch.long, device=self.device)
        return torch.clamp(
            self.time_steps[:, None] + offset[None, :],
            max=self.motion.time_step_total - 1,
        )

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
    def build(self, env) -> Gr3MotionCommand:
        return Gr3MotionCommand(self, env)
