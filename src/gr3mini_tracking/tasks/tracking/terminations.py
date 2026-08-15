"""Reference-relative termination conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from gr3mini_tracking.robots.gr3mini211 import SHOULDER_BODY_NAMES

from .commands import Gr3MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv) -> Gr3MotionCommand:
    return cast(Gr3MotionCommand, env.command_manager.get_term("motion"))


def bad_root_height(env: ManagerBasedRlEnv, threshold: float = 0.3) -> torch.Tensor:
    command = _command(env)
    current = command.robot.data.root_link_pos_w[:, 2]
    reference = command.body_pos_w[:, 0, 2]
    return (current - reference).abs() > threshold


def bad_shoulder_height(env: ManagerBasedRlEnv, threshold: float = 0.5) -> torch.Tensor:
    command = _command(env)
    idx = command.cfg.body_names.index(SHOULDER_BODY_NAMES[0])
    return (command.robot_body_pos_w[:, idx, 2] - command.body_pos_w[:, idx, 2]).abs() > threshold


def bad_body_position(env: ManagerBasedRlEnv, threshold: float = 0.5) -> torch.Tensor:
    command = _command(env)
    robot = command.robot
    count = len(command.cfg.body_names)
    current_root_pos = robot.data.root_link_pos_w
    current_root_quat = robot.data.root_link_quat_w
    reference_root_pos = command.body_pos_w[:, 0]
    reference_root_quat = command.body_quat_w[:, 0]
    current = quat_apply_inverse(
        current_root_quat[:, None, :].expand(-1, count, -1),
        command.robot_body_pos_w - current_root_pos[:, None, :],
    )
    reference = quat_apply_inverse(
        reference_root_quat[:, None, :].expand(-1, count, -1),
        command.body_pos_w - reference_root_pos[:, None, :],
    )
    return torch.any(torch.linalg.vector_norm(reference - current, dim=-1) > threshold, dim=-1)


def invalid_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    data = _command(env).robot.data
    return torch.isnan(data.joint_pos).any(dim=-1) | torch.isnan(data.joint_vel).any(dim=-1)
