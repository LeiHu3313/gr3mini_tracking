"""Tracking rewards matching the GR3Mini211 GeneralDR DiffCritic task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    euler_xyz_from_quat,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
)

from gr3mini_tracking.robots.gr3mini211 import (
    DOF_VEL_LIMITS,
    FEET_BODY_NAMES,
    FOOT_SITE_NAMES,
    LOWER_BODY_NAMES,
    SHOULDER_BODY_NAMES,
    UPPER_BODY_NAMES,
)

from .actions import ReferenceResidualJointPositionAction
from .commands import Gr3MotionCommand

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def _robot(env: ManagerBasedRlEnv) -> Entity:
    return env.scene["robot"]


def _command(env: ManagerBasedRlEnv) -> Gr3MotionCommand:
    return cast(Gr3MotionCommand, env.command_manager.get_term("motion"))


def _action(env: ManagerBasedRlEnv) -> ReferenceResidualJointPositionAction:
    return cast(ReferenceResidualJointPositionAction, env.action_manager.get_term("joint_pos"))


def _body_indexes(command: Gr3MotionCommand, body_names: tuple[str, ...]) -> list[int]:
    return [command.cfg.body_names.index(name) for name in body_names]


def _local_body_position_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    command = _command(env)
    robot = _robot(env)
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
    return reference - current


def body_position_tracking_exp(
    env: ManagerBasedRlEnv, body_names: tuple[str, ...], sigma: float
) -> torch.Tensor:
    command = _command(env)
    indexes = _body_indexes(command, body_names)
    error = _local_body_position_error(env)[:, indexes]
    return torch.exp(-error.abs().sum(dim=(-2, -1)) / sigma)


def feet_position_tracking_exp(env: ManagerBasedRlEnv, sigma: float = 1.0) -> torch.Tensor:
    return body_position_tracking_exp(env, FEET_BODY_NAMES, sigma)


def body_orientation_tracking_exp(env: ManagerBasedRlEnv, sigma: float) -> torch.Tensor:
    command = _command(env)
    count = len(command.cfg.body_names)
    current_root_inv = quat_inv(_robot(env).data.root_link_quat_w)
    reference_root_inv = quat_inv(command.body_quat_w[:, 0])
    current_local = quat_mul(
        current_root_inv[:, None, :].expand(-1, count, -1),
        command.robot_body_quat_w,
    )
    reference_local = quat_mul(
        reference_root_inv[:, None, :].expand(-1, count, -1),
        command.body_quat_w,
    )
    angle = quat_error_magnitude(reference_local, current_local)
    return torch.exp(-angle.mean(dim=-1) / sigma)


def body_linear_velocity_tracking_exp(env: ManagerBasedRlEnv, sigma: float) -> torch.Tensor:
    command = _command(env)
    count = len(command.cfg.body_names)
    current = quat_apply_inverse(
        _robot(env).data.root_link_quat_w[:, None, :].expand(-1, count, -1),
        command.robot_body_lin_vel_w,
    )
    reference = quat_apply_inverse(
        command.body_quat_w[:, 0, None, :].expand(-1, count, -1),
        command.body_lin_vel_w,
    )
    error = reference - current
    return torch.exp(-error.square().mean(dim=(-2, -1)) / sigma)


def body_angular_velocity_tracking_exp(env: ManagerBasedRlEnv, sigma: float) -> torch.Tensor:
    command = _command(env)
    count = len(command.cfg.body_names)
    current = quat_apply_inverse(
        _robot(env).data.root_link_quat_w[:, None, :].expand(-1, count, -1),
        command.robot_body_ang_vel_w,
    )
    reference = quat_apply_inverse(
        command.body_quat_w[:, 0, None, :].expand(-1, count, -1),
        command.body_ang_vel_w,
    )
    error = reference - current
    return torch.exp(-error.square().mean(dim=(-2, -1)) / sigma)


def joint_position_tracking_exp(env: ManagerBasedRlEnv, sigma: float) -> torch.Tensor:
    command = _command(env)
    error = command.joint_pos - command.robot_joint_pos
    return torch.exp(-error.abs().sum(dim=-1) / sigma)


def joint_velocity_tracking_exp(env: ManagerBasedRlEnv, sigma: float) -> torch.Tensor:
    command = _command(env)
    error = command.joint_vel - command.robot_joint_vel
    return torch.exp(-(error.abs().sum(dim=-1) * env.step_dt) / sigma)


def root_linear_velocity_tracking_exp(env: ManagerBasedRlEnv, sigma: float) -> torch.Tensor:
    command = _command(env)
    robot = _robot(env)
    ref = quat_apply_inverse(command.body_quat_w[:, 0], command.root_lin_vel_w)
    error = ref - robot.data.root_link_lin_vel_b
    return torch.exp(-error.abs().sum(dim=-1) / sigma)


def root_angular_velocity_tracking_exp(env: ManagerBasedRlEnv, sigma: float) -> torch.Tensor:
    command = _command(env)
    error = command.root_ang_vel_b - _robot(env).data.root_link_ang_vel_b
    return torch.exp(-error.abs().sum(dim=-1) / sigma)


def torso_roll_pitch_tracking_exp(env: ManagerBasedRlEnv, sigma: float) -> torch.Tensor:
    command = _command(env)
    current = euler_xyz_from_quat(_robot(env).data.root_link_quat_w)
    reference = euler_xyz_from_quat(command.body_quat_w[:, 0])
    error = torch.stack([reference[0] - current[0], reference[1] - current[1]], dim=-1)
    return torch.exp(-error.abs().sum(dim=-1) / sigma)


def root_height_tracking_exp(env: ManagerBasedRlEnv, sigma: float) -> torch.Tensor:
    command = _command(env)
    error = command.body_pos_w[:, 0, 2] - _robot(env).data.root_link_pos_w[:, 2]
    return torch.exp(-error.abs() / sigma)


def shoulder_height_tracking_exp(env: ManagerBasedRlEnv, sigma: float) -> torch.Tensor:
    command = _command(env)
    indexes = _body_indexes(command, SHOULDER_BODY_NAMES)
    error = command.body_pos_w[:, indexes, 2] - command.robot_body_pos_w[:, indexes, 2]
    return torch.exp(-error.abs().mean(dim=-1) / sigma)


def feet_height_tracking_exp(env: ManagerBasedRlEnv, sigma: float) -> torch.Tensor:
    command = _command(env)
    robot = _robot(env)
    site_ids = torch.tensor(
        robot.find_sites(FOOT_SITE_NAMES, preserve_order=True)[0],
        dtype=torch.long,
        device=env.device,
    )
    error = command.feet_height_w - robot.data.site_pose_w[:, site_ids, 2]
    return torch.exp(-error.abs().sum(dim=-1) / sigma)


def target_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    action = _action(env)
    return (action.last_target - action.previous_target).square().sum(dim=-1)


def joint_torque_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _robot(env).data.qfrc_actuator.square().sum(dim=-1)


def joint_smoothness(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = _robot(env)
    return (0.02 * robot.data.joint_vel.square() + robot.data.joint_acc.square()).sum(dim=-1)


def joint_position_soft_limit(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = _robot(env)
    limits = robot.data.soft_joint_pos_limits
    below = torch.clamp(limits[..., 0] - robot.data.joint_pos, min=0.0)
    above = torch.clamp(robot.data.joint_pos - limits[..., 1], min=0.0)
    return torch.clamp((below + above).sum(dim=-1), max=100.0)


def joint_velocity_limit(env: ManagerBasedRlEnv) -> torch.Tensor:
    limits = torch.tensor(DOF_VEL_LIMITS, device=env.device)
    return torch.clamp(_robot(env).data.joint_vel.abs() - limits, min=0.0, max=1.0).sum(dim=-1)


def self_collision_count(env: ManagerBasedRlEnv) -> torch.Tensor:
    sensor = cast(ContactSensor, env.scene["self_collision"])
    assert sensor.data.found is not None
    return (sensor.data.found > 0).float().sum(dim=-1)


def terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.termination_manager.terminated.float()


UPPER_REWARD_BODIES = UPPER_BODY_NAMES
LOWER_REWARD_BODIES = LOWER_BODY_NAMES
