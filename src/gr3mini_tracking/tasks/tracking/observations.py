"""Exact DiffCritic observation layouts from the Any2Track GR3Mini task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
)

from gr3mini_tracking.robots.gr3mini211 import FOOT_SITE_NAMES, TRACKED_BODY_NAMES

from .actions import ReferenceResidualJointPositionAction
from .commands import Gr3MotionCommand

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

FRAME_DIM = 81
FUTURE_FRAME_DIM = 37
ACTOR_HISTORY_LENGTH = 6
FUTURE_OFFSETS = (1, 2, 3, 4, 5)
ACTOR_OBS_DIM = ACTOR_HISTORY_LENGTH * FRAME_DIM + len(FUTURE_OFFSETS) * FUTURE_FRAME_DIM
CRITIC_CURRENT_DIM = 477
CRITIC_HISTORY_LENGTH = 6
CRITIC_HISTORY_DIM = CRITIC_HISTORY_LENGTH * CRITIC_CURRENT_DIM
CRITIC_OBS_DIM = CRITIC_HISTORY_DIM + len(FUTURE_OFFSETS) * FUTURE_FRAME_DIM
ADAPTER_HISTORY_LENGTH = 79
WORLD_STATE_DIM = 57
VELOCITY_SCALE = 0.05


def _robot(env: ManagerBasedRlEnv) -> Entity:
    return env.scene["robot"]


def _command(env: ManagerBasedRlEnv) -> Gr3MotionCommand:
    return cast(Gr3MotionCommand, env.command_manager.get_term("motion"))


def _action(env: ManagerBasedRlEnv) -> ReferenceResidualJointPositionAction:
    return cast(
        ReferenceResidualJointPositionAction,
        env.action_manager.get_term("joint_pos"),
    )


def _uniform_noise(value: torch.Tensor, magnitude: float) -> torch.Tensor:
    if magnitude == 0.0:
        return value
    return value + (2.0 * torch.rand_like(value) - 1.0) * magnitude


def actor_frame(env: ManagerBasedRlEnv, noisy: bool = True) -> torch.Tensor:
    """One 81-D state/target frame in the source field order."""
    robot = _robot(env)
    gravity = robot.data.projected_gravity_b
    gyro = robot.data.root_link_ang_vel_b
    joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
    joint_vel = robot.data.joint_vel
    if noisy:
        gravity = _uniform_noise(gravity, 0.05)
        gyro = _uniform_noise(gyro, 0.2)
        joint_pos = _uniform_noise(joint_pos, 0.03)
        joint_vel = _uniform_noise(joint_vel, 1.5)
    frame = torch.cat(
        [
            gravity,
            gyro * VELOCITY_SCALE,
            joint_pos,
            joint_vel * VELOCITY_SCALE,
            _action(env).last_target,
        ],
        dim=-1,
    )
    if frame.shape[-1] != FRAME_DIM:
        raise RuntimeError(f"actor frame has {frame.shape[-1]} dims, expected {FRAME_DIM}")
    return torch.nan_to_num(frame)


def adapter_history_frame(env: ManagerBasedRlEnv, noisy: bool = True) -> torch.Tensor:
    """One adapter history frame: gyro, gravity, q, qd, previous target."""
    robot = _robot(env)
    gyro = robot.data.root_link_ang_vel_b
    gravity = robot.data.projected_gravity_b
    joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
    joint_vel = robot.data.joint_vel
    if noisy:
        gyro = _uniform_noise(gyro, 0.2)
        gravity = _uniform_noise(gravity, 0.05)
        joint_pos = _uniform_noise(joint_pos, 0.03)
        joint_vel = _uniform_noise(joint_vel, 1.5)
    frame = torch.cat(
        [
            gyro * VELOCITY_SCALE,
            gravity,
            joint_pos,
            joint_vel * VELOCITY_SCALE,
            _action(env).last_target,
        ],
        dim=-1,
    )
    if frame.shape[-1] != FRAME_DIM:
        raise RuntimeError(
            f"adapter history frame has {frame.shape[-1]} dims, expected {FRAME_DIM}"
        )
    return torch.nan_to_num(frame)


def _future_components(command: Gr3MotionCommand) -> tuple[torch.Tensor, ...]:
    indexes = command.future_indexes(FUTURE_OFFSETS)
    motion = command.motion
    joint_pos = motion.joint_pos[indexes]
    root_pos = motion.body_pos_w[indexes, 0]
    root_quat = motion.body_quat_w[indexes, 0]
    root_lin_vel_w = motion.root_lin_vel_w[indexes]
    root_lin_vel_b = quat_apply_inverse(root_quat, root_lin_vel_w)
    root_ang_vel_b = motion.root_ang_vel_b[indexes]
    gravity_b = motion.root_projected_gravity_b[indexes]
    feet_height = motion.feet_height_w[indexes]
    return (
        joint_pos,
        root_pos[..., 2:3],
        root_lin_vel_b * VELOCITY_SCALE,
        root_ang_vel_b * VELOCITY_SCALE,
        gravity_b,
        feet_height,
    )


def future_raw_reference(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Five raw future frames, each [q, height, v, omega, gravity, feet-z]."""
    future = torch.cat(_future_components(_command(env)), dim=-1)
    if future.shape[-1] != FUTURE_FRAME_DIM:
        raise RuntimeError(f"future frame has {future.shape[-1]} dims, expected {FUTURE_FRAME_DIM}")
    return torch.nan_to_num(future.flatten(start_dim=1))


def current_tracking_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Current 37-D state subtracted from each DiffCritic future frame."""
    robot = _robot(env)
    site_ids = torch.tensor(
        robot.find_sites(FOOT_SITE_NAMES, preserve_order=True)[0],
        device=env.device,
        dtype=torch.long,
    )
    state = torch.cat(
        [
            robot.data.joint_pos,
            robot.data.root_link_pos_w[:, 2:3],
            robot.data.root_link_lin_vel_b * VELOCITY_SCALE,
            robot.data.root_link_ang_vel_b * VELOCITY_SCALE,
            robot.data.projected_gravity_b,
            robot.data.site_pose_w[:, site_ids, 2],
        ],
        dim=-1,
    )
    if state.shape[-1] != FUTURE_FRAME_DIM:
        raise RuntimeError(
            f"current tracking state has {state.shape[-1]} dims, expected {FUTURE_FRAME_DIM}"
        )
    return state


def future_relative_goal(env: ManagerBasedRlEnv) -> torch.Tensor:
    future = future_raw_reference(env).reshape(-1, len(FUTURE_OFFSETS), FUTURE_FRAME_DIM)
    return (future - current_tracking_state(env)[:, None, :]).flatten(start_dim=1)


def _body_local_state(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, ...]:
    command = _command(env)
    robot = _robot(env)
    body_pos = command.robot_body_pos_w
    body_quat = command.robot_body_quat_w
    body_lin_vel = command.robot_body_lin_vel_w
    body_ang_vel = command.robot_body_ang_vel_w
    root_pos = robot.data.root_link_pos_w
    root_quat = robot.data.root_link_quat_w
    root_quat_inv = quat_inv(root_quat)
    count = len(TRACKED_BODY_NAMES)
    root_quat_expanded = root_quat[:, None, :].expand(-1, count, -1)
    root_quat_inv_expanded = root_quat_inv[:, None, :].expand(-1, count, -1)
    local_pos = quat_apply_inverse(root_quat_expanded, body_pos - root_pos[:, None, :])
    local_quat = quat_mul(root_quat_inv_expanded, body_quat)
    local_rot = matrix_from_quat(local_quat)
    local_rot_6d = local_rot[..., :, :2].transpose(-2, -1).reshape(env.num_envs, -1)
    local_lin_vel = quat_apply_inverse(root_quat_expanded, body_lin_vel)
    local_ang_vel = quat_apply_inverse(root_quat_expanded, body_ang_vel)
    return local_pos, local_rot_6d, local_lin_vel, local_ang_vel


def feet_contact(env: ManagerBasedRlEnv, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    sensor = cast(ContactSensor, env.scene[sensor_name])
    assert sensor.data.found is not None
    return (sensor.data.found > 0).to(dtype=torch.float32)


def critic_current_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    """477-D uncorrupted privileged state in the source order."""
    robot = _robot(env)
    body_pos, body_rot_6d, body_lin_vel, body_ang_vel = _body_local_state(env)
    state = torch.cat(
        [
            robot.data.projected_gravity_b,
            robot.data.root_link_ang_vel_b * VELOCITY_SCALE,
            robot.data.root_link_lin_vel_b * VELOCITY_SCALE,
            robot.data.root_link_pos_w[:, 2:3],
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel * VELOCITY_SCALE,
            _action(env).last_target,
            feet_contact(env),
            body_pos.flatten(start_dim=1),
            body_rot_6d,
            body_lin_vel.flatten(start_dim=1) * VELOCITY_SCALE,
            body_ang_vel.flatten(start_dim=1) * VELOCITY_SCALE,
        ],
        dim=-1,
    )
    if state.shape[-1] != CRITIC_CURRENT_DIM:
        raise RuntimeError(
            f"critic current state has {state.shape[-1]} dims, expected {CRITIC_CURRENT_DIM}"
        )
    return torch.nan_to_num(state)


def world_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    """57-D current state used as the world-model input/target."""
    robot = _robot(env)
    state = torch.cat(
        [
            robot.data.root_link_ang_vel_b * VELOCITY_SCALE,
            robot.data.projected_gravity_b,
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel * VELOCITY_SCALE,
            robot.data.root_link_pos_w[:, 2:3],
        ],
        dim=-1,
    )
    if state.shape[-1] != WORLD_STATE_DIM:
        raise RuntimeError(f"world state has {state.shape[-1]} dims, expected {WORLD_STATE_DIM}")
    return torch.nan_to_num(state)


def reference_world_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    command = _command(env)
    state = torch.cat(
        [
            command.root_ang_vel_b * VELOCITY_SCALE,
            command.root_projected_gravity_b,
            command.joint_pos - command.robot.data.default_joint_pos,
            command.joint_vel * VELOCITY_SCALE,
            command.body_pos_w[:, 0, 2:3],
        ],
        dim=-1,
    )
    if state.shape[-1] != WORLD_STATE_DIM:
        raise RuntimeError(
            f"reference world state has {state.shape[-1]} dims, expected {WORLD_STATE_DIM}"
        )
    return torch.nan_to_num(state)


assert ACTOR_OBS_DIM == 671
assert CRITIC_OBS_DIM == 3047
