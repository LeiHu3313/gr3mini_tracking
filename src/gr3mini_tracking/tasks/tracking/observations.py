"""Observation terms for the deployable actor and asymmetric critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
)

from gr3mini_tracking.robots.gr3mini211 import (
    FOOT_SITE_NAMES,
    TORSO_BODY_NAME,
    TRACKED_BODY_NAMES,
)

from .actions import ReferenceResidualJointPositionAction
from .commands import Gr3MotionCommand

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


# Observation contract
FRAME_DIM = 81  # Adapter history frame
ACTOR_STATE_DIM = 56
ACTION_TARGET_DIM = 25
ACTOR_REFERENCE_DIM = 40
CRITIC_PRIVILEGED_STATE_DIM = 398
CRITIC_TRACKING_ERROR_DIM = 405
ACTOR_HISTORY_LENGTH = 6
FUTURE_OFFSETS = (1, 2, 3, 4, 5)
ADAPTER_HISTORY_LENGTH = 79
WORLD_STATE_DIM = 57
VELOCITY_SCALE = 0.05

ACTOR_STATE_HISTORY_DIM = ACTOR_HISTORY_LENGTH * ACTOR_STATE_DIM
ACTOR_ACTION_HISTORY_DIM = ACTOR_HISTORY_LENGTH * ACTION_TARGET_DIM
ACTOR_OBS_DIM = (
    ACTOR_STATE_HISTORY_DIM + ACTOR_ACTION_HISTORY_DIM + len(FUTURE_OFFSETS) * ACTOR_REFERENCE_DIM
)
CRITIC_OBS_DIM = ACTOR_OBS_DIM + CRITIC_PRIVILEGED_STATE_DIM + CRITIC_TRACKING_ERROR_DIM


@dataclass(frozen=True)
class _Proprioception:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    gyro: torch.Tensor
    gravity: torch.Tensor


@dataclass(frozen=True)
class _FutureReference:
    joint_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel_b: torch.Tensor
    root_ang_vel_b: torch.Tensor
    torso_height: torch.Tensor
    feet_height: torch.Tensor


@dataclass(frozen=True)
class _RootLocalBodies:
    position: torch.Tensor
    quaternion: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor


def _robot(env: ManagerBasedRlEnv) -> Entity:
    return env.scene["robot"]


def _command(env: ManagerBasedRlEnv) -> Gr3MotionCommand:
    return cast(Gr3MotionCommand, env.command_manager.get_term("motion"))


def _action(env: ManagerBasedRlEnv) -> ReferenceResidualJointPositionAction:
    return cast(ReferenceResidualJointPositionAction, env.action_manager.get_term("joint_pos"))


def _check_frame(name: str, value: torch.Tensor, expected_dim: int) -> torch.Tensor:
    if value.shape[-1] != expected_dim:
        raise RuntimeError(f"{name} has {value.shape[-1]} dims, expected {expected_dim}")
    return torch.nan_to_num(value)


def _flatten_frames(name: str, value: torch.Tensor, frame_dim: int) -> torch.Tensor:
    return _check_frame(name, value, frame_dim).flatten(start_dim=1)


def _uniform_noise(value: torch.Tensor, magnitude: float) -> torch.Tensor:
    if magnitude == 0.0:
        return value
    return value + (2.0 * torch.rand_like(value) - 1.0) * magnitude


def _proprioception(env: ManagerBasedRlEnv, noisy: bool) -> _Proprioception:
    robot = _robot(env)
    joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
    joint_vel = robot.data.joint_vel
    gyro = robot.data.root_link_ang_vel_b
    gravity = robot.data.projected_gravity_b
    if noisy:
        joint_pos = _uniform_noise(joint_pos, 0.03)
        joint_vel = _uniform_noise(joint_vel, 1.5)
        gyro = _uniform_noise(gyro, 0.2)
        gravity = _uniform_noise(gravity, 0.05)
    return _Proprioception(joint_pos, joint_vel, gyro, gravity)


def _centered_action_target(env: ManagerBasedRlEnv) -> torch.Tensor:
    return _action(env).last_target - _robot(env).data.default_joint_pos


def _torso_index(command: Gr3MotionCommand) -> int:
    try:
        return command.cfg.body_names.index(TORSO_BODY_NAME)
    except ValueError as error:
        raise RuntimeError(f"motion command does not track {TORSO_BODY_NAME!r}") from error


def _feet_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = _robot(env)
    site_ids = torch.tensor(
        robot.find_sites(FOOT_SITE_NAMES, preserve_order=True)[0],
        device=env.device,
        dtype=torch.long,
    )
    return robot.data.site_pose_w[:, site_ids, 2]


def _rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """Encode a quaternion as the first two matrix columns, column-major."""
    return matrix_from_quat(quat)[..., :, :2].transpose(-2, -1).flatten(start_dim=-2)


def _relative_rotation_6d(current_quat: torch.Tensor, reference_quat: torch.Tensor) -> torch.Tensor:
    """Reference orientation expressed relative to the current orientation."""
    current_inv = quat_inv(current_quat)
    if current_inv.ndim < reference_quat.ndim:
        current_inv = current_inv.unsqueeze(-2).expand_as(reference_quat)
    return _rotation_6d(quat_mul(current_inv, reference_quat))


def _future_reference(command: Gr3MotionCommand) -> _FutureReference:
    """Future reference channels in the reference-root frame."""
    indexes = command.future_indexes(FUTURE_OFFSETS)
    motion = command.motion
    root_quat = motion.body_quat_w[indexes, 0]
    return _FutureReference(
        joint_pos=motion.joint_pos[indexes],
        root_quat=root_quat,
        root_lin_vel_b=quat_apply_inverse(root_quat, motion.root_lin_vel_w[indexes]),
        root_ang_vel_b=motion.root_ang_vel_b[indexes],
        torso_height=motion.body_pos_w[indexes, _torso_index(command), 2:3],
        feet_height=motion.feet_height_w[indexes],
    )


def _root_local_bodies(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
) -> _RootLocalBodies:
    """Convert a whole-body state from world coordinates into its root frame."""
    body_count = body_pos_w.shape[1]
    if body_count != len(TRACKED_BODY_NAMES):
        raise RuntimeError(
            f"body state has {body_count} bodies, expected {len(TRACKED_BODY_NAMES)}"
        )
    root_quat = root_quat_w[:, None, :].expand(-1, body_count, -1)
    root_quat_inv = quat_inv(root_quat_w)[:, None, :].expand_as(root_quat)
    return _RootLocalBodies(
        position=quat_apply_inverse(root_quat, body_pos_w - root_pos_w[:, None, :]),
        quaternion=quat_mul(root_quat_inv, body_quat_w),
        linear_velocity=quat_apply_inverse(root_quat, body_lin_vel_w),
        angular_velocity=quat_apply_inverse(root_quat, body_ang_vel_w),
    )


def _current_root_local_bodies(env: ManagerBasedRlEnv) -> _RootLocalBodies:
    command = _command(env)
    robot = _robot(env)
    return _root_local_bodies(
        command.robot_body_pos_w,
        command.robot_body_quat_w,
        command.robot_body_lin_vel_w,
        command.robot_body_ang_vel_w,
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
    )


def _reference_root_local_bodies(command: Gr3MotionCommand) -> _RootLocalBodies:
    return _root_local_bodies(
        command.body_pos_w,
        command.body_quat_w,
        command.body_lin_vel_w,
        command.body_ang_vel_w,
        command.body_pos_w[:, 0],
        command.body_quat_w[:, 0],
    )


# Actor observations: encoder + IMU + applied action + offline reference only.
def actor_state_frame(env: ManagerBasedRlEnv, noisy: bool = True) -> torch.Tensor:
    """One 56-D deployable frame: q, qd, gyro, projected gravity."""
    state = _proprioception(env, noisy)
    return _check_frame(
        "actor state frame",
        torch.cat(
            [
                state.joint_pos,
                state.joint_vel * VELOCITY_SCALE,
                state.gyro * VELOCITY_SCALE,
                state.gravity,
            ],
            dim=-1,
        ),
        ACTOR_STATE_DIM,
    )


def actor_action_target(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One 25-D centered, physically applied joint-target frame."""
    return _check_frame("actor action target", _centered_action_target(env), ACTION_TARGET_DIM)


def actor_future_reference(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Five 40-D deployable future command frames."""
    robot = _robot(env)
    reference = _future_reference(_command(env))
    frames = torch.cat(
        [
            reference.joint_pos - robot.data.joint_pos[:, None, :],
            reference.root_lin_vel_b * VELOCITY_SCALE,
            (reference.root_ang_vel_b - robot.data.root_link_ang_vel_b[:, None, :])
            * VELOCITY_SCALE,
            _relative_rotation_6d(robot.data.root_link_quat_w, reference.root_quat),
            reference.torso_height,
            reference.feet_height,
        ],
        dim=-1,
    )
    return _flatten_frames("actor future reference", frames, ACTOR_REFERENCE_DIM)


# Critic observations: deployable actor prefix + current privileged state + errors.
def feet_contact(env: ManagerBasedRlEnv, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    sensor = cast(ContactSensor, env.scene[sensor_name])
    assert sensor.data.found is not None
    return (sensor.data.found > 0).to(dtype=torch.float32)


def critic_privileged_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    """398-D current simulation-only physical state."""
    command = _command(env)
    robot = _robot(env)
    body = _current_root_local_bodies(env)
    state = torch.cat(
        [
            robot.data.root_link_lin_vel_b * VELOCITY_SCALE,
            command.robot_body_pos_w[:, _torso_index(command), 2:3],
            _feet_height(env),
            feet_contact(env),
            body.position.flatten(start_dim=1),
            _rotation_6d(body.quaternion).flatten(start_dim=1),
            body.linear_velocity.flatten(start_dim=1) * VELOCITY_SCALE,
            body.angular_velocity.flatten(start_dim=1) * VELOCITY_SCALE,
        ],
        dim=-1,
    )
    return _check_frame("critic privileged state", state, CRITIC_PRIVILEGED_STATE_DIM)


def critic_tracking_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    """405-D current error in the same root-local geometry as the rewards."""
    command = _command(env)
    robot = _robot(env)
    current = _current_root_local_bodies(env)
    reference = _reference_root_local_bodies(command)
    reference_root_lin_vel_b = quat_apply_inverse(command.body_quat_w[:, 0], command.root_lin_vel_w)
    torso_index = _torso_index(command)
    error = torch.cat(
        [
            (reference.position - current.position).flatten(start_dim=1),
            _relative_rotation_6d(current.quaternion, reference.quaternion).flatten(start_dim=1),
            (reference.linear_velocity - current.linear_velocity).flatten(start_dim=1)
            * VELOCITY_SCALE,
            (reference.angular_velocity - current.angular_velocity).flatten(start_dim=1)
            * VELOCITY_SCALE,
            (reference_root_lin_vel_b - robot.data.root_link_lin_vel_b) * VELOCITY_SCALE,
            (command.root_ang_vel_b - robot.data.root_link_ang_vel_b) * VELOCITY_SCALE,
            _relative_rotation_6d(robot.data.root_link_quat_w, command.body_quat_w[:, 0]),
            command.body_pos_w[:, torso_index, 2:3] - command.robot_body_pos_w[:, torso_index, 2:3],
            command.feet_height_w - _feet_height(env),
        ],
        dim=-1,
    )
    return _check_frame("critic tracking error", error, CRITIC_TRACKING_ERROR_DIM)


# Adapter-only history and world-model state.
def adapter_history_frame(env: ManagerBasedRlEnv, noisy: bool = True) -> torch.Tensor:
    """One 81-D history frame: gyro, gravity, q, qd, previous target."""
    state = _proprioception(env, noisy)
    return _check_frame(
        "adapter history frame",
        torch.cat(
            [
                state.gyro * VELOCITY_SCALE,
                state.gravity,
                state.joint_pos,
                state.joint_vel * VELOCITY_SCALE,
                _centered_action_target(env),
            ],
            dim=-1,
        ),
        FRAME_DIM,
    )


def world_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    """57-D current state used as the world-model input and target."""
    robot = _robot(env)
    return _check_frame(
        "world state",
        torch.cat(
            [
                robot.data.root_link_ang_vel_b * VELOCITY_SCALE,
                robot.data.projected_gravity_b,
                robot.data.joint_pos - robot.data.default_joint_pos,
                robot.data.joint_vel * VELOCITY_SCALE,
                robot.data.root_link_pos_w[:, 2:3],
            ],
            dim=-1,
        ),
        WORLD_STATE_DIM,
    )


def reference_world_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    """57-D reference state used by the adapter world-model loss."""
    command = _command(env)
    return _check_frame(
        "reference world state",
        torch.cat(
            [
                command.root_ang_vel_b * VELOCITY_SCALE,
                command.root_projected_gravity_b,
                command.joint_pos - command.robot.data.default_joint_pos,
                command.joint_vel * VELOCITY_SCALE,
                command.body_pos_w[:, 0, 2:3],
            ],
            dim=-1,
        ),
        WORLD_STATE_DIM,
    )


assert ACTOR_OBS_DIM == 686
assert CRITIC_OBS_DIM == 1489
