"""Convert GR3Mini motion archives into mjlab's tracking schema."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np

from gr3mini_tracking.robots.gr3mini211 import (
    FOOT_SITE_NAMES,
    JOINT_NAMES,
    get_spec,
)

EXPECTED_BODY_NAMES = (
    "base_link",
    "waist_yaw_link",
    "torso_link",
    "head_yaw_link",
    "head_pitch_link",
    "left_upper_arm_pitch_link",
    "left_upper_arm_roll_link",
    "left_upper_arm_yaw_link",
    "left_lower_arm_pitch_link",
    "left_hand_yaw_link",
    "right_upper_arm_pitch_link",
    "right_upper_arm_roll_link",
    "right_upper_arm_yaw_link",
    "right_lower_arm_pitch_link",
    "right_hand_yaw_link",
    "left_thigh_pitch_link",
    "left_thigh_roll_link",
    "left_thigh_yaw_link",
    "left_shank_pitch_link",
    "left_foot_pitch_link",
    "left_foot_roll_link",
    "right_thigh_pitch_link",
    "right_thigh_roll_link",
    "right_thigh_yaw_link",
    "right_shank_pitch_link",
    "right_foot_pitch_link",
    "right_foot_roll_link",
    "imu_link",
)


def _metadata(source: Path, schema: str, frequency: float) -> np.ndarray:
    return np.asarray(
        json.dumps(
            {
                "source": str(source.resolve()),
                "source_schema": schema,
                "frequency": frequency,
                "quaternion_order": "wxyz",
            },
            ensure_ascii=False,
        )
    )


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    return np.asarray(np.gradient(values, dt, axis=0), dtype=np.float32)


def _quaternion_angular_velocity_w(quaternion_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Return world-frame angular velocity from consecutive WXYZ quaternions."""
    previous = quaternion_wxyz[:-1]
    following = quaternion_wxyz[1:]
    delta = np.empty_like(previous)
    delta[..., 0] = following[..., 0] * previous[..., 0] + np.sum(
        following[..., 1:] * previous[..., 1:], axis=-1
    )
    delta[..., 1:] = (
        following[..., 1:] * previous[..., :1]
        - following[..., :1] * previous[..., 1:]
        - np.cross(following[..., 1:], previous[..., 1:])
    )
    delta *= np.where(delta[..., :1] < 0.0, -1.0, 1.0)
    vector_norm = np.linalg.norm(delta[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, delta[..., :1])
    velocity = delta[..., 1:] * (angle / np.maximum(vector_norm, 1.0e-8)) / dt
    result = np.empty_like(quaternion_wxyz[..., :3])
    result[0] = velocity[0]
    result[-1] = velocity[-1]
    if len(result) > 2:
        result[1:-1] = 0.5 * (velocity[:-1] + velocity[1:])
    return result.astype(np.float32)


def _rotate_world_to_body(quaternion_wxyz: np.ndarray, vector_w: np.ndarray) -> np.ndarray:
    quaternion_vector = quaternion_wxyz[..., 1:]
    twice_cross = 2.0 * np.cross(quaternion_vector, vector_w)
    return (
        vector_w - quaternion_wxyz[..., :1] * twice_cross + np.cross(quaternion_vector, twice_cross)
    )


def _convert_any2track(data: np.lib.npyio.NpzFile, source: Path) -> dict[str, np.ndarray]:
    joint_names = tuple(str(name) for name in data["joint_names"].tolist())
    if joint_names != ("base_free_joint", *JOINT_NAMES):
        raise ValueError("source joint order does not match GR3Mini211 v2.1.1")
    body_names = tuple(str(name) for name in data["body_names"].tolist())
    if body_names != ("world", *EXPECTED_BODY_NAMES):
        raise ValueError("source body order does not match GR3Mini211 v2.1.1")
    site_names = tuple(str(name) for name in data["site_names"].tolist())
    missing_sites = set(("imu_in_base", *FOOT_SITE_NAMES)).difference(site_names)
    if missing_sites:
        raise ValueError(f"source motion is missing sites: {sorted(missing_sites)}")
    frequency = float(data["frequency"])
    if not np.isclose(frequency, 50.0):
        raise ValueError(f"expected 50 Hz motion, got {frequency}")

    imu_index = site_names.index("imu_in_base")
    foot_indexes = [site_names.index(name) for name in FOOT_SITE_NAMES]
    imu_rotation = data["site_xmat"][:, imu_index].reshape(-1, 3, 3)
    gravity = np.einsum("tji,j->ti", imu_rotation, np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    return {
        "joint_pos": np.asarray(data["qpos"][:, 7:], dtype=np.float32),
        "joint_vel": np.asarray(data["qvel"][:, 6:], dtype=np.float32),
        "body_pos_w": np.asarray(data["xpos"][:, 1:], dtype=np.float32),
        "body_quat_w": np.asarray(data["xquat"][:, 1:], dtype=np.float32),
        "body_lin_vel_w": np.asarray(data["cvel"][:, 1:, 3:], dtype=np.float32),
        "body_ang_vel_w": np.asarray(data["cvel"][:, 1:, :3], dtype=np.float32),
        "root_lin_vel_w": np.asarray(data["qvel"][:, :3], dtype=np.float32),
        "root_ang_vel_b": np.asarray(data["qvel"][:, 3:6], dtype=np.float32),
        "root_projected_gravity_b": np.asarray(gravity, dtype=np.float32),
        "feet_height_w": np.asarray(data["site_xpos"][:, foot_indexes, 2], dtype=np.float32),
        "joint_names": np.asarray(JOINT_NAMES),
        "body_names": np.asarray(EXPECTED_BODY_NAMES),
        "frequency": np.asarray(frequency),
        "metadata_json": _metadata(source, "any2track_gr3mini_v211", frequency),
    }


def _convert_isaaclab_gr3mini(data: np.lib.npyio.NpzFile, source: Path) -> dict[str, np.ndarray]:
    """Translate the fixed-head Isaac Lab reference onto mjlab body frames."""
    required = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "joint_names",
        "body_names",
        "fixed_joint_names",
    }
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Isaac Lab source is missing fields: {sorted(missing)}")

    frequency = float(np.asarray(data["fps"]).reshape(-1)[0])
    if not np.isclose(frequency, 50.0):
        raise ValueError(f"expected 50 Hz motion, got {frequency}")
    source_joint_names = tuple(str(name) for name in data["joint_names"].tolist())
    fixed_joint_names = tuple(str(name) for name in data["fixed_joint_names"].tolist())
    if set(source_joint_names).union(fixed_joint_names) != set(JOINT_NAMES):
        raise ValueError("Isaac Lab joint names do not cover the GR3Mini211 joints")
    if len(source_joint_names) != len(set(source_joint_names)):
        raise ValueError("Isaac Lab joint_names contains duplicates")
    source_body_names = tuple(str(name) for name in data["body_names"].tolist())
    if not source_body_names or source_body_names[0] != "base_link":
        raise ValueError("Isaac Lab body_names must start with base_link")

    joint_pos_source = np.asarray(data["joint_pos"], dtype=np.float32)
    joint_vel_source = np.asarray(data["joint_vel"], dtype=np.float32)
    body_pos_source = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_quat_source = np.asarray(data["body_quat_w"], dtype=np.float32)
    frame_count = joint_pos_source.shape[0]
    if (
        frame_count < 2
        or joint_pos_source.shape != (frame_count, len(source_joint_names))
        or joint_vel_source.shape != joint_pos_source.shape
        or body_pos_source.shape != (frame_count, len(source_body_names), 3)
        or body_quat_source.shape != (frame_count, len(source_body_names), 4)
    ):
        raise ValueError("Isaac Lab source arrays have incompatible shapes")
    source_arrays = (joint_pos_source, joint_vel_source, body_pos_source, body_quat_source)
    if not all(np.isfinite(value).all() for value in source_arrays):
        raise ValueError("Isaac Lab source contains non-finite values")

    joint_pos = np.zeros((frame_count, len(JOINT_NAMES)), dtype=np.float32)
    joint_vel = np.zeros_like(joint_pos)
    for target_index, name in enumerate(JOINT_NAMES):
        if name in source_joint_names:
            source_index = source_joint_names.index(name)
            joint_pos[:, target_index] = joint_pos_source[:, source_index]
            joint_vel[:, target_index] = joint_vel_source[:, source_index]

    model = get_spec().compile()
    runtime = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)  # pyright: ignore[reportAttributeAccessIssue]
            for name in EXPECTED_BODY_NAMES
        ]
    )
    foot_site_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)  # pyright: ignore[reportAttributeAccessIssue]
            for name in FOOT_SITE_NAMES
        ]
    )
    if np.any(body_ids < 0) or np.any(foot_site_ids < 0):
        raise ValueError("mjlab GR3Mini model is missing required tracking bodies or foot sites")

    root_pos = body_pos_source[:, 0]
    root_quat = body_quat_source[:, 0]
    body_pos = np.empty((frame_count, len(EXPECTED_BODY_NAMES), 3), dtype=np.float32)
    body_quat = np.empty((frame_count, len(EXPECTED_BODY_NAMES), 4), dtype=np.float32)
    feet_height = np.empty((frame_count, len(FOOT_SITE_NAMES)), dtype=np.float32)
    for frame in range(frame_count):
        runtime.qpos[:3] = root_pos[frame]
        runtime.qpos[3:7] = root_quat[frame]
        runtime.qpos[7:] = joint_pos[frame]
        mujoco.mj_forward(model, runtime)  # pyright: ignore[reportAttributeAccessIssue]
        body_pos[frame] = runtime.xpos[body_ids]
        body_quat[frame] = runtime.xquat[body_ids]
        feet_height[frame] = runtime.site_xpos[foot_site_ids, 2]

    dt = 1.0 / frequency
    root_lin_vel = _finite_difference(root_pos, dt)
    root_ang_vel_w = _quaternion_angular_velocity_w(root_quat, dt)
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": _finite_difference(body_pos, dt),
        "body_ang_vel_w": _quaternion_angular_velocity_w(body_quat, dt),
        "root_lin_vel_w": root_lin_vel,
        "root_ang_vel_b": _rotate_world_to_body(root_quat, root_ang_vel_w).astype(np.float32),
        "root_projected_gravity_b": _rotate_world_to_body(
            root_quat,
            np.broadcast_to(np.asarray([0.0, 0.0, -1.0], dtype=np.float32), root_pos.shape),
        ).astype(np.float32),
        "feet_height_w": feet_height,
        "joint_names": np.asarray(JOINT_NAMES),
        "body_names": np.asarray(EXPECTED_BODY_NAMES),
        "frequency": np.asarray(frequency),
        "metadata_json": _metadata(source, "isaaclab_gr3mini_v211_23dof", frequency),
    }


def _convert_csv_gr3mini(source: Path) -> dict[str, np.ndarray]:
    """Convert a retargeting-pipeline CSV (r2r / gr3mini_v211) to mjlab tracking format."""
    lines = source.read_text().splitlines()
    data_lines = [l for l in lines if not l.startswith("#") and l.strip()]
    if not data_lines:
        raise ValueError("CSV contains no data rows")

    # Parse header, validate columns
    header = [c.strip() for c in data_lines[0].split(",")]
    expected_cols = (
        ["time", "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"]
        + [f"dof_{name}" for name in JOINT_NAMES]
    )
    if header != expected_cols:
        raise ValueError(
            f"CSV columns do not match expected GR3Mini211 layout.\n"
            f"Got:      {header[:10]}...\n"
            f"Expected: {expected_cols[:10]}..."
        )

    rows = np.array(
        [[float(v) for v in l.split(",")] for l in data_lines[1:] if l.strip()],
        dtype=np.float32,
    )
    if rows.shape[1] != len(expected_cols):
        raise ValueError(f"CSV has {rows.shape[1]} columns, expected {len(expected_cols)}")

    root_pos = rows[:, 1:4]                       # x, y, z
    root_quat_xyzw = rows[:, 4:8]                 # qx, qy, qz, qw
    # Convert XYZW → WXYZ
    root_quat = np.concatenate([root_quat_xyzw[:, 3:4], root_quat_xyzw[:, :3]], axis=-1)
    joint_pos = rows[:, 8:]                        # 25 DOFs

    frequency = 50.0
    frame_count = rows.shape[0]
    if frame_count < 2:
        raise ValueError("CSV has fewer than 2 data rows")

    model = get_spec().compile()
    runtime = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)  # pyright: ignore[reportAttributeAccessIssue]
            for name in EXPECTED_BODY_NAMES
        ]
    )
    foot_site_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)  # pyright: ignore[reportAttributeAccessIssue]
            for name in FOOT_SITE_NAMES
        ]
    )
    if np.any(body_ids < 0) or np.any(foot_site_ids < 0):
        raise ValueError("mjlab GR3Mini model is missing required tracking bodies or foot sites")

    body_pos = np.empty((frame_count, len(EXPECTED_BODY_NAMES), 3), dtype=np.float32)
    body_quat = np.empty((frame_count, len(EXPECTED_BODY_NAMES), 4), dtype=np.float32)
    feet_height = np.empty((frame_count, len(FOOT_SITE_NAMES)), dtype=np.float32)
    for frame in range(frame_count):
        runtime.qpos[:3] = root_pos[frame]
        runtime.qpos[3:7] = root_quat[frame]
        runtime.qpos[7:] = joint_pos[frame]
        mujoco.mj_forward(model, runtime)  # pyright: ignore[reportAttributeAccessIssue]
        body_pos[frame] = runtime.xpos[body_ids]
        body_quat[frame] = runtime.xquat[body_ids]
        feet_height[frame] = runtime.site_xpos[foot_site_ids, 2]

    dt = 1.0 / frequency
    root_ang_vel_w = _quaternion_angular_velocity_w(root_quat, dt)
    return {
        "joint_pos": joint_pos,
        "joint_vel": _finite_difference(joint_pos, dt),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": _finite_difference(body_pos, dt),
        "body_ang_vel_w": _quaternion_angular_velocity_w(body_quat, dt),
        "root_lin_vel_w": _finite_difference(root_pos, dt),
        "root_ang_vel_b": _rotate_world_to_body(root_quat, root_ang_vel_w).astype(np.float32),
        "root_projected_gravity_b": _rotate_world_to_body(
            root_quat,
            np.broadcast_to(np.asarray([0.0, 0.0, -1.0], dtype=np.float32), root_pos.shape),
        ).astype(np.float32),
        "feet_height_w": feet_height,
        "joint_names": np.asarray(JOINT_NAMES),
        "body_names": np.asarray(EXPECTED_BODY_NAMES),
        "frequency": np.asarray(frequency),
        "metadata_json": _metadata(source, "csv_r2r_gr3mini_v211", frequency),
    }


def _convert_gmr_pkl(source: Path) -> dict[str, np.ndarray]:
    """Convert a gmr-pkl retargeted motion (root_pos/rot + dof_pos) to mjlab tracking format."""
    with source.open("rb") as f:
        raw = pickle.load(f)

    required = {"fps", "root_pos", "root_rot", "dof_pos", "dof_names", "root_quat_format"}
    missing = required.difference(raw.keys())
    if missing:
        raise ValueError(f"gmr-pkl source is missing fields: {sorted(missing)}")
    if raw["root_quat_format"] != "xyzw":
        raise ValueError(f"expected xyzw quaternion format, got {raw['root_quat_format']!r}")
    dof_names = tuple(raw["dof_names"])
    if dof_names != JOINT_NAMES:
        raise ValueError("gmr-pkl dof_names do not match GR3Mini211 v2.1.1 JOINT_NAMES")

    frequency = float(raw["fps"])
    if not np.isclose(frequency, 50.0):
        raise ValueError(f"expected 50 Hz motion, got {frequency}")

    root_pos = np.asarray(raw["root_pos"], dtype=np.float32)
    root_quat_xyzw = np.asarray(raw["root_rot"], dtype=np.float32)
    root_quat = np.concatenate([root_quat_xyzw[:, 3:4], root_quat_xyzw[:, :3]], axis=-1)
    joint_pos = np.asarray(raw["dof_pos"], dtype=np.float32)

    frame_count = root_pos.shape[0]
    if frame_count < 2:
        raise ValueError("gmr-pkl has fewer than 2 frames")

    model = get_spec().compile()
    runtime = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    body_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)  # pyright: ignore[reportAttributeAccessIssue]
            for name in EXPECTED_BODY_NAMES
        ]
    )
    foot_site_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)  # pyright: ignore[reportAttributeAccessIssue]
            for name in FOOT_SITE_NAMES
        ]
    )
    if np.any(body_ids < 0) or np.any(foot_site_ids < 0):
        raise ValueError("mjlab GR3Mini model is missing required tracking bodies or foot sites")

    body_pos = np.empty((frame_count, len(EXPECTED_BODY_NAMES), 3), dtype=np.float32)
    body_quat = np.empty((frame_count, len(EXPECTED_BODY_NAMES), 4), dtype=np.float32)
    feet_height = np.empty((frame_count, len(FOOT_SITE_NAMES)), dtype=np.float32)
    for frame in range(frame_count):
        runtime.qpos[:3] = root_pos[frame]
        runtime.qpos[3:7] = root_quat[frame]
        runtime.qpos[7:] = joint_pos[frame]
        mujoco.mj_forward(model, runtime)  # pyright: ignore[reportAttributeAccessIssue]
        body_pos[frame] = runtime.xpos[body_ids]
        body_quat[frame] = runtime.xquat[body_ids]
        feet_height[frame] = runtime.site_xpos[foot_site_ids, 2]

    dt = 1.0 / frequency
    root_ang_vel_w = _quaternion_angular_velocity_w(root_quat, dt)
    return {
        "joint_pos": joint_pos,
        "joint_vel": _finite_difference(joint_pos, dt),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": _finite_difference(body_pos, dt),
        "body_ang_vel_w": _quaternion_angular_velocity_w(body_quat, dt),
        "root_lin_vel_w": _finite_difference(root_pos, dt),
        "root_ang_vel_b": _rotate_world_to_body(root_quat, root_ang_vel_w).astype(np.float32),
        "root_projected_gravity_b": _rotate_world_to_body(
            root_quat,
            np.broadcast_to(np.asarray([0.0, 0.0, -1.0], dtype=np.float32), root_pos.shape),
        ).astype(np.float32),
        "feet_height_w": feet_height,
        "joint_names": np.asarray(JOINT_NAMES),
        "body_names": np.asarray(EXPECTED_BODY_NAMES),
        "frequency": np.asarray(frequency),
        "metadata_json": _metadata(source, "gmr_pkl_gr3mini_v211", frequency),
    }


def convert_motion_file(source: Path, destination: Path) -> None:
    if source.suffix.lower() == ".pkl":
        converted = _convert_gmr_pkl(source)
    elif source.suffix.lower() == ".csv":
        converted = _convert_csv_gr3mini(source)
    else:
        with np.load(source, allow_pickle=False) as data:
            converted = (
                _convert_any2track(data, source)
                if "qpos" in data.files
                else _convert_isaaclab_gr3mini(data, source)
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **converted)  # pyright: ignore[reportArgumentType]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert_motion_file(args.source, args.destination)
    print(f"converted {args.source} -> {args.destination}")
