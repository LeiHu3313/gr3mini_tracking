"""Convert the Any2Track trajectory archive into mjlab's tracking schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gr3mini_tracking.robots.gr3mini211 import JOINT_NAMES

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


def convert_motion_file(source: Path, destination: Path) -> None:
    data = np.load(source, allow_pickle=True)
    joint_names = tuple(str(name) for name in data["joint_names"].tolist())
    if joint_names != ("base_free_joint", *JOINT_NAMES):
        raise ValueError("source joint order does not match GR3Mini211 v2.1.1")
    body_names = tuple(str(name) for name in data["body_names"].tolist())
    if body_names != ("world", *EXPECTED_BODY_NAMES):
        raise ValueError("source body order does not match GR3Mini211 v2.1.1")
    site_names = tuple(str(name) for name in data["site_names"].tolist())
    required_sites = ("imu_in_base", "left_foot", "right_foot")
    missing_sites = set(required_sites).difference(site_names)
    if missing_sites:
        raise ValueError(f"source motion is missing sites: {sorted(missing_sites)}")
    frequency = float(data["frequency"])
    if not np.isclose(frequency, 50.0):
        raise ValueError(f"expected 50 Hz motion, got {frequency}")

    imu_index = site_names.index("imu_in_base")
    foot_indexes = [site_names.index("left_foot"), site_names.index("right_foot")]
    imu_rotation = data["site_xmat"][:, imu_index].reshape(-1, 3, 3)
    gravity = np.einsum("tji,j->ti", imu_rotation, np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    metadata = {
        "source": str(source.resolve()),
        "source_schema": "opentrack_gr3mini_v211",
        "frequency": frequency,
        "quaternion_order": "wxyz",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        joint_pos=np.asarray(data["qpos"][:, 7:], dtype=np.float32),
        joint_vel=np.asarray(data["qvel"][:, 6:], dtype=np.float32),
        body_pos_w=np.asarray(data["xpos"][:, 1:], dtype=np.float32),
        body_quat_w=np.asarray(data["xquat"][:, 1:], dtype=np.float32),
        body_lin_vel_w=np.asarray(data["cvel"][:, 1:, 3:], dtype=np.float32),
        body_ang_vel_w=np.asarray(data["cvel"][:, 1:, :3], dtype=np.float32),
        root_lin_vel_w=np.asarray(data["qvel"][:, :3], dtype=np.float32),
        root_ang_vel_b=np.asarray(data["qvel"][:, 3:6], dtype=np.float32),
        root_projected_gravity_b=np.asarray(gravity, dtype=np.float32),
        feet_height_w=np.asarray(data["site_xpos"][:, foot_indexes, 2], dtype=np.float32),
        joint_names=np.asarray(JOINT_NAMES),
        body_names=np.asarray(EXPECTED_BODY_NAMES),
        frequency=np.asarray(frequency),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert_motion_file(args.source, args.destination)
    print(f"converted {args.source} -> {args.destination}")
