"""Convert a mjlab tracking NPZ to a slim ref_motion.npz for mini_deploy_any2track_task.

Usage:
    gr3mini-export-ref-motion <tracking_npz> <output_dir>

Input: mjlab tracking format (from gr3mini-convert-motion)
Output: ref_motion.npz with fields for raw_critic_686 observation mode

Fields written
--------------
  joint_pos     (T, 25) - joint positions
  joint_vel     (T, 25) - joint velocities
  root_height   (T, 1)  - base_link Z (world frame)
  feet_height   (T, 2)  - left/right foot site Z (world frame)
  root_linvel   (T, 3)  - root linear velocity in reference body frame
  root_angvel   (T, 3)  - root angular velocity (body frame, = root_ang_vel_b)
  root_gvec     (T, 3)  - projected gravity in body frame
  root_quat     (T, 4)  - root quaternion wxyz (for rotation_6d in C++)
  torso_height  (T, 1)  - torso_link Z (world frame)
  fps           (1,)    - playback frequency

The first six fields are backward-compatible with raw_critic_671.
The last three (root_quat, torso_height) are required by raw_critic_686.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TORSO_BODY_INDEX = 2   # index of 'torso_link' in EXPECTED_BODY_NAMES


def _rotate_world_to_body(quat_wxyz: np.ndarray, vec_w: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors to body frame using wxyz quaternions (T,4) × (T,3)."""
    w = quat_wxyz[:, 0:1]
    x = quat_wxyz[:, 1:2]
    y = quat_wxyz[:, 2:3]
    z = quat_wxyz[:, 3:4]
    # R^T @ v  =  q^{-1} * v * q  (conjugate rotation)
    # Matrix form of conjugate (inverse for unit quats): negate x,y,z
    # R(q^{-1})_ij = R(q)^T_ij
    # Compute R^T @ vec inline
    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y - w*z)
    r20 = 2*(x*z + w*y)
    r01 = 2*(x*y + w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z - w*x)
    r02 = 2*(x*z - w*y)
    r12 = 2*(y*z + w*x)
    r22 = 1 - 2*(x*x + y*y)
    vx = vec_w[:, 0:1]
    vy = vec_w[:, 1:2]
    vz = vec_w[:, 2:3]
    # R @ v = v in world frame → R^T @ v = v in body frame
    body_x = r00*vx + r10*vy + r20*vz
    body_y = r01*vx + r11*vy + r21*vz
    body_z = r02*vx + r12*vy + r22*vz
    return np.concatenate([body_x, body_y, body_z], axis=-1).astype(np.float32)


def convert(tracking_npz: Path, output_dir: Path, extend_s: float = 0.0) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(tracking_npz, allow_pickle=False) as data:
        joint_pos = data["joint_pos"].astype(np.float32)       # (T, 25)
        joint_vel = data["joint_vel"].astype(np.float32)       # (T, 25)
        body_pos_w = data["body_pos_w"].astype(np.float32)     # (T, B, 3)
        body_quat_w = data["body_quat_w"].astype(np.float32)   # (T, B, 4) wxyz
        root_lin_vel_w = data["root_lin_vel_w"].astype(np.float32)  # (T, 3) world frame
        root_ang_vel_b = data["root_ang_vel_b"].astype(np.float32)  # (T, 3) body frame
        root_projected_gravity_b = data["root_projected_gravity_b"].astype(np.float32)  # (T, 3)
        feet_height_w = data["feet_height_w"].astype(np.float32)    # (T, 2)
        frequency = float(data["frequency"])

    T = joint_pos.shape[0]

    # root_height: base_link (body index 0) Z in world
    root_height = body_pos_w[:, 0, 2:3]  # (T, 1)

    # torso_height: torso_link (body index TORSO_BODY_INDEX) Z in world
    torso_height = body_pos_w[:, TORSO_BODY_INDEX, 2:3]  # (T, 1)

    # root_quat: wxyz format (MuJoCo convention, body index 0)
    root_quat = body_quat_w[:, 0, :]  # (T, 4) wxyz

    # root_linvel: reference root-linear-velocity in the REFERENCE body frame
    # training: quat_apply_inverse(ref_root_quat, ref_root_lin_vel_w)
    root_linvel = _rotate_world_to_body(root_quat, root_lin_vel_w)  # (T, 3)

    # root_angvel: body-frame angular velocity (already body frame from mjlab FK)
    root_angvel = root_ang_vel_b  # (T, 3)

    # root_gvec: projected gravity in body frame
    root_gvec = root_projected_gravity_b  # (T, 3)

    # feet_height: absolute Z of foot sites
    feet_height = feet_height_w  # (T, 2)

    # Extend the final frame by extend_s seconds so the history buffer has time
    # to flush backflip dynamics before the policy enters hold mode.
    if extend_s > 0.0:
        n_extend = max(1, round(extend_s * frequency))
        zeros2d = lambda shape: np.zeros(shape, dtype=np.float32)
        joint_pos   = np.vstack([joint_pos,   np.tile(joint_pos[-1:],   (n_extend, 1))])
        joint_vel   = np.vstack([joint_vel,   zeros2d((n_extend, joint_vel.shape[1]))])
        root_height = np.vstack([root_height, np.tile(root_height[-1:], (n_extend, 1))])
        torso_height= np.vstack([torso_height,np.tile(torso_height[-1:],(n_extend, 1))])
        feet_height = np.vstack([feet_height, np.tile(feet_height[-1:], (n_extend, 1))])
        root_quat   = np.vstack([root_quat,   np.tile(root_quat[-1:],   (n_extend, 1))])
        root_gvec   = np.vstack([root_gvec,   np.tile(root_gvec[-1:],   (n_extend, 1))])
        root_linvel = np.vstack([root_linvel,  zeros2d((n_extend, 3))])
        root_angvel = np.vstack([root_angvel,  zeros2d((n_extend, 3))])
        T = joint_pos.shape[0]

    output_path = output_dir / "ref_motion.npz"
    np.savez(
        output_path,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_height=root_height,
        feet_height=feet_height,
        root_linvel=root_linvel,
        root_angvel=root_angvel,
        root_gvec=root_gvec,
        root_quat=root_quat,
        torso_height=torso_height,
        fps=np.array([frequency], dtype=np.float32),
    )

    print(f"exported: {output_path}")
    print(f"  frames={T} fps={frequency} joint_pos=({T},25) root_quat=({T},4) torso_height=({T},1)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracking_npz", type=Path, help="mjlab tracking NPZ (from gr3mini-convert-motion)")
    parser.add_argument("output_dir", type=Path, help="directory to write ref_motion.npz")
    parser.add_argument(
        "--extend-s",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="repeat the last frame for this many seconds at the end of the trajectory "
             "(helps the history buffer flush dynamic data before policy hold mode)",
    )
    args = parser.parse_args()
    convert(
        args.tracking_npz.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        extend_s=args.extend_s,
    )


if __name__ == "__main__":
    main()
