"""Kinematically replay a GR3Mini tracking motion in MuJoCo's native viewer.

The script writes reference root and joint poses directly to the model.  It is
for checking motion conversion and visual quality only; it does not run a
policy or physics simulation.

Example:

    uv run gr3mini-replay-motion

Use ``--headless --max-frames 10`` for a non-interactive contract smoke test.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from gr3mini_tracking.cli.convert_motion import EXPECTED_BODY_NAMES
from gr3mini_tracking.robots.gr3mini211 import JOINT_NAMES, get_spec

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MOTION_FILE = (
    PROJECT_ROOT / "motions" / "Extended_3_stageii_from_g1_gr3mini_v211_isaaclab_mjlab.npz"
)


@dataclass(frozen=True)
class MotionReplayData:
    """Reference channels required for direct kinematic playback."""

    frequency: float
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_pos: np.ndarray

    @property
    def frame_count(self) -> int:
        return len(self.joint_pos)


def load_motion_file(path: Path) -> MotionReplayData:
    """Load and validate the mjlab tracking-motion contract."""
    if not path.is_file():
        raise FileNotFoundError(f"motion file does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "frequency",
            "joint_pos",
            "body_pos_w",
            "body_quat_w",
            "joint_names",
            "body_names",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path}: missing required fields {sorted(missing)}")

        frequency = float(np.asarray(data["frequency"]).reshape(-1)[0])
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
        joint_names = tuple(str(name) for name in data["joint_names"].tolist())
        body_names = tuple(str(name) for name in data["body_names"].tolist())

    frame_count = joint_pos.shape[0]
    if (
        not np.isfinite(frequency)
        or frequency <= 0.0
        or frame_count < 2
        or joint_pos.shape != (frame_count, len(JOINT_NAMES))
        or body_pos.shape != (frame_count, len(EXPECTED_BODY_NAMES), 3)
        or body_quat.shape != (frame_count, len(EXPECTED_BODY_NAMES), 4)
    ):
        raise ValueError(f"{path}: invalid mjlab motion shapes or frequency")
    if joint_names != JOINT_NAMES or body_names != EXPECTED_BODY_NAMES:
        raise ValueError(f"{path}: joint/body ordering does not match GR3Mini211")
    arrays = (joint_pos, body_pos, body_quat)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{path}: motion contains non-finite values")
    quaternion_norm = np.linalg.norm(body_quat[:, 0], axis=-1)
    if not np.allclose(quaternion_norm, 1.0, atol=1.0e-4):
        raise ValueError(f"{path}: root quaternions are not normalized")
    return MotionReplayData(
        frequency=frequency,
        root_pos=body_pos[:, 0],
        root_quat=body_quat[:, 0],
        joint_pos=joint_pos,
    )


def write_frame(model, data, motion: MotionReplayData, frame: int) -> None:
    """Set the MuJoCo state to one reference pose and recompute kinematics."""
    data.qpos[:3] = motion.root_pos[frame]
    data.qpos[3:7] = motion.root_quat[frame]
    data.qpos[7:] = motion.joint_pos[frame]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]


def _build_model() -> mujoco.MjModel:  # pyright: ignore[reportAttributeAccessIssue]
    """Compile the robot spec with a visible floor and directional light."""
    spec = get_spec()

    tex = spec.add_texture()
    tex.name = "floor_checker"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D  # pyright: ignore[reportAttributeAccessIssue]
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER  # pyright: ignore[reportAttributeAccessIssue]
    tex.rgb1 = [0.2, 0.3, 0.4]
    tex.rgb2 = [0.3, 0.4, 0.5]
    tex.width = 300
    tex.height = 300

    mat = spec.add_material()
    mat.name = "floor_mat"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_checker"  # pyright: ignore[reportAttributeAccessIssue]
    mat.texuniform = True
    mat.texrepeat = [4.0, 4.0]
    mat.reflectance = 0.2

    geom = spec.worldbody.add_geom()
    geom.name = "floor"
    geom.type = mujoco.mjtGeom.mjGEOM_PLANE  # pyright: ignore[reportAttributeAccessIssue]
    geom.size = [0, 0, 0.01]
    geom.material = "floor_mat"

    light = spec.worldbody.add_light()
    light.name = "sun"
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL  # pyright: ignore[reportAttributeAccessIssue]
    light.pos = [0, 0, 4]
    light.dir = [0, 0, -1]
    light.diffuse = [0.8, 0.8, 0.8]
    light.specular = [0.2, 0.2, 0.2]
    light.ambient = [0.3, 0.3, 0.3]

    return spec.compile()


def run_headless(motion: MotionReplayData, start_frame: int, max_frames: int) -> None:
    """Replay a finite frame range without opening a viewer."""
    model = _build_model()
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    for offset in range(max_frames):
        write_frame(model, data, motion, (start_frame + offset) % motion.frame_count)


def run_viewer(motion: MotionReplayData, start_frame: int, max_frames: int, speed: float) -> None:
    """Replay frames in a native MuJoCo viewer until closed or maxed out."""
    import mujoco.viewer

    model = _build_model()
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    frame = start_frame
    paused = False

    def on_key(key: int) -> None:
        nonlocal frame, paused
        if key == ord(" "):
            paused = not paused
        elif key in {ord("r"), ord("R")}:
            frame = start_frame

    interval = 1.0 / (motion.frequency * speed)
    rendered = 0
    deadline = time.perf_counter()
    with mujoco.viewer.launch_passive(  # pyright: ignore[reportAttributeAccessIssue]
        model, data, key_callback=on_key, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.cam.lookat[:] = motion.root_pos[frame]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        print("[INFO] Controls: Space pause/resume, R restart, close the window to stop.")
        while viewer.is_running():
            write_frame(model, data, motion, frame)
            viewer.sync()
            if not paused:
                frame = (frame + 1) % motion.frame_count
                rendered += 1
                if max_frames and rendered >= max_frames:
                    break
            deadline += interval
            time.sleep(max(0.0, deadline - time.perf_counter()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION_FILE)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames; 0 runs until closed.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument(
        "--headless", action="store_true", help="Validate and replay without opening a window."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_frame < 0 or args.max_frames < 0 or args.speed <= 0.0:
        raise ValueError(
            "--start-frame and --max-frames must be non-negative; --speed must be positive"
        )
    if args.headless and args.max_frames == 0:
        raise ValueError("--headless requires a positive --max-frames")

    path = args.motion_file.expanduser().resolve()
    motion = load_motion_file(path)
    start_frame = args.start_frame % motion.frame_count
    print(
        f"[INFO] Motion contract validated: {path} "
        f"({motion.frame_count} frames at {motion.frequency:g} Hz)."
    )
    if args.headless:
        run_headless(motion, start_frame, args.max_frames)
        print(f"[INFO] Replayed {args.max_frames} frames headlessly.")
    else:
        run_viewer(motion, start_frame, args.max_frames, args.speed)


if __name__ == "__main__":
    main()
