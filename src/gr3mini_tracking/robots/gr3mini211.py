"""Fourier GR3Mini v2.1.1 robot configuration for mjlab."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GR3MINI211_XML = PROJECT_ROOT / "assets" / "gr3mini211" / "gr3mini_v211.xml"

JOINT_NAMES = (
    "waist_yaw_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)

DEFAULT_JOINT_POS = (
    0.0,
    0.0,
    0.0,
    0.174533,
    0.349066,
    0.0,
    -0.349066,
    0.0,
    0.174533,
    -0.349066,
    0.0,
    -0.349066,
    0.0,
    -0.20,
    0.0,
    0.0,
    0.30,
    -0.10,
    0.0,
    -0.20,
    0.0,
    0.0,
    0.30,
    -0.10,
    0.0,
)

STIFFNESS = (
    100.0,
    15.0,
    15.0,
    50.0,
    30.0,
    30.0,
    30.0,
    15.0,
    50.0,
    30.0,
    30.0,
    30.0,
    15.0,
    130.0,
    100.0,
    50.0,
    130.0,
    53.0,
    26.0,
    130.0,
    100.0,
    50.0,
    130.0,
    53.0,
    26.0,
)

DAMPING = (
    5.0,
    2.0,
    2.0,
    4.0,
    1.0,
    1.0,
    1.0,
    2.0,
    4.0,
    1.0,
    1.0,
    1.0,
    2.0,
    6.0,
    5.0,
    4.0,
    5.0,
    2.65,
    1.3,
    6.0,
    5.0,
    4.0,
    5.0,
    2.65,
    1.3,
)

EFFORT_LIMITS = (
    120.0,
    14.0,
    14.0,
    40.0,
    28.5,
    28.5,
    28.5,
    14.0,
    40.0,
    28.5,
    28.5,
    28.5,
    14.0,
    120.0,
    40.0,
    40.0,
    120.0,
    52.0,
    25.0,
    120.0,
    40.0,
    40.0,
    120.0,
    52.0,
    25.0,
)

DOF_VEL_LIMITS = (
    13.3888,
    5.2330,
    5.2330,
    14.6608,
    18.3529,
    18.3529,
    18.3529,
    5.2330,
    14.6608,
    18.3529,
    18.3529,
    18.3529,
    5.2330,
    13.3888,
    14.6608,
    14.6608,
    13.3888,
    18.3529,
    18.3529,
    13.3888,
    14.6608,
    14.6608,
    13.3888,
    18.3529,
    18.3529,
)

LOWER_BODY_NAMES = (
    "base_link",
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
    "waist_yaw_link",
    "torso_link",
)

UPPER_BODY_NAMES = (
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
)

TRACKED_BODY_NAMES = LOWER_BODY_NAMES + UPPER_BODY_NAMES
TORSO_BODY_NAME = "torso_link"
FEET_BODY_NAMES = ("left_foot_roll_link", "right_foot_roll_link")
SHOULDER_BODY_NAMES = (
    "left_upper_arm_pitch_link",
    "right_upper_arm_pitch_link",
)
FOOT_SITE_NAMES = ("left_foot", "right_foot")

COLLISION_GEOM_NAMES = (
    "head_collision",
    "left_hand_collision",
    "left_lower_arm_collision",
    "left_upper_arm_collision",
    "right_hand_collision",
    "right_lower_arm_collision",
    "right_upper_arm_collision",
    "torso",
    "left_foot",
    "left_shin",
    "left_thigh",
    "right_foot",
    "right_shin",
    "right_thigh",
    "pelvis_collision",
)


def get_spec() -> mujoco.MjSpec:  # pyright: ignore[reportAttributeAccessIssue]
    """Load the upstream robot and replace its motor actuators with mjlab PD actuators."""
    spec = mujoco.MjSpec.from_file(  # pyright: ignore[reportAttributeAccessIssue]
        str(GR3MINI211_XML)
    )
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    # These pairs refer to the upstream scene's `floor` geom. The standalone mjlab
    # scene supplies ground contact through contype/conaffinity instead.
    for pair in list(spec.pairs):
        if pair.name.endswith("_floor"):
            spec.delete(pair)
    # Global simulation options are configured by ``SimulationCfg`` after the
    # robot is attached. Restore MjSpec defaults here to avoid attach conflicts
    # while keeping the source values in the environment configuration.
    spec.option.iterations = 100
    spec.option.ls_iterations = 50
    spec.option.disableflags = 0
    return spec


def _actuators() -> tuple[IdealPdActuatorCfg, ...]:
    # One group per joint preserves the non-uniform upstream gain/limit arrays.
    return tuple(
        IdealPdActuatorCfg(
            target_names_expr=(name,),
            stiffness=kp,
            damping=kd,
            effort_limit=effort,
        )
        for name, kp, kd, effort in zip(JOINT_NAMES, STIFFNESS, DAMPING, EFFORT_LIMITS, strict=True)
    )


def get_gr3mini211_robot_cfg() -> EntityCfg:
    """Return a fresh GR3Mini211 entity configuration."""
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.65),
            joint_pos=dict(zip(JOINT_NAMES, DEFAULT_JOINT_POS, strict=True)),
            joint_vel={".*": 0.0},
        ),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=_actuators(), soft_joint_pos_limit_factor=0.95
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=COLLISION_GEOM_NAMES,
                contype=1,
                conaffinity=0,
                condim={r"^(left|right)_foot$": 3, ".*": 1},
                priority={r"^(left|right)_foot$": 1, ".*": 0},
                friction={r"^(left|right)_foot$": (0.6,)},
            ),
        ),
        sort_actuators=True,
    )


assert len(JOINT_NAMES) == len(DEFAULT_JOINT_POS) == 25
assert len(TRACKED_BODY_NAMES) == 26
