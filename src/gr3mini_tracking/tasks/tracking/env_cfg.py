"""Environment configuration for the two-stage DiffCritic pipeline."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.terminations import time_out
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from gr3mini_tracking.robots.gr3mini211 import (
    COLLISION_GEOM_NAMES,
    TRACKED_BODY_NAMES,
    UNDESIRED_GROUND_CONTACT_GEOM_NAMES,
    UNDESIRED_SELF_COLLISION_PAIRS,
    get_gr3mini211_robot_cfg,
)

from . import observations as obs
from . import rewards as rew
from . import terminations as term
from .actions import ReferenceResidualJointPositionActionCfg
from .commands import Gr3MotionCommandCfg
from .events import push_planar_velocity

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MOTION_FILE = (
    PROJECT_ROOT / "motions" / "Extended_3_stageii_from_g1_gr3mini_v211_isaaclab_mjlab.npz"
)


def _observation_cfg(adapter: bool, play: bool) -> dict[str, ObservationGroupCfg]:
    groups = {
        "actor": ObservationGroupCfg(
            terms={
                "state_history": ObservationTermCfg(
                    func=obs.actor_state_frame,
                    params={"noisy": not play},
                    history_length=obs.ACTOR_HISTORY_LENGTH,
                    flatten_history_dim=True,
                ),
                "action_history": ObservationTermCfg(
                    func=obs.actor_action_target,
                    history_length=obs.ACTOR_HISTORY_LENGTH,
                    flatten_history_dim=True,
                ),
                "future_reference": ObservationTermCfg(func=obs.actor_future_reference),
            },
            concatenate_terms=True,
            enable_corruption=not play,
            nan_policy="error",
        ),
        "critic": ObservationGroupCfg(
            terms={
                # The first 686 dimensions mirror the deployable actor layout.
                # Keep this copy uncorrupted so the critic can use it together
                # with exact simulation-only state for value estimation.
                "actor_state_history": ObservationTermCfg(
                    func=obs.actor_state_frame,
                    params={"noisy": False},
                    history_length=obs.ACTOR_HISTORY_LENGTH,
                    flatten_history_dim=True,
                ),
                "actor_action_history": ObservationTermCfg(
                    func=obs.actor_action_target,
                    history_length=obs.ACTOR_HISTORY_LENGTH,
                    flatten_history_dim=True,
                ),
                "actor_future_reference": ObservationTermCfg(func=obs.actor_future_reference),
                "privileged_state": ObservationTermCfg(func=obs.critic_privileged_state),
                "tracking_error": ObservationTermCfg(func=obs.critic_tracking_error),
            },
            concatenate_terms=True,
            enable_corruption=False,
            nan_policy="error",
        ),
    }
    if adapter:
        groups.update(
            {
                "history": ObservationGroupCfg(
                    terms={
                        "frames": ObservationTermCfg(
                            func=obs.adapter_history_frame,
                            params={"noisy": not play},
                            history_length=obs.ADAPTER_HISTORY_LENGTH,
                            flatten_history_dim=True,
                        )
                    },
                    concatenate_terms=True,
                    enable_corruption=not play,
                    nan_policy="error",
                ),
                "world": ObservationGroupCfg(
                    terms={"state": ObservationTermCfg(func=obs.world_state)},
                    concatenate_terms=True,
                    enable_corruption=False,
                    nan_policy="error",
                ),
                "reference_world": ObservationGroupCfg(
                    terms={"state": ObservationTermCfg(func=obs.reference_world_state)},
                    concatenate_terms=True,
                    enable_corruption=False,
                    nan_policy="error",
                ),
            }
        )
    return groups


def _events_cfg(play: bool) -> dict[str, EventTermCfg]:
    if play:
        return {}
    robot_all = SceneEntityCfg("robot", joint_names=(".*",))
    return {
        "push_robot": EventTermCfg(
            func=push_planar_velocity,
            mode="interval",
            interval_range_s=(5.0, 10.0),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "magnitude_range": (0.1, 1.0),
            },
        ),
        "foot_friction": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=("left_foot", "right_foot")),
                "ranges": (0.4, 1.5),
                "operation": "abs",
                "shared_random": True,
            },
        ),
        "joint_friction": EventTermCfg(
            func=dr.joint_friction,
            mode="startup",
            params={
                "asset_cfg": robot_all,
                "ranges": (0.75, 1.25),
                "operation": "scale",
            },
        ),
        "joint_armature": EventTermCfg(
            func=dr.joint_armature,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "ranges": (1.0, 1.05),
                "operation": "scale",
            },
        ),
        "body_mass": EventTermCfg(
            func=dr.body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
                "ranges": (0.75, 1.25),
                "operation": "scale",
            },
        ),
        "torso_mass": EventTermCfg(
            func=dr.body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "ranges": (-1.0, 1.0),
                "operation": "add",
            },
        ),
        "torso_com": EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "ranges": {0: (-0.15, 0.15), 1: (-0.15, 0.15), 2: (-0.15, 0.15)},
                "operation": "add",
            },
        ),
        "pd_gains": EventTermCfg(
            func=dr.pd_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]),
                "kp_range": (0.75, 1.25),
                "kd_range": (0.75, 1.25),
                "operation": "scale",
            },
        ),
    }


def _undesired_self_collision_sensor_cfgs() -> tuple[ContactSensorCfg, ...]:
    return tuple(
        ContactSensorCfg(
            name=f"undesired_self_collision_{index}",
            primary=ContactMatch(mode="geom", pattern=primary, entity="robot"),
            secondary=ContactMatch(mode="geom", pattern=secondary, entity="robot"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )
        for index, (primary, secondary) in enumerate(UNDESIRED_SELF_COLLISION_PAIRS)
    )


def _rewards_cfg() -> dict[str, RewardTermCfg]:
    return {
        "body_local_pos": RewardTermCfg(
            func=rew.body_local_position_tracking_exp, weight=2.0, params={"sigma": 0.30}
        ),
        "body_local_rot": RewardTermCfg(
            func=rew.body_local_orientation_tracking_exp, weight=1.5, params={"sigma": 0.40}
        ),
        "body_global_linvel": RewardTermCfg(
            func=rew.body_global_linear_velocity_tracking_exp,
            weight=1.0,
            params={"sigma": 1.00},
        ),
        "body_global_angvel": RewardTermCfg(
            func=rew.body_global_angular_velocity_tracking_exp,
            weight=1.0,
            params={"sigma": 3.14},
        ),
        "torso_world_angvel": RewardTermCfg(
            func=rew.torso_world_angular_velocity_tracking_exp,
            weight=3.0,
            params={"sigma": 6.0},
        ),
        "joint_pos_tracking": RewardTermCfg(
            func=rew.joint_position_tracking_exp, weight=0.5, params={"sigma": 10.0}
        ),
        "joint_vel_tracking": RewardTermCfg(
            func=rew.joint_velocity_tracking_exp, weight=0.5, params={"sigma": 1.0}
        ),
        "root_pos": RewardTermCfg(
            func=rew.root_position_tracking_exp, weight=0.5, params={"sigma": 0.30}
        ),
        "root_orientation": RewardTermCfg(
            func=rew.root_orientation_tracking_exp, weight=1.0, params={"sigma": 0.40}
        ),
        "torso_world_orientation": RewardTermCfg(
            func=rew.torso_world_orientation_tracking_exp, weight=3.0, params={"sigma": 0.40}
        ),
        "torso_height_tracking": RewardTermCfg(
            func=rew.torso_height_tracking_exp, weight=1.0, params={"sigma": 0.15}
        ),
        "feet_height_tracking": RewardTermCfg(
            func=rew.feet_height_tracking_exp, weight=1.0, params={"sigma": 0.15}
        ),
        "residual_action_rate": RewardTermCfg(func=rew.residual_action_rate_l2, weight=-0.1),
        "penalty_torque": RewardTermCfg(func=rew.joint_torque_l2, weight=-1.0e-5),
        "smoothness_joint": RewardTermCfg(func=rew.joint_smoothness, weight=-1.0e-6),
        "feet_slip": RewardTermCfg(func=rew.feet_slip, weight=-0.5),
        "dof_pos_limit": RewardTermCfg(func=rew.joint_position_soft_limit, weight=-5.0),
        "dof_vel_limit": RewardTermCfg(func=rew.joint_velocity_limit, weight=-10.0),
        "undesired_self_collision": RewardTermCfg(
            func=rew.undesired_self_collision_count, weight=-10.0
        ),
        "undesired_ground_contact": RewardTermCfg(
            func=rew.undesired_ground_contact_count, weight=-10.0
        ),
        "termination": RewardTermCfg(func=rew.terminated, weight=-100.0),
    }


def gr3mini_diff_critic_env_cfg(
    *, adapter: bool = False, play: bool = False
) -> ManagerBasedRlEnvCfg:
    """Create the standalone GR3Mini DiffCritic environment."""
    motion: CommandTermCfg = Gr3MotionCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        motion_file=str(DEFAULT_MOTION_FILE),
        anchor_body_name="torso_link",
        body_names=TRACKED_BODY_NAMES,
        pose_range={} if play else {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.27, 0.27)},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
        sampling_mode="start" if play else "adaptive",
        adaptive_bin_duration_s=0.25,
        adaptive_uniform_ratio=0.25,
        adaptive_tracking_error_weight=0.25,
        adaptive_alpha=0.01,
        debug_vis=play,
    )
    sensors = (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=r"^(left_foot|right_foot)$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
        *_undesired_self_collision_sensor_cfgs(),
        ContactSensorCfg(
            name="undesired_ground_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=UNDESIRED_GROUND_CONTACT_GEOM_NAMES,
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="torso_ground_contact",
            primary=ContactMatch(mode="geom", pattern="torso", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
            track_air_time=True,
        ),
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=1 if play else 4096,
            env_spacing=2.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_gr3mini211_robot_cfg()},
            sensors=sensors,
        ),
        observations=_observation_cfg(adapter, play),
        actions={
            "joint_pos": ReferenceResidualJointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                preserve_order=True,
                scale=1.0,
                use_default_offset=False,
                command_name="motion",
            )
        },
        commands={"motion": motion},
        events=_events_cfg(play),
        rewards=_rewards_cfg(),
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
            # A physical fall gate.  The 0.30 m reference-relative threshold
            # matches the original tracker and is intentionally the only
            # tracking-error termination.
            "root_height": TerminationTermCfg(func=term.bad_root_height),
            "torso_ground_contact_too_long": TerminationTermCfg(
                func=term.torso_ground_contact_too_long,
                params={"duration_s": 1.0},
            ),
            # Always reset invalid MuJoCo states rather than letting them enter
            # a rollout.  This is a numerical-safety check, not a tracking term.
            "invalid_state": TerminationTermCfg(func=term.invalid_state),
            # These are useful tracking diagnostics, but not fall conditions:
            # the target motion contains one-leg/hand-supported poses.  Making
            # either a termination causes short episodes before PPO can learn
            # the recovery.  Their corresponding rewards remain active.
            # "shoulder_height": TerminationTermCfg(func=term.bad_shoulder_height),
            # "body_position": TerminationTermCfg(func=term.bad_body_position),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=2.8,
            fovy=55.0,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=100,
            njmax=500,
            mujoco=MujocoCfg(
                timestep=0.002,
                integrator="euler",
                iterations=3,
                ls_iterations=5,
                disableflags=("eulerdamp",),
            ),
        ),
        decimation=10,
        episode_length_s=1.0e9 if play else 20.0,
        scale_rewards_by_dt=True,
    )
    return cfg


assert len(COLLISION_GEOM_NAMES) == 15
