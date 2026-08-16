from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch import nn

from gr3mini_tracking.adapter.distribution import TanhGaussianDistribution
from gr3mini_tracking.adapter.models import HistoryEncoder, WorldModel
from gr3mini_tracking.algorithms.tanh_ppo import TanhMLPModel, TanhPPO
from gr3mini_tracking.cli.convert_motion import EXPECTED_BODY_NAMES, convert_motion_file
from gr3mini_tracking.robots.gr3mini211 import JOINT_NAMES, TRACKED_BODY_NAMES
from gr3mini_tracking.tasks.tracking import rewards as rew
from gr3mini_tracking.tasks.tracking import terminations as term
from gr3mini_tracking.tasks.tracking.commands import (
    Gr3MotionCommandCfg,
    adaptive_sampling_probabilities,
    temporal_bin_count,
)
from gr3mini_tracking.tasks.tracking.env_cfg import _rewards_cfg, gr3mini_diff_critic_env_cfg
from gr3mini_tracking.tasks.tracking.observations import (
    ACTION_TARGET_DIM,
    ACTOR_ACTION_HISTORY_DIM,
    ACTOR_OBS_DIM,
    ACTOR_REFERENCE_DIM,
    ACTOR_STATE_DIM,
    ACTOR_STATE_HISTORY_DIM,
    ADAPTER_HISTORY_LENGTH,
    CRITIC_OBS_DIM,
    CRITIC_PRIVILEGED_STATE_DIM,
    CRITIC_TRACKING_ERROR_DIM,
    FRAME_DIM,
    WORLD_STATE_DIM,
    _rotation_6d,
)


def test_observation_contract_dimensions() -> None:
    assert ACTOR_STATE_DIM == 56
    assert ACTION_TARGET_DIM == 25
    assert ACTOR_REFERENCE_DIM == 40
    assert ACTOR_STATE_HISTORY_DIM == 6 * ACTOR_STATE_DIM
    assert ACTOR_ACTION_HISTORY_DIM == 6 * ACTION_TARGET_DIM
    assert ACTOR_OBS_DIM == 686
    assert CRITIC_PRIVILEGED_STATE_DIM == 398
    assert CRITIC_TRACKING_ERROR_DIM == 405
    assert CRITIC_OBS_DIM == ACTOR_OBS_DIM + CRITIC_PRIVILEGED_STATE_DIM + CRITIC_TRACKING_ERROR_DIM
    assert CRITIC_OBS_DIM == 1489
    assert ADAPTER_HISTORY_LENGTH * FRAME_DIM == 6399
    assert WORLD_STATE_DIM == 57
    assert len(JOINT_NAMES) == 25
    assert len(TRACKED_BODY_NAMES) == 26


def test_general_tracking_reward_configuration() -> None:
    rewards = _rewards_cfg()
    expected_weights = {
        "body_local_pos": 2.0,
        "feet_pos_tracking": 2.1,
        "body_local_rot": 0.5,
        "body_global_linvel": 1.0,
        "body_global_angvel": 1.0,
        "torso_world_angvel": 3.0,
        "root_orientation": 1.0,
        "torso_world_orientation": 3.0,
        "torso_height_tracking": 2.0,
        "joint_pos_tracking": 0.75,
        "joint_vel_tracking": 0.75,
        "feet_height_tracking": 2.0,
        "residual_action_rate": -0.1,
        "penalty_torque": -1.0e-5,
        "smoothness_joint": -1.0e-6,
        "feet_slip": -0.5,
        "dof_pos_limit": -5.0,
        "dof_vel_limit": -10.0,
        "undesired_self_collision": -10.0,
        "undesired_ground_contact": -10.0,
        "termination": -100.0,
    }
    assert {name: cfg.weight for name, cfg in rewards.items()} == expected_weights
    assert "body_local_linvel" not in rewards
    assert "body_local_angvel" not in rewards
    assert "root_linvel_tracking" not in rewards
    assert "root_angvel_tracking" not in rewards
    assert rewards["body_local_pos"].params == {"sigma": 0.30}
    assert rewards["feet_pos_tracking"].params == {"sigma": 1.0}
    assert rewards["body_local_rot"].params == {"sigma": 1.0}
    assert rewards["body_global_linvel"].params == {"sigma": 1.00}
    assert rewards["body_global_angvel"].params == {"sigma": 3.14}
    assert rewards["joint_vel_tracking"].params == {"sigma": 1.0}
    assert rewards["root_orientation"].params == {"sigma": 0.50}
    assert rewards["joint_vel_tracking"].func is rew.joint_velocity_tracking_exp
    assert rewards["root_orientation"].func is rew.root_orientation_tracking_exp
    assert rewards["feet_pos_tracking"].func is rew.feet_local_position_tracking_exp
    assert rewards["torso_world_angvel"].params == {"sigma": 6.0}
    assert rewards["torso_world_angvel"].func is rew.torso_world_angular_velocity_tracking_exp
    assert rewards["torso_world_orientation"].params == {"sigma": 0.50}
    assert rewards["torso_world_orientation"].func is rew.torso_world_orientation_tracking_exp

    motion_cfg = cast(Gr3MotionCommandCfg, gr3mini_diff_critic_env_cfg().commands["motion"])
    assert motion_cfg.sampling_mode == "adaptive"
    assert motion_cfg.adaptive_bin_duration_s == 0.25
    assert motion_cfg.adaptive_uniform_ratio == 0.25
    assert motion_cfg.adaptive_tracking_error_weight == 0.25


def test_tracking_terminations_only_reset_falls_or_invalid_states() -> None:
    terminations = gr3mini_diff_critic_env_cfg().terminations

    assert set(terminations) == {
        "time_out",
        "root_height",
        "shoulder_height",
        "body_position",
        "torso_ground_contact_too_long",
        "invalid_state",
    }
    assert terminations["root_height"].func is term.bad_root_height
    assert terminations["shoulder_height"].func is term.bad_shoulder_height
    assert terminations["shoulder_height"].params == {"threshold": 0.30}
    assert terminations["body_position"].func is term.bad_body_position
    assert terminations["body_position"].params == {"threshold": 0.50}
    assert terminations["torso_ground_contact_too_long"].func is term.torso_ground_contact_too_long
    assert terminations["torso_ground_contact_too_long"].params == {"duration_s": 1.0}
    assert terminations["invalid_state"].func is term.invalid_state


def test_invalid_state_detects_nonfinite_root_and_joint_state() -> None:
    data = SimpleNamespace(
        root_link_pos_w=torch.zeros(2, 3),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        root_link_lin_vel_b=torch.zeros(2, 3),
        root_link_ang_vel_b=torch.zeros(2, 3),
        joint_pos=torch.zeros(2, 2),
        joint_vel=torch.zeros(2, 2),
    )
    command = SimpleNamespace(robot=SimpleNamespace(data=data))
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command)),
    )

    torch.testing.assert_close(term.invalid_state(env), torch.tensor([False, False]))
    data.root_link_pos_w[1, 0] = torch.inf
    data.joint_vel[0, 0] = torch.nan
    torch.testing.assert_close(term.invalid_state(env), torch.tensor([True, True]))


def test_console_episode_extras_are_grouped() -> None:
    from gr3mini_tracking.console_logging import group_episode_extras

    console = (
        "header\n"
        " Metrics/motion/error_body_pos: 0.3000\n"
        "Episode_Reward/feet_slip: -0.0380\n"
        "Episode_Termination/root_height: 16.4500\n"
        "Episode_Reward/body_local_pos: 0.0672\n"
        "footer\n"
    )

    assert group_episode_extras(console) == (
        "header\n"
        "\n[Episode_Reward]\n"
        "Episode_Reward/body_local_pos: 0.0672\n"
        "Episode_Reward/feet_slip: -0.0380\n"
        "\n[Episode_Termination]\n"
        "Episode_Termination/root_height: 16.4500\n"
        "\n[Metrics]\n"
        " Metrics/motion/error_body_pos: 0.3000\n"
        "footer\n"
    )


def test_tracking_rewards_are_normalized_at_zero_error() -> None:
    body_names = (
        "base_link",
        "torso_link",
        "left_foot_roll_link",
        "right_foot_roll_link",
    )
    body_count = len(body_names)
    body_pos = torch.zeros(1, body_count, 3)
    body_quat = torch.zeros(1, body_count, 4)
    body_quat[..., 0] = 1.0
    body_vel = torch.zeros(1, body_count, 3)
    robot_data = SimpleNamespace(
        root_link_pos_w=body_pos[:, 0],
        root_link_quat_w=body_quat[:, 0],
        root_link_lin_vel_b=torch.zeros(1, 3),
        root_link_ang_vel_b=torch.zeros(1, 3),
        site_pose_w=torch.zeros(1, 2, 7),
        joint_vel=torch.zeros(1, 2),
    )
    robot = SimpleNamespace(
        data=robot_data,
        find_sites=lambda names, preserve_order: ([0, 1], list(names)),
    )
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=body_names),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_vel,
        body_ang_vel_w=body_vel,
        robot_body_pos_w=body_pos.clone(),
        robot_body_quat_w=body_quat.clone(),
        robot_body_lin_vel_w=body_vel.clone(),
        robot_body_ang_vel_w=body_vel.clone(),
        root_lin_vel_w=torch.zeros(1, 3),
        root_ang_vel_b=torch.zeros(1, 3),
        feet_height_w=torch.zeros(1, 2),
        joint_vel=torch.zeros(1, 2),
        robot_joint_vel=torch.zeros(1, 2),
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            device="cpu",
            step_dt=0.02,
            scene={
                "robot": robot,
                "feet_ground_contact": SimpleNamespace(
                    data=SimpleNamespace(found=torch.zeros(1, 2))
                ),
                "self_collision_0": SimpleNamespace(data=SimpleNamespace(found=torch.zeros(1, 1))),
                "self_collision_1": SimpleNamespace(data=SimpleNamespace(found=torch.zeros(1, 1))),
                "undesired_ground_contact": SimpleNamespace(
                    data=SimpleNamespace(found=torch.zeros(1, 3))
                ),
            },
            command_manager=SimpleNamespace(get_term=lambda name: command),
        ),
    )

    for func, sigma in (
        (rew.body_local_position_tracking_exp, 0.30),
        (rew.feet_local_position_tracking_exp, 1.0),
        (rew.body_local_orientation_tracking_exp, 1.00),
        (rew.body_global_linear_velocity_tracking_exp, 1.00),
        (rew.body_global_angular_velocity_tracking_exp, 3.14),
        (rew.torso_world_angular_velocity_tracking_exp, 6.0),
        (rew.root_orientation_tracking_exp, 0.50),
        (rew.torso_world_orientation_tracking_exp, 0.50),
        (rew.torso_height_tracking_exp, 0.15),
        (rew.feet_height_tracking_exp, 0.15),
        (rew.joint_velocity_tracking_exp, 1.00),
    ):
        torch.testing.assert_close(func(env, sigma=sigma), torch.ones(1))

    command.robot_body_ang_vel_w[:, 1, 0] = 6.0
    torch.testing.assert_close(
        rew.torso_world_angular_velocity_tracking_exp(env, sigma=6.0),
        torch.exp(torch.tensor([-1.0])),
    )

    command.robot_body_pos_w[:, 2, 0] = 0.5
    torch.testing.assert_close(
        rew.feet_local_position_tracking_exp(env, sigma=1.0),
        torch.exp(torch.tensor([-0.5])),
    )
    command.robot_body_pos_w[:, 2, 0] = 0.0

    command.robot_body_quat_w[:, 1] = torch.tensor(
        [torch.cos(torch.tensor(0.5)), 0.0, 0.0, torch.sin(torch.tensor(0.5))]
    )
    torch.testing.assert_close(
        rew.body_local_orientation_tracking_exp(env, sigma=1.0),
        torch.exp(torch.tensor([-1.0 / body_count])),
    )
    torch.testing.assert_close(
        rew.torso_world_orientation_tracking_exp(env, sigma=0.5),
        torch.exp(torch.tensor([-2.0])),
    )
    command.robot_body_quat_w[:, 1] = body_quat[:, 1]

    robot.data.root_link_quat_w = torch.tensor(
        [[torch.cos(torch.tensor(0.5)), 0.0, 0.0, torch.sin(torch.tensor(0.5))]]
    )
    torch.testing.assert_close(
        rew.root_orientation_tracking_exp(env, sigma=0.5),
        torch.exp(torch.tensor([-2.0])),
    )
    robot.data.root_link_quat_w = body_quat[:, 0]

    command.robot_body_pos_w[:, 1, 2] = 0.30
    torch.testing.assert_close(
        rew.torso_height_tracking_exp(env, sigma=0.15),
        torch.exp(torch.tensor([-2.0])),
    )
    command.robot_body_pos_w[:, 1, 2] = 0.0

    robot.data.site_pose_w[:, :, 2] = 0.30
    torch.testing.assert_close(
        rew.feet_height_tracking_exp(env, sigma=0.15),
        torch.exp(torch.tensor([-2.0])),
    )
    robot.data.site_pose_w[:, :, 2] = 0.0

    command.robot_body_pos_w[:, 1, 0] = 0.30
    torch.testing.assert_close(
        rew.body_local_position_tracking_exp(env, sigma=0.30), torch.exp(torch.tensor([-0.25]))
    )
    command.robot_body_lin_vel_w[:, 2, 0] = 0.30
    env.scene["feet_ground_contact"].data.found[:, 0] = 1.0
    torch.testing.assert_close(rew.feet_slip(env), torch.tensor([0.0625]))
    command.robot_joint_vel[:, 0] = 5.0
    torch.testing.assert_close(
        rew.joint_velocity_tracking_exp(env, sigma=1.0), torch.exp(torch.tensor([-0.10]))
    )
    env.scene["self_collision_0"].data.found[:] = 1.0
    env.scene["self_collision_1"].data.found[:] = 1.0
    env.scene["undesired_ground_contact"].data.found[:, [0, 2]] = 1.0
    torch.testing.assert_close(
        rew.undesired_self_collision_count(
            env, sensor_names=("self_collision_0", "self_collision_1")
        ),
        torch.tensor([2.0]),
    )
    torch.testing.assert_close(rew.undesired_ground_contact_count(env), torch.tensor([2.0]))


def test_torso_ground_contact_termination_uses_continuous_contact_time() -> None:
    contact_time = torch.tensor([[0.999], [1.0], [1.25]])
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            scene={
                "torso_ground_contact": SimpleNamespace(
                    data=SimpleNamespace(current_contact_time=contact_time)
                )
            }
        ),
    )

    torch.testing.assert_close(
        term.torso_ground_contact_too_long(env, duration_s=1.0),
        torch.tensor([False, True, True]),
    )
    with pytest.raises(ValueError, match="duration_s must be positive"):
        term.torso_ground_contact_too_long(env, duration_s=0.0)


def test_residual_action_rate_excludes_reference_motion() -> None:
    action = SimpleNamespace(
        raw_action=torch.tensor([[0.25, -0.25]]),
        previous_raw_actions=torch.tensor([[0.05, -0.05]]),
    )
    env = cast(
        ManagerBasedRlEnv,
        SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: action)),
    )
    torch.testing.assert_close(rew.residual_action_rate_l2(env), torch.tensor([0.08]))


def test_adaptive_sampling_contracts() -> None:
    assert temporal_bin_count(2647, 0.02, 0.25) == 212
    probabilities = adaptive_sampling_probabilities(torch.tensor([0.0, 1.0, 0.0, 0.0]), 0.25)
    torch.testing.assert_close(probabilities, torch.tensor([0.0625, 0.8125, 0.0625, 0.0625]))
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))


def test_isaaclab_motion_conversion_contract(tmp_path) -> None:
    source_joint_names = tuple(
        name for name in JOINT_NAMES if name not in {"head_yaw_joint", "head_pitch_joint"}
    )
    source = tmp_path / "isaaclab_motion.npz"
    destination = tmp_path / "mjlab_motion.npz"
    joint_pos = np.zeros((2, len(source_joint_names)), dtype=np.float32)
    body_pos = np.zeros((2, 1, 3), dtype=np.float32)
    body_quat = np.zeros((2, 1, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez_compressed(
        source,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=np.zeros_like(joint_pos),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        joint_names=np.asarray(source_joint_names),
        body_names=np.asarray(("base_link",)),
        fixed_joint_names=np.asarray(("head_yaw_joint", "head_pitch_joint")),
    )

    convert_motion_file(source, destination)

    with np.load(destination, allow_pickle=False) as converted:
        assert converted["joint_pos"].shape == (2, len(JOINT_NAMES))
        assert converted["body_pos_w"].shape == (2, len(EXPECTED_BODY_NAMES), 3)
        assert converted["feet_height_w"].shape == (2, 2)
        assert tuple(converted["joint_names"]) == JOINT_NAMES
        assert tuple(converted["body_names"]) == EXPECTED_BODY_NAMES
        np.testing.assert_allclose(converted["root_projected_gravity_b"], [[0.0, 0.0, -1.0]] * 2)


def test_rotation_6d_is_column_major_and_identity_at_zero_error() -> None:
    identity_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(
        _rotation_6d(identity_quaternion),
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
    )


def test_history_encoder_shape() -> None:
    encoder = HistoryEncoder()
    history = torch.zeros(3, ADAPTER_HISTORY_LENGTH * FRAME_DIM)
    assert encoder(history).shape == (3, 128)


def test_tanh_gaussian_contract() -> None:
    distribution = TanhGaussianDistribution(25)
    parameters = torch.zeros(4, 2, 25)
    distribution.update(parameters)
    actions = distribution.sample()
    assert actions.shape == (4, 25)
    assert torch.all(actions > -1.0)
    assert torch.all(actions < 1.0)
    assert torch.isfinite(distribution.log_prob(actions)).all()
    assert torch.isfinite(distribution.entropy).all()
    torch.testing.assert_close(distribution.deterministic_output(parameters), torch.zeros(4, 25))


def test_tanh_gaussian_caps_raw_standard_deviation() -> None:
    distribution = TanhGaussianDistribution(2, min_std=0.001, max_std=1.0)
    distribution.update(torch.tensor([[[0.0, 0.0], [100.0, 100.0]]]))

    torch.testing.assert_close(distribution.std, torch.ones(1, 2))


def test_tanh_gaussian_preserves_raw_samples_for_ppo() -> None:
    distribution = TanhGaussianDistribution(1)
    distribution.update(torch.tensor([[[20.0], [0.0]]]))
    bounded = distribution.sample()
    raw = distribution.raw_sample

    torch.testing.assert_close(bounded, torch.tanh(raw))
    torch.testing.assert_close(distribution.log_prob_raw(raw), -distribution.entropy)
    # Reconstructing a saturated raw sample from tanh(raw) changes the likelihood.
    assert not torch.allclose(distribution.log_prob(bounded), distribution.log_prob_raw(raw))


def test_tanh_ppo_stores_raw_actions_and_executes_bounded_actions() -> None:
    observations = TensorDict(
        {"actor": torch.zeros(3, 2), "critic": torch.zeros(3, 2)}, batch_size=[3]
    )
    actor = TanhMLPModel(
        observations,
        {"actor": ["actor"], "critic": ["critic"]},
        "actor",
        1,
        hidden_dims=(4,),
        distribution_cfg={
            "class_name": "gr3mini_tracking.adapter.distribution:TanhGaussianDistribution"
        },
    )
    critic = MLPModel(
        observations,
        {"actor": ["actor"], "critic": ["critic"]},
        "critic",
        1,
        hidden_dims=(4,),
    )
    storage = RolloutStorage("rl", 3, 1, observations, [1])
    algorithm = TanhPPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1)

    applied_actions = algorithm.act(observations)
    raw_actions = algorithm.transition.actions
    assert raw_actions is not None
    torch.testing.assert_close(applied_actions, torch.tanh(raw_actions))
    torch.testing.assert_close(
        algorithm.transition.actions_log_prob,
        actor.get_output_log_prob(raw_actions),
    )

    algorithm.process_env_step(
        observations,
        torch.zeros(3),
        torch.zeros(3, dtype=torch.bool),
        {},
    )
    torch.testing.assert_close(storage.actions[0], raw_actions)
    algorithm.compute_returns(observations)
    losses = algorithm.update()
    assert all(np.isfinite(value) for value in losses.values())


def test_adapter_branches_start_at_zero() -> None:
    from tensordict import TensorDict

    from gr3mini_tracking.adapter.models import ResidualAdapterActor

    encoder = HistoryEncoder()
    observations = TensorDict(
        {
            "actor": torch.zeros(2, ACTOR_OBS_DIM),
            "history": torch.zeros(2, ADAPTER_HISTORY_LENGTH * FRAME_DIM),
        },
        batch_size=[2],
    )
    actor = ResidualAdapterActor(
        observations,
        {"actor": ["actor"]},
        "actor",
        25,
        history_encoder=encoder,
        hidden_dims=(512, 512, 256, 256, 128),
        activation="swish",
        distribution_cfg={
            "class_name": "gr3mini_tracking.adapter.distribution:TanhGaussianDistribution"
        },
    )
    assert all(not parameter.requires_grad for parameter in actor.mlp.parameters())
    for layer in actor.adapter_layers:
        assert isinstance(layer, nn.Linear)
        assert torch.count_nonzero(layer.weight) == 0
        assert torch.count_nonzero(layer.bias) == 0
    with torch.no_grad():
        teacher_parameters = actor.mlp(observations["actor"])
        teacher_action = torch.tanh(teacher_parameters[..., 0, :])
        torch.testing.assert_close(actor(observations), teacher_action)

    frozen_before = [parameter.detach().clone() for parameter in actor.mlp.parameters()]
    optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad], lr=1e-3
    )
    optimizer.zero_grad()
    actor(observations).sum().backward()
    optimizer.step()
    for before, after in zip(frozen_before, actor.mlp.parameters(), strict=True):
        torch.testing.assert_close(before, after)


def test_world_model_shape_and_normalized_gravity() -> None:
    model = WorldModel()
    state = torch.zeros(4, WORLD_STATE_DIM)
    state[:, 5] = -1.0
    prediction = model(state, torch.zeros(4, 128), torch.zeros(4, 25))
    assert prediction.shape == (4, WORLD_STATE_DIM)
    torch.testing.assert_close(torch.linalg.vector_norm(prediction[:, 3:6], dim=-1), torch.ones(4))
