from __future__ import annotations

from types import SimpleNamespace

import torch

from gr3mini_tracking.robots.gr3mini211 import get_gr3mini211_robot_cfg
from gr3mini_tracking.robots.motor_model import (
    RFI_LIM_SCALE_RANGE,
    TN_X1,
    TN_X2,
    TN_X3,
    TN_Y1,
    TN_Y2,
    TN_Y3,
    TorqueSpeedPdActuatorCfg,
    apply_gr3mini211_motor_model,
)
from gr3mini_tracking.tasks.tracking.env_cfg import gr3mini_diff_critic_env_cfg
from gr3mini_tracking.tasks.tracking.events import randomize_motor_rfi_scale


def _scalar(value: float) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.float)


def test_gr3mini211_robot_cfg_uses_torque_speed_actuators() -> None:
    robot_cfg = get_gr3mini211_robot_cfg()

    assert len(robot_cfg.articulation.actuators) == len(TN_X1)
    first = robot_cfg.articulation.actuators[0]
    assert isinstance(first, TorqueSpeedPdActuatorCfg)
    assert first.tn_x1 == TN_X1[0]
    assert first.tn_x2 == TN_X2[0]
    assert first.tn_x3 == TN_X3[0]
    assert first.tn_y1 == TN_Y1[0]
    assert first.tn_y2 == TN_Y2[0]
    assert first.tn_y3 == TN_Y3[0]


def test_gr3mini211_motor_model_matches_piecewise_limits() -> None:
    base_args = dict(
        force_limit=_scalar(140.0),
        tn_x1=_scalar(TN_X1[0]),
        tn_x2=_scalar(TN_X2[0]),
        tn_x3=_scalar(TN_X3[0]),
        tn_y1=_scalar(TN_Y1[0]),
        tn_y2=_scalar(TN_Y2[0]),
        tn_y3=_scalar(TN_Y3[0]),
        static_friction=_scalar(0.0),
        dynamic_friction=_scalar(0.0),
        friction_activation_velocity=_scalar(0.01),
    )

    low_speed = apply_gr3mini211_motor_model(
        effort=_scalar(200.0),
        joint_vel=_scalar(0.0),
        rfi_noise=None,
        **base_args,
    )
    torch.testing.assert_close(low_speed, _scalar(140.0))

    mid_speed = 0.5 * (TN_X1[0] + TN_X2[0])
    middle_segment = apply_gr3mini211_motor_model(
        effort=_scalar(200.0),
        joint_vel=_scalar(mid_speed),
        rfi_noise=None,
        **base_args,
    )
    torch.testing.assert_close(middle_segment, _scalar(105.0), atol=1.0e-5, rtol=1.0e-5)

    braking = apply_gr3mini211_motor_model(
        effort=_scalar(-200.0),
        joint_vel=_scalar(mid_speed),
        rfi_noise=None,
        **base_args,
    )
    torch.testing.assert_close(braking, _scalar(-140.0))

    noisy = apply_gr3mini211_motor_model(
        effort=_scalar(100.0),
        joint_vel=_scalar(0.0),
        rfi_noise=_scalar(10.0),
        **base_args,
    )
    torch.testing.assert_close(noisy, _scalar(110.0))


def test_motor_rfi_scale_randomization_applies_per_env() -> None:
    cfg = TorqueSpeedPdActuatorCfg(
        target_names_expr=("joint",),
        stiffness=1.0,
        damping=1.0,
        effort_limit=120.0,
        tn_x1=TN_X1[0],
        tn_x2=TN_X2[0],
        tn_x3=TN_X3[0],
        tn_y1=TN_Y1[0],
        tn_y2=TN_Y2[0],
        tn_y3=TN_Y3[0],
    )
    actuator = cfg.build(SimpleNamespace(), [0], ["joint"])
    actuator.rfi_lim_scale = torch.zeros(3, 1)
    asset = SimpleNamespace(actuators=[actuator])
    env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot": asset})
    asset_cfg = SimpleNamespace(name="robot", actuator_ids=[0])

    torch.manual_seed(7)
    randomize_motor_rfi_scale(env, torch.tensor([0, 2]), RFI_LIM_SCALE_RANGE, asset_cfg)

    assert torch.allclose(actuator.rfi_lim_scale[1], torch.zeros(1))
    assert torch.all(
        (actuator.rfi_lim_scale[[0, 2]] >= 0.05) & (actuator.rfi_lim_scale[[0, 2]] <= 0.15)
    )


def test_training_env_cfg_includes_reset_time_motor_randomization() -> None:
    teacher_cfg = gr3mini_diff_critic_env_cfg(adapter=False)
    adapter_cfg = gr3mini_diff_critic_env_cfg(adapter=True)

    for cfg in (teacher_cfg, adapter_cfg):
        assert cfg.events["pd_gains"].mode == "reset"
        assert cfg.events["motor_rfi_noise"].mode == "reset"
        assert cfg.events["motor_rfi_noise"].params["rfi_lim_scale_range"] == RFI_LIM_SCALE_RANGE
