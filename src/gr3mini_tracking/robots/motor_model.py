"""GR3Mini211 motor model helpers used by the training tasks.

The source implementation combines three pieces in the control path:

1. A PD torque target.
2. An episode-level RFI scale that injects bounded torque noise.
3. A piecewise torque-speed envelope plus a small friction term.

This module keeps that logic in one place so teacher and adapter can share it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg, pd_torque

if TYPE_CHECKING:
    from mjlab.entity import Entity

RFI_LIM_SCALE = 0.1
RFI_LIM_SCALE_RANGE = (0.5, 1.5)

TN_X1 = (
    5.2359878,
    0.0,
    0.0,
    6.2831853,
    7.8539816,
    7.8539816,
    7.8539816,
    0.0,
    6.2831853,
    7.8539816,
    7.8539816,
    7.8539816,
    0.0,
    5.2359878,
    6.2831853,
    6.2831853,
    5.2359878,
    7.8539816,
    7.8539816,
    5.2359878,
    6.2831853,
    6.2831853,
    5.2359878,
    7.8539816,
    7.8539816,
)

TN_X2 = (
    13.089969,
    5.2359878,
    5.2359878,
    14.660766,
    18.849556,
    18.849556,
    18.849556,
    5.2359878,
    14.660766,
    18.849556,
    18.849556,
    18.849556,
    5.2359878,
    13.089969,
    14.660766,
    14.660766,
    13.089969,
    18.849556,
    18.849556,
    13.089969,
    14.660766,
    14.660766,
    13.089969,
    18.849556,
    18.849556,
)

TN_X3 = TN_X2

TN_Y1 = (
    140.0,
    11.0,
    11.0,
    42.0,
    28.5,
    28.5,
    28.5,
    11.0,
    42.0,
    28.5,
    28.5,
    28.5,
    11.0,
    140.0,
    42.0,
    42.0,
    140.0,
    52.0,
    25.0,
    140.0,
    42.0,
    42.0,
    140.0,
    52.0,
    25.0,
)

TN_Y2 = (
    70.0,
    7.3333333,
    7.3333333,
    21.0,
    10.178571,
    10.178571,
    10.178571,
    7.3333333,
    21.0,
    10.178571,
    10.178571,
    10.178571,
    7.3333333,
    70.0,
    21.0,
    21.0,
    70.0,
    18.571429,
    8.9285714,
    70.0,
    21.0,
    21.0,
    70.0,
    18.571429,
    8.9285714,
)

TN_Y3 = (
    140.0,
    11.0,
    11.0,
    42.0,
    28.5,
    28.5,
    28.5,
    11.0,
    42.0,
    28.5,
    28.5,
    28.5,
    11.0,
    140.0,
    42.0,
    42.0,
    140.0,
    52.0,
    25.0,
    140.0,
    42.0,
    42.0,
    140.0,
    52.0,
    25.0,
)

STATIC_FRICTION = 0.0
DYNAMIC_FRICTION = 0.01
FRICTION_ACTIVATION_VELOCITY = 0.01


def apply_gr3mini211_motor_model(
    effort: torch.Tensor,
    joint_vel: torch.Tensor,
    force_limit: torch.Tensor,
    tn_x1: torch.Tensor,
    tn_x2: torch.Tensor,
    tn_x3: torch.Tensor,
    tn_y1: torch.Tensor,
    tn_y2: torch.Tensor,
    tn_y3: torch.Tensor,
    static_friction: torch.Tensor,
    dynamic_friction: torch.Tensor,
    friction_activation_velocity: torch.Tensor,
    rfi_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the source torque-speed envelope and friction model."""
    if rfi_noise is not None:
        effort = effort + rfi_noise

    same_direction = joint_vel * effort > 0.0
    abs_vel = torch.abs(joint_vel)

    x2_minus_x1 = torch.clamp(tn_x2 - tn_x1, min=1.0e-6)
    motor_d1 = (tn_y2 - tn_y1) / x2_minus_x1 * (abs_vel - tn_x1) + tn_y1

    x3_minus_x2 = torch.clamp(tn_x3 - tn_x2, min=1.0e-6)
    motor_d2 = -tn_y2 / x3_minus_x2 * (abs_vel - tn_x2) + tn_y2

    motor_limit = torch.where(
        abs_vel <= tn_x1,
        tn_y1,
        torch.where(
            abs_vel <= tn_x2,
            motor_d1,
            torch.where(abs_vel <= tn_x3, motor_d2, torch.zeros_like(tn_y1)),
        ),
    )
    motor_limit = torch.clamp(motor_limit, min=0.0)
    brake_limit = tn_y3
    velocity_limited_torque = torch.minimum(
        torch.where(same_direction, motor_limit, brake_limit),
        force_limit,
    )
    effort = torch.clamp(effort, -velocity_limited_torque, velocity_limited_torque)

    friction_activation_velocity = torch.clamp(friction_activation_velocity, min=1.0e-6)
    friction = static_friction * torch.tanh(joint_vel / friction_activation_velocity)
    friction = friction + dynamic_friction * joint_vel
    return effort - friction


@dataclass(kw_only=True)
class TorqueSpeedPdActuatorCfg(IdealPdActuatorCfg):
    """PD actuator with the GR3Mini211 torque-speed envelope."""

    tn_x1: float
    tn_x2: float
    tn_x3: float
    tn_y1: float
    tn_y2: float
    tn_y3: float
    static_friction: float = STATIC_FRICTION
    dynamic_friction: float = DYNAMIC_FRICTION
    friction_activation_velocity: float = FRICTION_ACTIVATION_VELOCITY

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> TorqueSpeedPdActuator:
        return TorqueSpeedPdActuator(self, entity, target_ids, target_names)


class TorqueSpeedPdActuator(IdealPdActuator[TorqueSpeedPdActuatorCfg]):
    """Ideal PD actuator with the source torque-speed and friction model."""

    param_names: ClassVar[tuple[str, ...]] = (
        *IdealPdActuator.param_names,
        "tn_x1",
        "tn_x2",
        "tn_x3",
        "tn_y1",
        "tn_y2",
        "tn_y3",
        "static_friction",
        "dynamic_friction",
        "friction_activation_velocity",
        "rfi_lim_scale",
    )

    def __init__(
        self,
        cfg: TorqueSpeedPdActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self.tn_x1: torch.Tensor | None = None
        self.tn_x2: torch.Tensor | None = None
        self.tn_x3: torch.Tensor | None = None
        self.tn_y1: torch.Tensor | None = None
        self.tn_y2: torch.Tensor | None = None
        self.tn_y3: torch.Tensor | None = None
        self.static_friction: torch.Tensor | None = None
        self.dynamic_friction: torch.Tensor | None = None
        self.friction_activation_velocity: torch.Tensor | None = None
        self.rfi_lim_scale: torch.Tensor | None = None
        self.default_rfi_lim_scale: torch.Tensor | None = None

    @property
    def num_targets(self) -> int:
        return len(self._target_ids_list)

    @staticmethod
    def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
        torque = pd_torque(params["stiffness"], params["damping"], cmd)
        rfi_noise = (
            (2.0 * torch.rand_like(torque) - 1.0) * params["rfi_lim_scale"] * params["force_limit"]
        )
        return apply_gr3mini211_motor_model(
            torque,
            cmd.vel,
            params["force_limit"],
            params["tn_x1"],
            params["tn_x2"],
            params["tn_x3"],
            params["tn_y1"],
            params["tn_y2"],
            params["tn_y3"],
            params["static_friction"],
            params["dynamic_friction"],
            params["friction_activation_velocity"],
            rfi_noise=rfi_noise,
        )

    def initialize(
        self,
        mj_model,
        model,
        data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)

        num_envs = data.nworld
        num_joints = len(self._target_names)
        self.tn_x1 = torch.full(
            (num_envs, num_joints), self.cfg.tn_x1, dtype=torch.float, device=device
        )
        self.tn_x2 = torch.full(
            (num_envs, num_joints), self.cfg.tn_x2, dtype=torch.float, device=device
        )
        self.tn_x3 = torch.full(
            (num_envs, num_joints), self.cfg.tn_x3, dtype=torch.float, device=device
        )
        self.tn_y1 = torch.full(
            (num_envs, num_joints), self.cfg.tn_y1, dtype=torch.float, device=device
        )
        self.tn_y2 = torch.full(
            (num_envs, num_joints), self.cfg.tn_y2, dtype=torch.float, device=device
        )
        self.tn_y3 = torch.full(
            (num_envs, num_joints), self.cfg.tn_y3, dtype=torch.float, device=device
        )
        self.static_friction = torch.full(
            (num_envs, num_joints), self.cfg.static_friction, dtype=torch.float, device=device
        )
        self.dynamic_friction = torch.full(
            (num_envs, num_joints), self.cfg.dynamic_friction, dtype=torch.float, device=device
        )
        self.friction_activation_velocity = torch.full(
            (num_envs, num_joints),
            self.cfg.friction_activation_velocity,
            dtype=torch.float,
            device=device,
        )
        self.rfi_lim_scale = torch.zeros((num_envs, num_joints), dtype=torch.float, device=device)
        self.default_rfi_lim_scale = self.rfi_lim_scale.clone()

    def set_rfi_lim_scale(self, env_ids: torch.Tensor | slice, rfi_lim_scale: torch.Tensor) -> None:
        """Set the episode-level RFI torque noise amplitude."""
        assert self.rfi_lim_scale is not None
        if rfi_lim_scale.ndim == 1:
            rfi_lim_scale = rfi_lim_scale.unsqueeze(-1)
        self.rfi_lim_scale[env_ids] = rfi_lim_scale
