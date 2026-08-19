"""Task-specific domain-randomization events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from gr3mini_tracking.robots.motor_model import (
    RFI_LIM_SCALE,
    TorqueSpeedPdActuator,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def randomize_foot_contact_softness(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    timeconst_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomize foot geom solref[0] (contact time constant) to simulate different ground materials.

    Lower values (~0.005s) = hard ground; higher values (~0.04s) = soft/compliant surface.
    MuJoCo default is 0.02s.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids].to(torch.long)
    n_envs = len(env_ids)
    n_geoms = len(geom_ids)
    timeconst = torch.empty(n_envs, n_geoms, device=env.device).uniform_(*timeconst_range)
    env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
    env.sim.model.geom_solref[env_grid, geom_grid, 0] = timeconst


def push_planar_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    magnitude_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Add a random-direction planar velocity kick, matching the source task."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    velocity = asset.data.root_link_vel_w[env_ids].clone()
    theta = 2.0 * torch.pi * torch.rand(len(env_ids), device=env.device)
    magnitude = torch.empty(len(env_ids), device=env.device).uniform_(*magnitude_range)
    velocity[:, 0] += torch.cos(theta) * magnitude
    velocity[:, 1] += torch.sin(theta) * magnitude
    asset.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)


def randomize_motor_rfi_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    rfi_lim_scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Sample the episode-level RFI torque-noise scale for each selected motor."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]

    if isinstance(asset_cfg.actuator_ids, list):
        actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
    elif isinstance(asset_cfg.actuator_ids, slice):
        actuators = asset.actuators[asset_cfg.actuator_ids]
    else:
        actuators = [asset.actuators[asset_cfg.actuator_ids]]

    for actuator in actuators:
        if not isinstance(actuator, TorqueSpeedPdActuator):
            raise TypeError(
                "randomize_motor_rfi_scale expects TorqueSpeedPdActuator instances, "
                f"got {type(actuator).__name__}"
            )
        assert actuator.rfi_lim_scale is not None
        scale = torch.empty(len(env_ids), actuator.num_targets, device=env.device).uniform_(
            *rfi_lim_scale_range
        )
        scale = scale * RFI_LIM_SCALE
        actuator.set_rfi_lim_scale(env_ids, scale)
