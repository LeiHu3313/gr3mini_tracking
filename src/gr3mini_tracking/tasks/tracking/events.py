"""Task-specific domain-randomization events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


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
