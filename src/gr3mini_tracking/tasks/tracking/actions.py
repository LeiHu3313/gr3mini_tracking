"""Reference-relative residual action used by both stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from .commands import Gr3MotionCommand


@dataclass(kw_only=True)
class ReferenceResidualJointPositionActionCfg(JointPositionActionCfg):
    command_name: str = "motion"

    def build(self, env) -> ReferenceResidualJointPositionAction:
        return ReferenceResidualJointPositionAction(self, env)


class ReferenceResidualJointPositionAction(JointPositionAction):
    """Set target to the current reference joint position plus policy residual."""

    cfg: ReferenceResidualJointPositionActionCfg

    def __init__(self, cfg: ReferenceResidualJointPositionActionCfg, env) -> None:
        # This term supplies its own reference offset.
        cfg.use_default_offset = False
        super().__init__(cfg, env)
        current = self._entity.data.joint_pos[:, self._target_ids]
        self._last_target = current.clone()
        self._previous_target = current.clone()

    @property
    def last_target(self) -> torch.Tensor:
        return self._last_target

    @property
    def previous_target(self) -> torch.Tensor:
        return self._previous_target

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        command = cast(
            Gr3MotionCommand,
            self._env.command_manager.get_term(self.cfg.command_name),
        )
        self._previous_target.copy_(self._last_target)
        self._processed_actions = command.joint_pos[:, self._target_ids] + actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        self._last_target.copy_(self._processed_actions)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        current = self._entity.data.joint_pos[:, self._target_ids]
        self._last_target[env_ids] = current[env_ids]
        self._previous_target[env_ids] = current[env_ids]
