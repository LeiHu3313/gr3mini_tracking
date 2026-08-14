"""Teacher-compatible residual adapter, history encoder, and world model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict
from torch import nn

from gr3mini_tracking.tasks.tracking.observations import (
    ADAPTER_HISTORY_LENGTH,
    FRAME_DIM,
    VELOCITY_SCALE,
    WORLD_STATE_DIM,
)

ADAPTER_EMBEDDING_DIM = 128


class HistoryEncoder(nn.Module):
    """79x81 history -> 128-D dynamics embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(FRAME_DIM, 64, kernel_size=9, stride=5),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=6, stride=3),
            nn.SiLU(),
        )
        self.projection = nn.Linear(64 * 4, ADAPTER_EMBEDDING_DIM)
        self.activation = nn.SiLU()

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[-1] == ADAPTER_HISTORY_LENGTH * FRAME_DIM:
            history = history.reshape(*history.shape[:-1], ADAPTER_HISTORY_LENGTH, FRAME_DIM)
        if history.shape[-2:] != (ADAPTER_HISTORY_LENGTH, FRAME_DIM):
            raise ValueError(
                f"history must end in {(ADAPTER_HISTORY_LENGTH, FRAME_DIM)}, got {history.shape}"
            )
        leading = history.shape[:-2]
        history = history.reshape(-1, ADAPTER_HISTORY_LENGTH, FRAME_DIM).transpose(1, 2)
        encoded = self.conv(history).flatten(start_dim=1)
        encoded = self.activation(self.projection(encoded))
        return encoded.reshape(*leading, ADAPTER_EMBEDDING_DIM)


class ResidualAdapterActor(MLPModel):
    """Frozen teacher MLP with a zero-initialized adapter at every linear layer."""

    history_encoder: HistoryEncoder

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        history_encoder: HistoryEncoder,
        history_group: str = "history",
        **kwargs: Any,
    ) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        object.__setattr__(self, "history_encoder", history_encoder)
        self.history_group = history_group
        linears = [layer for layer in self.mlp if isinstance(layer, nn.Linear)]
        adapter_layers: list[nn.Linear] = []
        for index, layer in enumerate(linears):
            input_dim = ADAPTER_EMBEDDING_DIM if index == 0 else layer.in_features
            adapter = nn.Linear(input_dim, layer.out_features)
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)
            adapter_layers.append(adapter)
        self.adapter_layers = nn.ModuleList(adapter_layers)
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state
        if masks is not None:
            obs = cast(TensorDict, unpad_trajectories(obs, masks))
        base = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        base = self.obs_normalizer(base)
        with torch.no_grad():
            embedding = self.history_encoder(obs[self.history_group])
        adapter_index = 0
        value = base
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                adapter_input = embedding if adapter_index == 0 else value
                value = layer(value) + self.adapter_layers[adapter_index](adapter_input)
                adapter_index += 1
            else:
                value = layer(value)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(value)
                return self.distribution.sample()
            return self.distribution.deterministic_output(value)
        return value

    def load_teacher(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("actor_state_dict", payload.get("model_state_dict"))
        if state is None:
            raise ValueError(f"{checkpoint} does not contain an RSL-RL actor state")
        if "mlp.0.weight" not in state:
            raise ValueError("teacher checkpoint has no mlp.0.weight")
        expected_first = self.mlp[0].weight.shape  # type: ignore[union-attr]
        if state["mlp.0.weight"].shape != expected_first:
            raise ValueError(
                f"teacher actor input mismatch: {state['mlp.0.weight'].shape} != {expected_first}"
            )
        final_index = max(
            int(key.split(".")[1])
            for key in state
            if key.startswith("mlp.") and key.endswith(".weight")
        )
        linear_layers = [layer for layer in self.mlp if isinstance(layer, nn.Linear)]
        expected_final = linear_layers[-1].weight.shape
        if state[f"mlp.{final_index}.weight"].shape != expected_final:
            raise ValueError(
                "teacher actor output mismatch: "
                f"{state[f'mlp.{final_index}.weight'].shape} != {expected_final}"
            )
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed_missing = {key for key in self.state_dict() if key.startswith("adapter_layers.")}
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                f"teacher actor checkpoint mismatch; missing={missing}, unexpected={unexpected}"
            )


class AdapterCritic(MLPModel):
    """Trainable critic conditioned on the frozen-with-respect-to-PPO embedding."""

    history_encoder: HistoryEncoder

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        history_encoder: HistoryEncoder,
        history_group: str = "history",
        hidden_dims=(512, 512, 256, 256, 128),
        activation: str = "swish",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            **kwargs,
        )
        object.__setattr__(self, "history_encoder", history_encoder)
        self.history_group = history_group
        self.mlp = MLP(self.obs_dim + ADAPTER_EMBEDDING_DIM, output_dim, hidden_dims, activation)

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        del masks, hidden_state
        privileged = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        privileged = self.obs_normalizer(privileged)
        with torch.no_grad():
            embedding = self.history_encoder(obs[self.history_group])
        return torch.cat([privileged, embedding], dim=-1)


class WorldModel(nn.Module):
    """Predict 29 dynamics deltas and integrate them into a 57-D next state."""

    def __init__(
        self,
        action_dim: int = 25,
        hidden_dims: tuple[int, ...] = (512, 512, 256, 256, 256, 128),
    ) -> None:
        super().__init__()
        self.delta_model = MLP(
            WORLD_STATE_DIM + ADAPTER_EMBEDDING_DIM + action_dim,
            29,
            hidden_dims,
            "swish",
        )

    def forward(
        self, state: torch.Tensor, embedding: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        delta = self.delta_model(torch.cat([state, embedding, action], dim=-1))
        gyro = state[..., 0:3] + delta[..., 0:3]
        gravity = state[..., 3:6]
        joint_pos = state[..., 6:31]
        joint_vel = state[..., 31:56] + delta[..., 3:28]
        height = state[..., 56:57] + delta[..., 28:29]
        angular_velocity = gyro / VELOCITY_SCALE
        gravity = gravity - torch.cross(angular_velocity, gravity, dim=-1) * 0.02
        gravity = torch.nn.functional.normalize(gravity, dim=-1)
        joint_pos = joint_pos + (joint_vel / VELOCITY_SCALE) * 0.02
        return torch.cat([gyro, gravity, joint_pos, joint_vel, height], dim=-1)


def world_model_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gyro = (prediction[..., 0:3] - target[..., 0:3]).abs().sum(dim=-1).mean()
    gravity = (1.0 - (prediction[..., 3:6] * target[..., 3:6]).sum(dim=-1)).mean()
    joint_pos = (prediction[..., 6:31] - target[..., 6:31]).abs().sum(dim=-1).mean()
    joint_vel = (prediction[..., 31:56] - target[..., 31:56]).abs().sum(dim=-1).mean()
    height = (prediction[..., 56:57] - target[..., 56:57]).abs().sum(dim=-1).mean()
    return 500.0 * gyro + 500.0 * gravity + joint_pos + 0.5 * joint_vel + 500.0 * height
