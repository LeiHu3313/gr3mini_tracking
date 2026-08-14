from __future__ import annotations

import torch
from torch import nn

from gr3mini_tracking.adapter.distribution import TanhGaussianDistribution
from gr3mini_tracking.adapter.models import HistoryEncoder, WorldModel
from gr3mini_tracking.robots.gr3mini211 import JOINT_NAMES, TRACKED_BODY_NAMES
from gr3mini_tracking.tasks.tracking.observations import (
    ACTOR_OBS_DIM,
    ADAPTER_HISTORY_LENGTH,
    CRITIC_OBS_DIM,
    FRAME_DIM,
    WORLD_STATE_DIM,
)


def test_observation_contract_dimensions() -> None:
    assert ACTOR_OBS_DIM == 671
    assert CRITIC_OBS_DIM == 662
    assert ADAPTER_HISTORY_LENGTH * FRAME_DIM == 6399
    assert WORLD_STATE_DIM == 57
    assert len(JOINT_NAMES) == 25
    assert len(TRACKED_BODY_NAMES) == 26


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
