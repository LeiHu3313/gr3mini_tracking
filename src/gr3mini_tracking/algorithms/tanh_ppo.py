"""PPO support for Brax-compatible tanh-normal action distributions."""

from __future__ import annotations

from typing import cast

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from gr3mini_tracking.adapter.distribution import TanhGaussianDistribution


class TanhRawActionModelMixin:
    """Expose a tanh-normal model's pre-tanh sample to PPO."""

    def _tanh_distribution(self) -> TanhGaussianDistribution:
        distribution = getattr(self, "distribution", None)
        if not isinstance(distribution, TanhGaussianDistribution):
            raise TypeError("TanhPPO requires TanhGaussianDistribution")
        return distribution

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._tanh_distribution().raw_sample

    def get_output_log_prob(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return self._tanh_distribution().log_prob_raw(raw_actions)


class TanhMLPModel(TanhRawActionModelMixin, MLPModel):
    """Standard MLP actor with the raw-action interface required by ``TanhPPO``."""


class TanhPPO(PPO):
    """Store pre-tanh actions for PPO while executing their bounded counterparts.

    Brax's ``NormalTanhDistribution`` computes likelihoods from the Gaussian
    sample, not from a reconstructed ``atanh(tanh(sample))``. Reconstructing
    fails for saturated actions and corrupts PPO's probability ratio.
    """

    applied_actions: torch.Tensor | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.applied_actions = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample a bounded environment action and retain its raw PPO action."""
        self.transition.hidden_states = (
            self.actor.get_hidden_state(),
            self.critic.get_hidden_state(),
        )
        applied_actions = self.actor(obs, stochastic_output=True).detach()
        tanh_actor = cast(TanhRawActionModelMixin, self._raw_actor)
        try:
            raw_actions = tanh_actor.raw_actions.detach()
        except AttributeError as error:
            raise TypeError(
                "TanhPPO requires an actor implementing TanhRawActionModelMixin"
            ) from error
        self.transition.actions = raw_actions
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = tanh_actor.get_output_log_prob(raw_actions).detach()
        self.transition.distribution_params = tuple(
            p.detach() for p in self.actor.output_distribution_params
        )
        self.transition.observations = obs
        self.applied_actions = applied_actions
        return applied_actions
