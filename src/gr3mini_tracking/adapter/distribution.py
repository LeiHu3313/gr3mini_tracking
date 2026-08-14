"""Brax-compatible tanh-normal action distribution for RSL-RL."""

from __future__ import annotations

import math

import torch
from rsl_rl.modules.distribution import Distribution
from torch import nn
from torch.distributions import Normal


class _TanhLocation(nn.Module):
    def forward(self, parameters: torch.Tensor) -> torch.Tensor:
        location = parameters[..., 0, :]
        return torch.tanh(location)


class TanhGaussianDistribution(Distribution):
    """Normal distribution followed by tanh, matching Brax's parameterization."""

    def __init__(self, output_dim: int, min_std: float = 0.001, var_scale: float = 1.0):
        super().__init__(output_dim)
        self.min_std = min_std
        self.var_scale = var_scale
        self._normal: Normal | None = None
        self._location: torch.Tensor | None = None
        self._scale: torch.Tensor | None = None
        self._raw_sample: torch.Tensor | None = None
        self._last_log_prob: torch.Tensor | None = None
        Normal.set_default_validate_args(False)

    @property
    def input_dim(self) -> list[int]:
        return [2, self.output_dim]

    def update(self, mlp_output: torch.Tensor) -> None:
        location, scale_parameter = torch.unbind(mlp_output, dim=-2)
        scale = (torch.nn.functional.softplus(scale_parameter) + self.min_std) * self.var_scale
        self._location = location
        self._scale = scale
        self._normal = Normal(location, scale)
        self._raw_sample = None
        self._last_log_prob = None

    def sample(self) -> torch.Tensor:
        assert self._normal is not None
        raw = self._normal.sample()
        action = torch.tanh(raw)
        self._raw_sample = raw
        self._last_log_prob = self._log_prob_raw(raw)
        return action

    @property
    def raw_sample(self) -> torch.Tensor:
        """Return the pre-tanh sample from the most recent ``sample`` call."""
        if self._raw_sample is None:
            raise RuntimeError("raw_sample is available only immediately after sample()")
        return self._raw_sample

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output[..., 0, :])

    def as_deterministic_output_module(self) -> nn.Module:
        return _TanhLocation()

    @property
    def mean(self) -> torch.Tensor:
        assert self._location is not None
        return torch.tanh(self._location)

    @property
    def std(self) -> torch.Tensor:
        assert self._scale is not None
        return self._scale

    @property
    def entropy(self) -> torch.Tensor:
        if self._last_log_prob is not None:
            return -self._last_log_prob
        assert self._normal is not None
        return self._normal.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        assert self._location is not None and self._scale is not None
        return self._location, self._scale

    def _log_prob_raw(self, raw: torch.Tensor) -> torch.Tensor:
        assert self._normal is not None
        # Stable log(d tanh(x) / dx), identical to Brax's TanhBijector.
        log_det = 2.0 * (math.log(2.0) - raw - torch.nn.functional.softplus(-2.0 * raw))
        return (self._normal.log_prob(raw) - log_det).sum(dim=-1)

    def log_prob_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """Compute the tanh-normal log-probability from a pre-tanh action."""
        return self._log_prob_raw(raw)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        bounded = outputs.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        raw = torch.atanh(bounded)
        # Keep ``_last_log_prob`` tied to the fresh sample produced by
        # ``sample()``. During PPO updates RSL-RL asks for the log-probability
        # of rollout actions before reading entropy; overwriting it here would
        # turn the entropy estimate into ``-log p(old_action)``.
        return self._log_prob_raw(raw)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_location, old_scale = old_params
        new_location, new_scale = new_params
        return torch.distributions.kl_divergence(
            Normal(old_location, old_scale), Normal(new_location, new_scale)
        ).sum(dim=-1)
