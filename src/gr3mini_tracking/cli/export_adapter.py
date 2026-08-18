"""Export a trained adapter checkpoint to ONNX for the mini_deploy_any2track_task.

Usage:
    gr3mini-export-adapter <checkpoint.pt> <output_dir>

Writes:
    <output_dir>/policy.onnx   - ONNX model (obs:[1,686], history:[1,81,79] -> continuous_actions:[1,25])
    <output_dir>/metadata.json - provenance record
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tensordict import TensorDict
from torch import nn

from gr3mini_tracking.adapter.models import ADAPTER_EMBEDDING_DIM, HistoryEncoder, ResidualAdapterActor
from gr3mini_tracking.tasks.tracking.observations import (
    ADAPTER_HISTORY_LENGTH,
    ACTOR_OBS_DIM,
    FRAME_DIM,
)

HIDDEN_DIMS = (512, 512, 256, 256, 128)
POLICY_JOINT_NUM = 25
DISTRIBUTION_CFG = {
    "class_name": "gr3mini_tracking.adapter.distribution:TanhGaussianDistribution",
    "min_std": 0.001,
    "max_std": 1.0,
    "var_scale": 1.0,
}


class _DeployWrapper(nn.Module):
    """ONNX-exportable adapter wrapper.

    Takes (obs [B, 686], history [B, 81, 79] channel-first) and returns
    deterministic actions [B, 25] in [-1, 1].

    The history input is channel-first to match the C++ flattenHistoryChannelFirst()
    layout, so we bypass HistoryEncoder.forward (which expects time-first) and
    feed directly into the Conv1d layers.
    """

    def __init__(self, actor: ResidualAdapterActor) -> None:
        super().__init__()
        encoder = actor.history_encoder
        self.hist_conv = encoder.conv          # nn.Sequential: Conv1d → SiLU → Conv1d → SiLU
        self.hist_proj = encoder.projection    # Linear(256, 128)
        self.hist_act = encoder.activation     # SiLU
        self.mlp = actor.mlp                   # nn.Sequential including Unflatten at end
        self.adapter_layers = actor.adapter_layers  # nn.ModuleList of 6 Linear layers
        self.obs_normalizer = actor.obs_normalizer  # nn.Identity (obs_normalization=False)

    def forward(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        # history: [B, FRAME_DIM=81, ADAPTER_HISTORY_LENGTH=79] channel-first
        # feed directly to Conv1d, skipping HistoryEncoder's internal transpose
        embedding = self.hist_act(
            self.hist_proj(self.hist_conv(history).flatten(start_dim=1))
        )  # [B, 128]

        value = self.obs_normalizer(obs)  # [B, 686], identity
        adapter_index = 0
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                adapter_input = embedding if adapter_index == 0 else value
                value = layer(value) + self.adapter_layers[adapter_index](adapter_input)
                adapter_index += 1
            else:
                value = layer(value)  # SiLU activations and final Unflatten([2,25])

        # value: [B, 2, 25] (after Unflatten); [0] = loc, [1] = log_std
        return torch.tanh(value[..., 0, :])  # [B, 25]


def _build_actor(checkpoint: dict) -> ResidualAdapterActor:
    history_encoder = HistoryEncoder()
    fake_obs = TensorDict({"actor": torch.zeros(1, ACTOR_OBS_DIM)}, batch_size=[1])
    actor = ResidualAdapterActor(
        obs=fake_obs,
        obs_groups={"actor": ("actor",)},
        obs_set="actor",
        output_dim=POLICY_JOINT_NUM,
        history_encoder=history_encoder,
        hidden_dims=HIDDEN_DIMS,
        activation="swish",
        obs_normalization=False,
        distribution_cfg=dict(DISTRIBUTION_CFG),
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    history_encoder.load_state_dict(checkpoint["history_encoder_state_dict"])
    return actor


def export(checkpoint_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = _build_actor(payload)
    actor.eval()

    wrapper = _DeployWrapper(actor)
    wrapper.eval()

    dummy_obs = torch.zeros(1, ACTOR_OBS_DIM)
    dummy_history = torch.zeros(1, FRAME_DIM, ADAPTER_HISTORY_LENGTH)

    onnx_path = output_dir / "policy.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_history),
        str(onnx_path),
        export_params=True,
        opset_version=18,
        input_names=["obs", "history"],
        output_names=["continuous_actions"],
        dynamic_axes={
            "obs": {0: "batch"},
            "history": {0: "batch"},
            "continuous_actions": {0: "batch"},
        },
    )

    metadata = {
        "iteration": int(payload.get("iter", 0)),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "actor_obs_dim": ACTOR_OBS_DIM,
        "history_frame_dim": FRAME_DIM,
        "history_length": ADAPTER_HISTORY_LENGTH,
        "policy_joint_num": POLICY_JOINT_NUM,
        "adapter_embedding_dim": ADAPTER_EMBEDDING_DIM,
        "onnx_inputs": {
            "obs": f"[1, {ACTOR_OBS_DIM}]",
            "history": f"[1, {FRAME_DIM}, {ADAPTER_HISTORY_LENGTH}]",
        },
        "onnx_output": f"continuous_actions: [1, {POLICY_JOINT_NUM}]",
        "observation_mode": "raw_critic_686",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"exported: {onnx_path}")
    print(f"  obs: [1, {ACTOR_OBS_DIM}]  history: [1, {FRAME_DIM}, {ADAPTER_HISTORY_LENGTH}]  "
          f"-> continuous_actions: [1, {POLICY_JOINT_NUM}]")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="path to model_XXXX.pt")
    parser.add_argument("output_dir", type=Path, help="directory to write policy.onnx + metadata.json")
    args = parser.parse_args()
    export(args.checkpoint.expanduser().resolve(), args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
