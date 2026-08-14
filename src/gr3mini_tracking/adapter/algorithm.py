"""PPO plus the source adapter's independently optimized world-model objective."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_obs_groups
from tensordict import TensorDict
from torch import nn

from .models import (
    AdapterCritic,
    HistoryEncoder,
    ResidualAdapterActor,
    WorldModel,
    world_model_loss,
)


class AdapterPPO(PPO):
    """PPO for adapter/critic plus autoregressive world-model training."""

    def __init__(
        self,
        actor: ResidualAdapterActor,
        critic: AdapterCritic,
        storage: RolloutStorage,
        history_encoder: HistoryEncoder,
        world_model: WorldModel,
        world_model_learning_rate: float = 1.0e-4,
        world_model_num_mini_batches: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.history_encoder = history_encoder.to(self.device)
        self.world_model = world_model.to(self.device)
        self.world_model_num_mini_batches = world_model_num_mini_batches
        self.world_optimizer = torch.optim.Adam(
            list(self.history_encoder.parameters()) + list(self.world_model.parameters()),
            lr=world_model_learning_rate,
        )
        self._next_world = torch.zeros(
            storage.num_transitions_per_env,
            storage.num_envs,
            57,
            device=self.device,
        )

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        self._next_world[self.storage.step].copy_(obs["world"])
        super().process_env_step(obs, rewards, dones, extras)

    def _update_world_model(self) -> float:
        observations = self.storage.observations
        history = observations["history"]
        world = observations["world"]
        actions = self.storage.actions
        dones = self.storage.dones.bool()
        env_order = torch.randperm(self.storage.num_envs, device=self.device)
        chunk_count = min(self.world_model_num_mini_batches, self.storage.num_envs)
        chunks = torch.chunk(env_order, chunk_count)
        total_loss = 0.0
        updates = 0
        for env_ids in chunks:
            state = world[0, env_ids]
            history_state = history[0, env_ids]
            loss = torch.zeros((), device=self.device)
            for step in range(self.storage.num_transitions_per_env):
                embedding = self.history_encoder(history_state)
                prediction = self.world_model(state, embedding, actions[step, env_ids])
                reset = dones[step, env_ids]
                target = self._next_world[step, env_ids]
                # Auto-reset observations belong to a new episode. Use them to restart
                # autoregression, but do not train a discontinuous terminal transition.
                loss_target = torch.where(reset, prediction.detach(), target)
                loss = loss + world_model_loss(prediction, loss_target)
                state = torch.where(reset, target, prediction)
                if step + 1 < self.storage.num_transitions_per_env:
                    # Preserve the source implementation's autoregressive history
                    # update: shift 56 values and append predicted world state without
                    # root height. A reset restores the actual next history.
                    predicted_history = torch.cat(
                        [history_state[..., 56:], prediction[..., :-1]], dim=-1
                    )
                    actual_next_history = history[step + 1, env_ids]
                    history_state = torch.where(reset, actual_next_history, predicted_history)
            loss = loss / self.storage.num_transitions_per_env
            self.world_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.history_encoder.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.world_model.parameters(), self.max_grad_norm)
            self.world_optimizer.step()
            total_loss += loss.item()
            updates += 1
        return total_loss / max(updates, 1)

    def update(self) -> dict[str, float]:
        world_loss = self._update_world_model()
        losses = super().update()
        losses["world_model"] = world_loss
        return losses

    def train_mode(self) -> None:
        super().train_mode()
        self.history_encoder.train()
        self.world_model.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.history_encoder.eval()
        self.world_model.eval()

    def save(self) -> dict:
        payload = super().save()
        payload.update(
            {
                "history_encoder_state_dict": self.history_encoder.state_dict(),
                "world_model_state_dict": self.world_model.state_dict(),
                "world_optimizer_state_dict": self.world_optimizer.state_dict(),
                "adapter_contract": {
                    "history_length": 79,
                    "frame_dim": 81,
                    "embedding_dim": 128,
                    "world_state_dim": 57,
                },
            }
        )
        return payload

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        result = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("world_model", True):
            self.history_encoder.load_state_dict(
                loaded_dict["history_encoder_state_dict"], strict=strict
            )
            self.world_model.load_state_dict(loaded_dict["world_model_state_dict"], strict=strict)
            if load_cfg is None or load_cfg.get("optimizer", True):
                self.world_optimizer.load_state_dict(loaded_dict["world_optimizer_state_dict"])
        return result

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> AdapterPPO:
        teacher_checkpoint = cfg.pop("teacher_checkpoint", "")
        world_lr = cfg.pop("world_model_learning_rate", 1.0e-4)
        world_batches = cfg.pop("world_model_num_mini_batches", 32)
        cfg["algorithm"].pop("share_cnn_encoders", None)
        cfg["algorithm"].pop("class_name", None)
        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["algorithm"].setdefault("symmetry_cfg", None)
        cfg["actor"].pop("class_name", None)
        cfg["critic"].pop("class_name", None)
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        if cfg.get("multi_gpu") is not None:
            raise NotImplementedError("Adapter world-model updates currently support one GPU")

        history_encoder = HistoryEncoder().to(device)
        actor = ResidualAdapterActor(
            obs,
            cfg["obs_groups"],
            "actor",
            env.num_actions,
            history_encoder=history_encoder,
            **cfg["actor"],
        ).to(device)
        if teacher_checkpoint:
            actor.load_teacher(teacher_checkpoint)
        critic = AdapterCritic(
            obs,
            cfg["obs_groups"],
            "critic",
            1,
            history_encoder=history_encoder,
            **cfg["critic"],
        ).to(device)
        print(f"Adapter Actor Model: {actor}")
        print(f"Adapter Critic Model: {critic}")
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        algorithm = AdapterPPO(
            actor,
            critic,
            storage,
            history_encoder=history_encoder,
            world_model=WorldModel(action_dim=env.num_actions),
            world_model_learning_rate=world_lr,
            world_model_num_mini_batches=world_batches,
            device=device,
            multi_gpu_cfg=cfg["multi_gpu"],
            **cfg["algorithm"],
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm
