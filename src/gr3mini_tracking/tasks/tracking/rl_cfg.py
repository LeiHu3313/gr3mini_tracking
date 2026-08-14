"""RSL-RL configurations mapped from the source Brax PPO settings."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

HIDDEN_DIMS = (512, 512, 256, 256, 128)
TANH_NORMAL = {
    "class_name": "gr3mini_tracking.adapter.distribution:TanhGaussianDistribution",
    "min_std": 0.001,
    "var_scale": 1.0,
}


@dataclass
class AdapterRunnerCfg(RslRlOnPolicyRunnerCfg):
    teacher_checkpoint: str = ""
    world_model_learning_rate: float = 1.0e-4
    world_model_num_mini_batches: int = 32


def _algorithm(learning_rate: float, class_name: str = "PPO") -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        class_name=class_name,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=4,
        num_mini_batches=32,
        learning_rate=learning_rate,
        schedule="fixed",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )


def teacher_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=0,
        actor=RslRlModelCfg(
            hidden_dims=HIDDEN_DIMS,
            activation="swish",
            obs_normalization=False,
            distribution_cfg=dict(TANH_NORMAL),
        ),
        critic=RslRlModelCfg(
            hidden_dims=HIDDEN_DIMS,
            activation="swish",
            obs_normalization=False,
        ),
        algorithm=_algorithm(3.0e-4),
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        experiment_name="gr3mini_diffcritic_teacher",
        run_name="teacher",
        logger="tensorboard",
        upload_model=False,
        clip_actions=None,
        save_interval=500,
        num_steps_per_env=20,
        max_iterations=36_622,
    )


def adapter_runner_cfg() -> AdapterRunnerCfg:
    return AdapterRunnerCfg(
        seed=0,
        class_name="AdapterOnPolicyRunner",
        actor=RslRlModelCfg(
            class_name="gr3mini_tracking.adapter.models:ResidualAdapterActor",
            hidden_dims=HIDDEN_DIMS,
            activation="swish",
            obs_normalization=False,
            distribution_cfg=dict(TANH_NORMAL),
        ),
        critic=RslRlModelCfg(
            class_name="gr3mini_tracking.adapter.models:AdapterCritic",
            hidden_dims=HIDDEN_DIMS,
            activation="swish",
            obs_normalization=False,
        ),
        algorithm=_algorithm(1.0e-4, "gr3mini_tracking.adapter.algorithm:AdapterPPO"),
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        experiment_name="gr3mini_diffcritic_adapter",
        run_name="adapter",
        logger="tensorboard",
        upload_model=False,
        clip_actions=None,
        save_interval=500,
        num_steps_per_env=20,
        max_iterations=36_622,
    )
