"""Runner enforcing explicit teacher initialization for adapter training."""

from __future__ import annotations

from pathlib import Path

from mjlab.rl.runner import MjlabOnPolicyRunner


class AdapterOnPolicyRunner(MjlabOnPolicyRunner):
    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
    ) -> None:
        del registry_name
        teacher = train_cfg.get("teacher_checkpoint", "")
        if log_dir is not None:
            if not teacher:
                raise ValueError(
                    "Adapter training requires --agent.teacher-checkpoint /path/to/model.pt"
                )
            if not Path(teacher).is_file():
                raise FileNotFoundError(f"teacher checkpoint not found: {teacher}")
        super().__init__(env, train_cfg, log_dir, device)
