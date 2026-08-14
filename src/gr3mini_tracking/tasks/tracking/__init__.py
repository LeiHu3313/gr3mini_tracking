"""Register only the requested DiffCritic teacher and its adapter."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from gr3mini_tracking import ADAPTER_TASK, TEACHER_TASK
from gr3mini_tracking.adapter.runner import AdapterOnPolicyRunner

from .env_cfg import gr3mini_diff_critic_env_cfg
from .rl_cfg import adapter_runner_cfg, teacher_runner_cfg

register_mjlab_task(
    task_id=TEACHER_TASK,
    env_cfg=gr3mini_diff_critic_env_cfg(adapter=False),
    play_env_cfg=gr3mini_diff_critic_env_cfg(adapter=False, play=True),
    rl_cfg=teacher_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id=ADAPTER_TASK,
    env_cfg=gr3mini_diff_critic_env_cfg(adapter=True),
    play_env_cfg=gr3mini_diff_critic_env_cfg(adapter=True, play=True),
    rl_cfg=adapter_runner_cfg(),
    runner_cls=AdapterOnPolicyRunner,
)
