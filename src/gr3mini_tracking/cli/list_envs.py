"""List entry point including externally registered tasks."""


def main() -> None:
    from mjlab.scripts.list_envs import main as mjlab_list_envs

    import gr3mini_tracking.tasks  # noqa: F401

    mjlab_list_envs()
