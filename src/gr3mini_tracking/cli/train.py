"""Train entry point that registers project tasks before delegating to mjlab."""


def main() -> None:
    from mjlab.scripts.train import main as mjlab_train

    import gr3mini_tracking.tasks  # noqa: F401

    mjlab_train()
