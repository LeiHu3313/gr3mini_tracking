"""Train entry point that registers project tasks before delegating to mjlab."""


def main() -> None:
    from mjlab.scripts.train import main as mjlab_train

    import gr3mini_tracking.tasks  # noqa: F401
    from gr3mini_tracking.console_logging import install_grouped_console_logger

    install_grouped_console_logger()
    mjlab_train()
