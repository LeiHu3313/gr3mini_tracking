"""Play entry point that registers project tasks before delegating to mjlab."""


def main() -> None:
    from mjlab.scripts.play import main as mjlab_play

    import gr3mini_tracking.tasks  # noqa: F401

    mjlab_play()
