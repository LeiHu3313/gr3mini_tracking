"""Project-local formatting for RSL-RL's terminal training logs."""

from __future__ import annotations

import io
import re
from collections import defaultdict
from contextlib import redirect_stdout
from typing import Any

from rsl_rl.utils.logger import Logger

_EXTRA_PREFIXES = ("Episode_Reward", "Episode_Termination", "Metrics")
_EXTRA_LINE = re.compile(r"^\s*((?:Episode_Reward|Episode_Termination|Metrics)/[^:]+):\s+.*$")


def group_episode_extras(console_text: str) -> str:
    """Group RSL-RL episode extras without changing their values or TensorBoard tags."""
    grouped: dict[str, list[str]] = defaultdict(list)
    remaining: list[str] = []
    first_extra_index: int | None = None

    for line in console_text.splitlines(keepends=True):
        match = _EXTRA_LINE.match(line)
        if match is None:
            remaining.append(line)
            continue

        if first_extra_index is None:
            first_extra_index = len(remaining)
        key = match.group(1)
        grouped[key.partition("/")[0]].append(line)

    if first_extra_index is None:
        return console_text

    sections: list[str] = []
    for prefix in _EXTRA_PREFIXES:
        lines = grouped.get(prefix, [])
        if lines:
            sections.append(f"\n[{prefix}]\n")
            sections.extend(sorted(lines, key=str.strip))
    remaining[first_extra_index:first_extra_index] = sections
    return "".join(remaining)


class GroupedConsoleLogger(Logger):
    """Keep RSL-RL logging intact while making terminal episode extras readable."""

    def log(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("print_minimal", False):
            super().log(*args, **kwargs)
            return

        captured = io.StringIO()
        with redirect_stdout(captured):
            super().log(*args, **kwargs)
        print(group_episode_extras(captured.getvalue()), end="")


def install_grouped_console_logger() -> None:
    """Install the project logger before an RSL-RL runner is constructed."""
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.Logger = GroupedConsoleLogger
