"""Per-kind subprocess command builders for LocalJobWorker.

LocalJobWorker keeps job claiming, process monitoring, settlement and
scheduling; the command assembly that used to be one long if/elif chain in
worker.py lives in this package, split by domain.  ``build_command`` is a
pure dispatch: behavior for every kind is byte-identical to the old chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import data, evaluation, ops, research, simulation

_COMMAND_BUILDERS = {
    **data.COMMANDS,
    **evaluation.COMMANDS,
    **simulation.COMMANDS,
    **research.COMMANDS,
    **ops.COMMANDS,
}

KNOWN_JOB_COMMAND_KINDS = frozenset(_COMMAND_BUILDERS)


def build_command(
    worker: Any, job: dict
) -> tuple[list[str], Path | None, dict[str, str]]:
    kind = str(job["kind"])
    builder = _COMMAND_BUILDERS.get(kind)
    if builder is None and kind.startswith(data.SUPPLEMENTAL_PREFIX):
        builder = data.intraday_download_command
    if builder is None:
        raise ValueError(f"unsupported job kind: {kind}")
    return builder(worker, job)
