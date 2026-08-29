from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from quant_platform import control_plane_lock as lock_module

pytestmark = pytest.mark.no_database


def test_control_plane_lock_is_reentrant_but_rejects_another_thread(
    tmp_path: Path,
) -> None:
    target = tmp_path / "quantlab-control-plane.lock"

    def contend() -> str:
        with lock_module.control_plane_lock(target):
            return "unexpected"

    with lock_module.control_plane_lock(target):
        with lock_module.control_plane_lock(target):
            assert target.is_file()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(contend)
            with pytest.raises(
                lock_module.ControlPlaneBusyError,
                match="another control-plane operation owns",
            ):
                future.result()

    with lock_module.control_plane_lock(target):
        assert target.is_file()


def test_control_plane_decorator_uses_the_same_reentrant_host_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "decorated.lock"
    monkeypatch.setattr(
        lock_module,
        "default_control_plane_lock_path",
        lambda: target,
    )

    @lock_module.control_plane_locked
    def inner() -> str:
        return "done"

    with lock_module.control_plane_lock(target):
        assert inner() == "done"
