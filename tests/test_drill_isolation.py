from __future__ import annotations

import os

import pytest

from quant_platform.drill_isolation import (
    DRILL_COMPOSE_ENVIRONMENT_KEYS,
    isolated_drill_environment,
)

pytestmark = pytest.mark.no_database


def test_drill_environment_removes_inherited_compose_values_and_restores_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QUANTLAB_DATA_HOST_PATH", "/production/data")
    monkeypatch.setenv("RDAGENT_DOCKER_HOST_PATH", "/production/dind")
    monkeypatch.delenv("MODEL_SANDBOX_IMAGE", raising=False)

    @isolated_drill_environment
    def probe() -> None:
        assert all(name not in os.environ for name in DRILL_COMPOSE_ENVIRONMENT_KEYS)
        os.environ["MODEL_SANDBOX_IMAGE"] = "unexpected-drill-value"

    probe()

    assert os.environ["QUANTLAB_DATA_HOST_PATH"] == "/production/data"
    assert os.environ["RDAGENT_DOCKER_HOST_PATH"] == "/production/dind"
    assert "MODEL_SANDBOX_IMAGE" not in os.environ


def test_drill_environment_restores_values_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RDAGENT_REGISTRY_PORT", "55000")

    @isolated_drill_environment
    def fail() -> None:
        assert "RDAGENT_REGISTRY_PORT" not in os.environ
        raise RuntimeError("drill failed")

    with pytest.raises(RuntimeError, match="drill failed"):
        fail()

    assert os.environ["RDAGENT_REGISTRY_PORT"] == "55000"
