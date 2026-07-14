from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from quant_platform.api import create_app
from quant_platform.platform_config_store import PlatformConfigStore


def test_platform_config_store_is_versioned_and_preserves_history(database_url: str) -> None:
    store = PlatformConfigStore(database_url)
    assert store.get("multifactor_strategy_defaults") is None

    first = store.put(
        "multifactor_strategy_defaults",
        {"stop_loss": 0.07, "take_profit": 0.20},
        actor="risk-admin",
        reason="Establish the documented default risk template.",
    )
    second = store.put(
        "multifactor_strategy_defaults",
        {"stop_loss": 0.08, "take_profit": 0.22},
        actor="risk-admin",
        reason="Use a separate reviewed template revision for new strategies.",
    )

    assert first["revision"] == 1
    assert second["revision"] == 2
    assert second["value"]["stop_loss"] == 0.08
    history = store.list_revisions("multifactor_strategy_defaults")
    assert [item["revision"] for item in history] == [2, 1]
    assert history[1]["value"]["stop_loss"] == 0.07
    with pytest.raises(ValueError, match="at least 10"):
        store.put(
            "multifactor_strategy_defaults",
            {"stop_loss": 0.09},
            actor="risk-admin",
            reason="short",
        )


def test_strategy_defaults_api_validates_and_versions_web_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    database_url: str,
) -> None:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("RUN_EMBEDDED_WORKER", "false")
    monkeypatch.setenv("AUTH_MODE", "disabled")
    with TestClient(create_app(tmp_path)) as client:
        initial = client.get("/api/settings/strategy-defaults")
        assert initial.status_code == 200
        assert initial.json()["source"] == "built_in"
        assert initial.json()["config"]["stop_loss"] == 0.07
        assert initial.json()["config"]["max_volume_participation"] == 0.01
        assert initial.json()["config"]["max_industry_weight"] == 0.30

        config = initial.json()["config"]
        config["stop_loss"] = 0.08
        config["take_profit_partial"] = 0.13
        config["take_profit"] = 0.21
        saved = client.put(
            "/api/settings/strategy-defaults",
            json={
                "config": config,
                "reason": "Adjust defaults for newly created strategy versions only.",
            },
        )
        assert saved.status_code == 200
        assert saved.json()["source"] == "database"
        assert saved.json()["revision"] == 1
        assert saved.json()["config"]["stop_loss"] == 0.08

        invalid = dict(config)
        invalid["take_profit_partial"] = invalid["take_profit"]
        rejected = client.put(
            "/api/settings/strategy-defaults",
            json={
                "config": invalid,
                "reason": "This invalid threshold order must be rejected.",
            },
        )
        assert rejected.status_code == 422
        revisions = client.get("/api/settings/strategy-defaults/revisions").json()
        assert len(revisions) == 1
        assert revisions[0]["value"]["stop_loss"] == 0.08
