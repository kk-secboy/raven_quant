"""Execute the migrated runtime CHECKs in a disposable PostgreSQL database.

The temporary probe copies the real columns and CHECK definitions, not the full
StrategyVersion foreign keys or other governance constraints. Full admission
remains covered by the strategy-store tests. This file is intentionally not
marked no_database: the repository fixture requires an isolated test database.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import pytest
from alembic import command
from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError

from quant_data.database import open_database
from quant_platform import transparent_baseline_runner as runtime
from quant_platform.db_cli import alembic_config

V39_CHECK = "ck_strategy_versions_v39_runtime_identity"
V40_CHECK = "ck_strategy_versions_v40_runtime_identity"
HEAD = "0112_strategy_runtime_v40"
RECIPES = ("short_relative_strength", "swing_trend", "long_quality_value")


def _checks(connection: Connection) -> dict[str, str]:
    return dict(connection.execute(text(
        "SELECT conname, pg_get_constraintdef(oid, false) FROM pg_constraint "
        "WHERE conrelid = 'quantlab.strategy_versions'::regclass AND contype = 'c'"
    )).all())


def _head(connection: Connection) -> str:
    return connection.execute(text(
        "SELECT version_num FROM quantlab.alembic_version"
    )).scalar_one()


@pytest.fixture
def runtime_probe(database_url: str) -> Iterator[Connection]:
    engine = open_database(database_url)
    try:
        with engine.begin() as connection:
            definitions = _checks(connection)
            assert {V39_CHECK, V40_CHECK} <= definitions.keys()
            connection.execute(text(
                "CREATE TEMPORARY TABLE v40_runtime_check_probe ON COMMIT DROP AS "
                "SELECT config_json, evidence_mode FROM quantlab.strategy_versions WITH NO DATA"
            ))
            for name in (V39_CHECK, V40_CHECK):
                # Both names are constants; each expression comes from this test DB's catalog.
                connection.execute(text(
                    f"ALTER TABLE pg_temp.v40_runtime_check_probe "
                    f"ADD CONSTRAINT {name} {definitions[name]}"
                ))
            yield connection
    finally:
        engine.dispose()


def _config(recipe_id: str, *, historical: bool = False) -> dict[str, Any]:
    prefix = "STRATEGY_RESEARCH_V39_TARGET_" if historical else "STRATEGY_RESEARCH_TARGET_"
    return {
        "recipe_id": recipe_id,
        "recipe_version": getattr(runtime, prefix + "RECIPE_VERSION"),
        "transparent_baseline_bootstrap": {
            "target_runner_sha256": getattr(runtime, prefix + "RUNNER_SHA256"),
            "target_runtime_bundle_sha256": getattr(runtime, prefix + "RUNTIME_BUNDLE_SHA256"),
            "target_worker_runtime_image_digest": "sha256:" + "1" * 64,
        },
    }


def _insert(connection: Connection, config: dict[str, Any], mode: str | None) -> None:
    connection.execute(text(
        "INSERT INTO pg_temp.v40_runtime_check_probe (config_json, evidence_mode) "
        "VALUES (CAST(:config AS jsonb), :mode)"
    ), {"config": json.dumps(config), "mode": mode})


def test_migrated_checks_admit_all_current_and_historical_recipes(
    runtime_probe: Connection,
) -> None:
    for recipe_id in RECIPES:
        for historical in (False, True):
            _insert(runtime_probe, _config(recipe_id, historical=historical), "sealed_final_oos")
    assert runtime_probe.execute(text(
        "SELECT count(*) FROM pg_temp.v40_runtime_check_probe"
    )).scalar_one() == 6
    assert runtime_probe.execute(text(
        "SELECT count(*) FROM pg_temp.v40_runtime_check_probe "
        "WHERE config_json ->> 'recipe_version' = :version "
        "AND config_json -> 'transparent_baseline_bootstrap' ->> "
        "'target_runtime_bundle_sha256' = :bundle"
    ), {
        "version": runtime.STRATEGY_RESEARCH_V39_TARGET_RECIPE_VERSION,
        "bundle": runtime.STRATEGY_RESEARCH_V39_TARGET_RUNTIME_BUNDLE_SHA256,
    }).scalar_one() == 3


def test_migrated_v40_check_rejects_incomplete_or_mismatched_identity(
    runtime_probe: Connection,
) -> None:
    for recipe_id in RECIPES:
        valid = _config(recipe_id)
        cases: list[tuple[str, dict[str, Any], str | None]] = []
        for field in ("recipe_id", "transparent_baseline_bootstrap"):
            missing = deepcopy(valid)
            missing.pop(field)
            cases.append((f"missing {field}", missing, "sealed_final_oos"))
            cases.append((f"null {field}", {**deepcopy(valid), field: None}, "sealed_final_oos"))
        cases.append((
            "unapproved recipe", {**valid, "recipe_id": "unapproved"}, "sealed_final_oos",
        ))
        for field in valid["transparent_baseline_bootstrap"]:
            for replacement in ("missing", None, "invalid"):
                invalid = deepcopy(valid)
                if replacement == "missing":
                    invalid["transparent_baseline_bootstrap"].pop(field)
                else:
                    invalid["transparent_baseline_bootstrap"][field] = replacement
                cases.append((f"{field}: {replacement}", invalid, "sealed_final_oos"))
        old_bundle = deepcopy(valid)
        old_bundle["transparent_baseline_bootstrap"]["target_runtime_bundle_sha256"] = (
            runtime.STRATEGY_RESEARCH_V39_TARGET_RUNTIME_BUNDLE_SHA256
        )
        cases.append(("old v39 bundle", old_bundle, "sealed_final_oos"))
        unaccepted_candidate = deepcopy(valid)
        unaccepted_candidate["transparent_baseline_bootstrap"].update({
            "target_runner_sha256":
                "48f241a9d03f63a87a54413f77b37577443a49285a28774716a4567382545c45",
            "target_runtime_bundle_sha256":
                "43951d567fa610b8deed4ba4fec31716b0c7a1c7b9537ad37e5a9087ff1a4ec7",
        })
        cases.append(("unaccepted v40 candidate", unaccepted_candidate, "sealed_final_oos"))
        factor_candidate = deepcopy(valid)
        factor_candidate["transparent_baseline_bootstrap"].update({
            "target_runner_sha256":
                "59f2552f6c8eb5b13600a4785aa019ef9cfe045798114533b792a2731bb3d43a",
            "target_runtime_bundle_sha256":
                "00b2b471dea7783256b88dcd370ad29a1b48815183c495684c8534e61bfe2689",
        })
        cases.append(("unaccepted c8353aa candidate", factor_candidate, "sealed_final_oos"))
        output_candidate = deepcopy(valid)
        output_candidate["transparent_baseline_bootstrap"].update({
            "target_runner_sha256":
                "59f2552f6c8eb5b13600a4785aa019ef9cfe045798114533b792a2731bb3d43a",
            "target_runtime_bundle_sha256":
                "7114f04a1e6b45188fbd4ba870ec546a3d612aab7b81561b596161adc59ccd23",
        })
        cases.append(("unaccepted 6888293 candidate", output_candidate, "sealed_final_oos"))
        wrong_runner = deepcopy(valid)
        wrong_runner["transparent_baseline_bootstrap"]["target_runner_sha256"] = "0" * 64
        cases.append(("wrong SHA256 runner", wrong_runner, "sealed_final_oos"))
        for invalid_digest in ("sha256:" + "A" * 64, "sha256:" + "1" * 63, "1" * 64):
            invalid = deepcopy(valid)
            invalid["transparent_baseline_bootstrap"]["target_worker_runtime_image_digest"] = (
                invalid_digest
            )
            cases.append(("malformed worker digest", invalid, "sealed_final_oos"))
        for mode in (None, "pre_registered_replay", "research_only"):
            cases.append((f"evidence mode: {mode}", valid, mode))
        for label, config, mode in cases:
            with pytest.raises(IntegrityError) as rejected, runtime_probe.begin_nested():
                _insert(runtime_probe, config, mode)
            assert rejected.value.orig.sqlstate == "23514", (recipe_id, label)
            assert rejected.value.orig.diag.constraint_name == V40_CHECK, (recipe_id, label)
    assert runtime_probe.execute(text(
        "SELECT count(*) FROM pg_temp.v40_runtime_check_probe"
    )).scalar_one() == 0


def test_v40_real_migration_roundtrip_preserves_all_historical_checks(
    database_url: str,
) -> None:
    engine = open_database(database_url)
    config = alembic_config(database_url)
    try:
        with engine.connect() as connection:
            assert _head(connection) == HEAD
            before = _checks(connection)
            assert {V39_CHECK, V40_CHECK} <= before.keys()
        command.downgrade(config, "0111_autopilot_research_events")
        with engine.connect() as connection:
            assert _head(connection) == "0111_autopilot_research_events"
            after_downgrade = _checks(connection)
            assert after_downgrade == {
                name: definition for name, definition in before.items() if name != V40_CHECK
            }
            assert after_downgrade[V39_CHECK].encode() == before[V39_CHECK].encode()
        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert _head(connection) == HEAD
            assert _checks(connection) == before
    finally:
        try:
            command.upgrade(config, "head")
        finally:
            engine.dispose()
