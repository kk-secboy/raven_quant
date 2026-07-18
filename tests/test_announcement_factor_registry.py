from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import quant_platform.db_cli as db_cli
from quant_platform import announcement_factor_registry as registry
from quant_platform import announcement_nlp as nlp
from quant_platform.research_store import ResearchStore

NOW = datetime(2026, 7, 18, 12, 0, 0, tzinfo=UTC)
# Lazy engine only: validation must fail before any database round-trip.
DUMMY_URL = "postgresql+psycopg://quantlab:quantlab@127.0.0.1:55433/quantlab_test"


def _fields_frame(*, tone_override: float | None = None) -> pd.DataFrame:
    scores = [0.6, 0.2, -0.4]
    if tone_override is not None:
        scores[0] = tone_override
    return pd.DataFrame(
        {
            "available_at": pd.to_datetime(["2026-07-20", "2026-07-20", "2026-07-21"]),
            "ts_code": ["600519.SH", "600519.SH", "000001.SZ"],
            "tone_score": scores,
        }
    )


def _write_artifact(factors_dir: Path, *, tone_override: float | None = None) -> dict:
    return nlp.write_factor_artifact(
        _fields_frame(tone_override=tone_override),
        factors_dir,
        name=nlp.FACTOR_NAME,
        model="test-model",
        now=NOW,
    )


def _tamper_parquet(artifact: dict, *, tone_override: float) -> None:
    """Rewrite only the values parquet, leaving the manifest sha256 stale."""

    series = nlp.build_tone_factor_series(
        _fields_frame(tone_override=tone_override), nlp.FACTOR_NAME
    )
    series.rename(nlp.FACTOR_NAME).reset_index().to_parquet(
        artifact["artifact_path"], index=False, compression="zstd", engine="pyarrow"
    )


def _unused_store() -> ResearchStore:
    return ResearchStore(DUMMY_URL)


def _candidates(store: ResearchStore) -> list[dict]:
    return [item for item in store.list_candidates(limit=100) if item["name"] == nlp.FACTOR_NAME]


def _import_runs(store: ResearchStore) -> list[dict]:
    return [item for item in store.list_runs(limit=100) if item["kind"] == registry.IMPORT_RUN_KIND]


@pytest.mark.no_database
def test_verified_artifact_round_trips_manifest(tmp_path: Path) -> None:
    factors_dir = tmp_path / "factors"
    artifact = _write_artifact(factors_dir)
    manifest, artifact_path, values_sha256 = registry._verified_artifact(
        factors_dir, nlp.FACTOR_NAME
    )
    assert manifest == artifact["manifest"]
    assert artifact_path == artifact["artifact_path"]
    assert values_sha256 == artifact["manifest"]["sha256"]
    assert values_sha256 == hashlib.sha256(artifact_path.read_bytes()).hexdigest()


@pytest.mark.no_database
def test_code_artifact_source_is_deterministic_and_recomputes_values(
    tmp_path: Path,
) -> None:
    factors_dir = tmp_path / "factors"
    artifact = _write_artifact(factors_dir)
    manifest = artifact["manifest"]
    sha256 = manifest["sha256"]
    first = registry._code_artifact_source(
        factor_name=nlp.FACTOR_NAME, manifest=manifest, values_sha256=sha256
    )
    second = registry._code_artifact_source(
        factor_name=nlp.FACTOR_NAME, manifest=manifest, values_sha256=sha256
    )
    assert first == second
    assert nlp.PROMPT_VERSION in first
    assert "test-model" in first
    assert manifest["availability_policy"][nlp.FACTOR_NAME] in first
    assert sha256 in first
    namespace: dict = {}
    exec(compile(first, "<code-artifact>", "exec"), namespace)
    recomputed = namespace["compute_factor"](_fields_frame())
    expected = nlp.build_tone_factor_series(_fields_frame(), nlp.FACTOR_NAME)
    assert recomputed.equals(expected)


@pytest.mark.no_database
def test_missing_manifest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest is missing"):
        registry.register_announcement_factor(_unused_store(), tmp_path / "factors")


@pytest.mark.no_database
def test_unreadable_manifest_fails_closed(tmp_path: Path) -> None:
    factors_dir = tmp_path / "factors"
    factors_dir.mkdir(parents=True)
    (factors_dir / f"{nlp.FACTOR_NAME}.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest is unreadable"):
        registry.register_announcement_factor(_unused_store(), factors_dir)


@pytest.mark.no_database
def test_manifest_factor_mismatch_fails_closed(tmp_path: Path) -> None:
    factors_dir = tmp_path / "factors"
    artifact = _write_artifact(factors_dir)
    manifest = {**artifact["manifest"], "factor": "other_factor"}
    artifact["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the requested factor"):
        registry.register_announcement_factor(_unused_store(), factors_dir)


@pytest.mark.no_database
def test_manifest_invalid_sha256_fails_closed(tmp_path: Path) -> None:
    factors_dir = tmp_path / "factors"
    artifact = _write_artifact(factors_dir)
    manifest = {**artifact["manifest"], "sha256": "not-a-sha256"}
    artifact["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid sha256"):
        registry.register_announcement_factor(_unused_store(), factors_dir)


@pytest.mark.no_database
def test_manifest_missing_availability_policy_fails_closed(tmp_path: Path) -> None:
    factors_dir = tmp_path / "factors"
    artifact = _write_artifact(factors_dir)
    manifest = {**artifact["manifest"], "availability_policy": {}}
    artifact["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="availability policy"):
        registry.register_announcement_factor(_unused_store(), factors_dir)


@pytest.mark.no_database
def test_manifest_unexpected_source_fails_closed(tmp_path: Path) -> None:
    factors_dir = tmp_path / "factors"
    artifact = _write_artifact(factors_dir)
    manifest = {
        **artifact["manifest"],
        "source": {"dataset": "somewhere_else", "prompt_version": "x", "model": "y"},
    }
    artifact["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source identity"):
        registry.register_announcement_factor(_unused_store(), factors_dir)


@pytest.mark.no_database
def test_missing_values_artifact_fails_closed(tmp_path: Path) -> None:
    factors_dir = tmp_path / "factors"
    artifact = _write_artifact(factors_dir)
    artifact["artifact_path"].unlink()
    with pytest.raises(ValueError, match="artifact is missing"):
        registry.register_announcement_factor(_unused_store(), factors_dir)


@pytest.mark.no_database
def test_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    factors_dir = tmp_path / "factors"
    artifact = _write_artifact(factors_dir)
    _tamper_parquet(artifact, tone_override=-0.9)
    assert artifact["manifest"]["sha256"] != hashlib.sha256(
        artifact["artifact_path"].read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="does not match the manifest sha256"):
        registry.register_announcement_factor(_unused_store(), factors_dir)


def test_register_success(database_url: str, tmp_path: Path) -> None:
    factors_dir = tmp_path / "factors"
    artifact = _write_artifact(factors_dir)
    store = ResearchStore(database_url)

    result = registry.register_announcement_factor(store, factors_dir)

    assert result["created"] is True
    assert result["values_sha256"] == artifact["manifest"]["sha256"]
    candidate = store.get_candidate(result["candidate_id"])
    assert candidate["name"] == nlp.FACTOR_NAME
    assert candidate["status"] == "awaiting_evaluation"
    assert candidate["values_path"] == str(artifact["artifact_path"])
    assert candidate["values_sha256"] == artifact["manifest"]["sha256"]
    code_path = Path(candidate["code_path"])
    assert code_path.is_file()
    assert candidate["code_sha256"] == hashlib.sha256(code_path.read_bytes()).hexdigest()
    assert result["code_sha256"] == candidate["code_sha256"]
    variables = candidate["variables"]
    assert variables["availability_policy"] == dict(nlp.AVAILABILITY_POLICY)
    assert variables["source"]["prompt_version"] == nlp.PROMPT_VERSION
    assert variables["source"]["model"] == "test-model"
    assert variables["rows"] == artifact["manifest"]["rows"]
    policy_text = nlp.AVAILABILITY_POLICY[nlp.FACTOR_NAME]
    assert policy_text in candidate["description"]
    assert "available_at" in candidate["description"]
    run = store.get_run(result["run_id"])
    assert run["kind"] == registry.IMPORT_RUN_KIND
    assert run["status"] == "succeeded"
    assert run["dataset"] == registry.SOURCE_DATASET
    events = {event["event_type"] for event in store.list_events(run["id"])}
    assert {"run.created", "candidate.imported", "run.succeeded"} <= events


def test_register_is_idempotent_for_same_sha256(database_url: str, tmp_path: Path) -> None:
    factors_dir = tmp_path / "factors"
    _write_artifact(factors_dir)
    store = ResearchStore(database_url)

    first = registry.register_announcement_factor(store, factors_dir)
    second = registry.register_announcement_factor(store, factors_dir)

    assert first["created"] is True
    assert second["created"] is False
    assert second["candidate_id"] == first["candidate_id"]
    assert second["run_id"] == first["run_id"]
    assert second["values_sha256"] == first["values_sha256"]
    assert len(_candidates(store)) == 1
    assert len(_import_runs(store)) == 1


def test_register_new_artifact_version_creates_new_candidate(
    database_url: str, tmp_path: Path
) -> None:
    factors_dir = tmp_path / "factors"
    _write_artifact(factors_dir)
    store = ResearchStore(database_url)
    first = registry.register_announcement_factor(store, factors_dir)

    _write_artifact(factors_dir, tone_override=-0.9)
    second = registry.register_announcement_factor(store, factors_dir)

    assert second["created"] is True
    assert second["candidate_id"] != first["candidate_id"]
    assert second["run_id"] != first["run_id"]
    assert second["values_sha256"] != first["values_sha256"]
    assert len(_candidates(store)) == 2
    assert len(_import_runs(store)) == 2


def test_register_checksum_mismatch_writes_nothing(database_url: str, tmp_path: Path) -> None:
    factors_dir = tmp_path / "factors"
    artifact = _write_artifact(factors_dir)
    _tamper_parquet(artifact, tone_override=-0.9)
    store = ResearchStore(database_url)

    with pytest.raises(ValueError, match="does not match the manifest sha256"):
        registry.register_announcement_factor(store, factors_dir)

    assert store.list_candidates(limit=100) == []
    assert store.list_runs(limit=100) == []


def test_cli_registers_factor_idempotently(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    factors_dir = registry.default_factors_dir(data_root)
    _write_artifact(factors_dir)
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    runner = CliRunner()
    first = runner.invoke(db_cli.app, ["register-announcement-factor"])
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    assert payload["created"] is True
    assert payload["factor_name"] == nlp.FACTOR_NAME

    second = runner.invoke(db_cli.app, ["register-announcement-factor"])
    assert second.exit_code == 0, second.output
    repeated = json.loads(second.output)
    assert repeated["created"] is False
    assert repeated["candidate_id"] == payload["candidate_id"]
