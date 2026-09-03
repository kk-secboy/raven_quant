from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform.cost_model import COST_SCHEDULE_VERSION
from quant_platform.feature_set_registry import get_feature_set
from quant_platform.rdagent_dataset_view import isolate_rdagent_periods
from quant_platform.strategy_proposal import STRATEGY_PROPOSAL_VERSION
from quant_platform.strategy_recipes import get_strategy_recipe
from quant_platform.strategy_rule_compiler import compile_strategy_proposal
from quant_platform.strategy_store import _compiled_fin_strategy_materialization_binding
from quant_platform.worker import Worker

pytestmark = pytest.mark.no_database


class _ArtifactRecorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_run_artifact(self, **kwargs):
        path = Path(kwargs["storage_path"])
        assert path.is_file()
        row = {
            "id": f"artifact-{len(self.calls) + 1}",
            "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        self.calls.append({**kwargs, "row": row})
        return row


class _StrategyRecorder:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def find_version_by_source_artifact(self, source_research_artifact_id: str):
        return None

    def find_version_by_compiled_fin_strategy_artifact(self, artifact: dict):
        return None

    def get_version(self, version_id: str) -> dict:
        assert version_id == "a" * 32
        return {
            "id": version_id,
            "strategy_id": "parent-family",
            "horizon_profile": "short_1_5d",
            "universe": "cn_all",
        }

    def create_version(self, strategy_id: str, **kwargs) -> dict:
        assert strategy_id == "parent-family"
        self.created.append(kwargs)
        return {
            "id": "candidate-version",
            "strategy_id": strategy_id,
            "status": "draft",
            "horizon_profile": "short_1_5d",
            "strategy_rules_sha256": kwargs["config"]["strategy_rules_sha256"],
            "config": kwargs["config"],
        }

    def create_fin_strategy_version_if_absent(
        self, strategy_id: str, **kwargs
    ) -> dict:
        return self.create_version(strategy_id, **kwargs)


class _ColdstartStrategyRecorder:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self._versions_by_source_artifact: dict[str, dict] = {}

    def find_version_by_source_artifact(self, source_research_artifact_id: str):
        return self._versions_by_source_artifact.get(source_research_artifact_id)

    def find_version_by_compiled_fin_strategy_artifact(self, artifact: dict):
        artifact_sha256 = artifact["artifact_sha256"]
        for version in self._versions_by_source_artifact.values():
            if (
                version["config"].get("strategy_research_artifact_sha256")
                == artifact_sha256
            ):
                return version
        return None

    def create(self, **kwargs) -> dict:
        self.created.append(kwargs)
        config = kwargs["config"]
        version = {
            "id": "coldstart-candidate-version",
            "strategy_id": "coldstart-family",
            "status": "draft",
            "horizon_profile": "short_1_5d",
            "strategy_rules_sha256": config["strategy_rules_sha256"],
            "config": config,
        }
        self._versions_by_source_artifact[config["source_research_artifact_id"]] = version
        return {
            "id": "coldstart-family",
            "versions": [version],
        }

    def create_version(self, strategy_id: str, **kwargs) -> dict:
        raise AssertionError(
            f"cold-start materialization must create a strategy family, got {strategy_id}"
        )


def _compiled_short_artifact(
    monkeypatch: pytest.MonkeyPatch,
    *,
    feature_set: dict,
    periods: dict[str, str],
    dataset_identity_sha256: str,
    incumbent_id: str | None,
) -> dict:
    isolated = isolate_rdagent_periods(periods)
    monkeypatch.setenv("QUANTLAB_DATASET_SNAPSHOT_ID", dataset_identity_sha256)
    monkeypatch.setenv("QUANTLAB_FEATURE_SET_ID", feature_set["id"])
    monkeypatch.setenv(
        "QUANTLAB_FEATURE_SET_DEFINITION_SHA256", feature_set["definition_sha256"]
    )
    for name, value in isolated.items():
        monkeypatch.setenv(f"QLIB_QUANT_{name.upper()}", value)
    recipe = get_strategy_recipe("short_relative_strength")
    slots = deepcopy(recipe["strategy_rule_ir"]["slots"])
    selected = list(feature_set["features"])[:8]
    slots["alpha_rank"]["components"] = [
        {
            "component": "weighted_factor_rank",
            "parameters": {
                "weights": {factor_id: 1.0 / len(selected) for factor_id in selected}
            },
        }
    ]
    proposal = {
        "contract_version": STRATEGY_PROPOSAL_VERSION,
        "delivery_status": "research_only",
        "name": "short_1_5d governed challenger",
        "description": "A falsifiable research-only structured-rule challenger.",
        "horizon": "short_1_5d",
        "economic_hypothesis": (
            "Test a falsifiable short-horizon ranking change after all costs."
        ),
        "baseline_recipe_id": "short_relative_strength",
        "baseline_recipe_version": recipe["version"],
        "baseline_rules_sha256": recipe["strategy_rule_ir"]["rules_sha256"],
        "parent_strategy_version_id": incumbent_id,
        "changed_slots": ["alpha_rank"],
        "data_contract": {
            "dataset_snapshot_id": dataset_identity_sha256,
            "feature_set_id": feature_set["id"],
            "feature_set_definition_sha256": feature_set["definition_sha256"],
            "research_periods": isolated,
            "decision_frequency": "day",
            "label_horizon_trading_days": 5,
        },
        "evaluation_contract": {
            "benchmark": "SH000300",
            "primary_metric": "after_cost_information_ratio",
            "cost_schedule_version": COST_SCHEDULE_VERSION,
            "rolling_folds": 5,
            "minimum_oos_observations": 252,
            "final_oos_visible_during_selection": False,
        },
        "slots": slots,
    }
    return compile_strategy_proposal(
        proposal,
        allowed_factor_ids=set(feature_set["features"]),
    )


def test_worker_coldstart_materialization_creates_one_draft_and_reuses_it_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periods = {
        "train_start": "2008-01-02",
        "train_end": "2018-12-28",
        "valid_start": "2019-01-02",
        "valid_end": "2022-12-30",
        "test_start": "2023-01-03",
        "test_end": "2025-12-31",
    }
    feature_set = get_feature_set("governed-baseline")
    artifact = _compiled_short_artifact(
        monkeypatch,
        feature_set=feature_set,
        periods=periods,
        dataset_identity_sha256="b" * 64,
        incumbent_id=None,
    )
    binding = _compiled_fin_strategy_materialization_binding(artifact)
    assert binding["dataset_snapshot_id"] == "b" * 64
    strategies = _ColdstartStrategyRecorder()
    worker = SimpleNamespace(strategies=strategies)

    first = Worker._materialize_fin_strategy_candidate(
        worker,
        artifact,
        compiled_artifact_id="compiled-artifact-1",
        allowed_factor_ids=set(feature_set["features"]),
    )
    retried = Worker._materialize_fin_strategy_candidate(
        worker,
        artifact,
        compiled_artifact_id="compiled-artifact-1",
        allowed_factor_ids=set(feature_set["features"]),
    )

    assert first["id"] == "coldstart-candidate-version"
    assert retried == first
    assert first["status"] == "draft"
    assert first["config"]["source_research_artifact_id"] == "compiled-artifact-1"
    assert len(strategies.created) == 1
    assert strategies.created[0]["actor"] == "system:strategy-research"
    assert strategies.created[0]["universe"] == "cn_all"


def test_worker_coldstart_reuses_same_content_from_a_new_compiled_artifact_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periods = {
        "train_start": "2008-01-02",
        "train_end": "2018-12-28",
        "valid_start": "2019-01-02",
        "valid_end": "2022-12-30",
        "test_start": "2023-01-03",
        "test_end": "2025-12-31",
    }
    feature_set = get_feature_set("governed-baseline")
    artifact = _compiled_short_artifact(
        monkeypatch,
        feature_set=feature_set,
        periods=periods,
        dataset_identity_sha256="b" * 64,
        incumbent_id=None,
    )
    strategies = _ColdstartStrategyRecorder()
    worker = SimpleNamespace(strategies=strategies)

    first = Worker._materialize_fin_strategy_candidate(
        worker,
        artifact,
        compiled_artifact_id="compiled-artifact-1",
        allowed_factor_ids=set(feature_set["features"]),
    )
    retried = Worker._materialize_fin_strategy_candidate(
        worker,
        artifact,
        compiled_artifact_id="compiled-artifact-2",
        allowed_factor_ids=set(feature_set["features"]),
    )

    assert retried == first
    assert retried["config"]["source_research_artifact_id"] == "compiled-artifact-1"
    assert len(strategies.created) == 1


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda version: version.update(status="candidate"),
            "exact draft retry",
        ),
        (
            lambda version: version["config"][
                "strategy_research_data_contract"
            ].update(dataset_snapshot_id="c" * 64),
            "exact draft retry",
        ),
        (
            lambda version: version["config"].update(recipe_version="changed"),
            "exact draft retry",
        ),
        (
            lambda version: version.update(strategy_rules_sha256="d" * 64),
            "exact draft retry",
        ),
    ],
)
def test_worker_does_not_reuse_changed_or_non_draft_semantic_candidate(
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    expected: str,
) -> None:
    periods = {
        "train_start": "2008-01-02",
        "train_end": "2018-12-28",
        "valid_start": "2019-01-02",
        "valid_end": "2022-12-30",
        "test_start": "2023-01-03",
        "test_end": "2025-12-31",
    }
    feature_set = get_feature_set("governed-baseline")
    artifact = _compiled_short_artifact(
        monkeypatch,
        feature_set=feature_set,
        periods=periods,
        dataset_identity_sha256="b" * 64,
        incumbent_id=None,
    )
    strategies = _ColdstartStrategyRecorder()
    worker = SimpleNamespace(strategies=strategies)
    Worker._materialize_fin_strategy_candidate(
        worker,
        artifact,
        compiled_artifact_id="compiled-artifact-1",
        allowed_factor_ids=set(feature_set["features"]),
    )
    stored = next(iter(strategies._versions_by_source_artifact.values()))
    mutate(stored)

    with pytest.raises(ValueError, match=expected):
        Worker._materialize_fin_strategy_candidate(
            worker,
            artifact,
            compiled_artifact_id="compiled-artifact-2",
            allowed_factor_ids=set(feature_set["features"]),
        )

    assert len(strategies.created) == 1


def test_worker_materializes_distinct_sealed_content_as_a_new_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periods = {
        "train_start": "2008-01-02",
        "train_end": "2018-12-28",
        "valid_start": "2019-01-02",
        "valid_end": "2022-12-30",
        "test_start": "2023-01-03",
        "test_end": "2025-12-31",
    }
    feature_set = get_feature_set("governed-baseline")
    first_artifact = _compiled_short_artifact(
        monkeypatch,
        feature_set=feature_set,
        periods=periods,
        dataset_identity_sha256="b" * 64,
        incumbent_id=None,
    )
    second_artifact = _compiled_short_artifact(
        monkeypatch,
        feature_set=feature_set,
        periods=periods,
        dataset_identity_sha256="c" * 64,
        incumbent_id=None,
    )
    strategies = _ColdstartStrategyRecorder()
    worker = SimpleNamespace(strategies=strategies)

    Worker._materialize_fin_strategy_candidate(
        worker,
        first_artifact,
        compiled_artifact_id="compiled-artifact-1",
        allowed_factor_ids=set(feature_set["features"]),
    )
    Worker._materialize_fin_strategy_candidate(
        worker,
        second_artifact,
        compiled_artifact_id="compiled-artifact-2",
        allowed_factor_ids=set(feature_set["features"]),
    )

    assert len(strategies.created) == 2
    assert (
        strategies.created[0]["config"]["strategy_research_artifact_sha256"]
        != strategies.created[1]["config"]["strategy_research_artifact_sha256"]
    )


def test_policy_queue_revalidates_with_the_frozen_feature_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_set = get_feature_set("governed-baseline")
    observed: dict[str, set[str] | None] = {}

    def validate(_artifact, *, allowed_factor_ids=None):
        observed["allowed_factor_ids"] = allowed_factor_ids
        raise RuntimeError("validator-observed")

    monkeypatch.setattr(
        "quant_platform.worker.validate_compiled_strategy_artifact",
        validate,
    )
    worker = SimpleNamespace()
    job = {
        "payload": {
            "dataset": "snapshot",
            "dataset_path": str(tmp_path),
            "dataset_identity_sha256": "b" * 64,
            "feature_set": feature_set,
            "periods": {
                "train_start": "2008-01-02",
                "train_end": "2019-11-28",
                "valid_start": "2020-06-10",
                "valid_end": "2023-07-20",
                "test_start": "2024-01-25",
                "test_end": "2026-03-03",
            },
        }
    }

    with pytest.raises(RuntimeError, match="validator-observed"):
        Worker._queue_fin_strategy_policy_evaluations(
            worker,
            "run-1",
            job,
            {"strategy_proposals": [{}]},
            {"strategy_proposal_artifacts": [{}]},
        )

    assert observed["allowed_factor_ids"] == set(feature_set["features"])


def test_policy_queue_derives_competition_from_governed_periods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The competition spans the full governed validation window.

    Proposals record the isolated pre-final research-loop view (enforced at
    archive time).  Deriving the competition from that halved view makes the
    swing/long minimum-OOS floors unreachable, so the queue must derive from
    the governed payload periods instead.
    """

    periods = {
        "train_start": "2008-01-02",
        "train_end": "2019-11-28",
        "valid_start": "2020-06-10",
        "valid_end": "2023-07-20",
        "test_start": "2024-01-25",
        "test_end": "2026-03-03",
    }
    feature_set = get_feature_set("governed-baseline")
    artifact = _compiled_short_artifact(
        monkeypatch,
        feature_set=feature_set,
        periods=periods,
        dataset_identity_sha256="b" * 64,
        incumbent_id=None,
    )
    observed: dict[str, dict] = {}

    def derive(research_periods, **_kwargs):
        observed["periods"] = dict(research_periods)
        raise RuntimeError("derive-observed")

    monkeypatch.setattr(
        "quant_platform.worker.derive_strategy_research_competition_periods",
        derive,
    )
    monkeypatch.setattr(
        "quant_platform.worker.build_public_strategy_control_config",
        lambda _config: {},
    )
    monkeypatch.setattr(
        "quant_platform.worker.build_transparent_full_stack_control_config",
        lambda _config: {},
    )
    strategies = SimpleNamespace(
        get_version=lambda version_id: {
            "id": version_id,
            "config": {"outer_purge_days": 6},
            "benchmark": "SH000300",
            "universe": "cn_all",
        }
    )
    worker = SimpleNamespace(strategies=strategies)
    job = {
        "payload": {
            "dataset": "snapshot",
            "dataset_path": str(tmp_path),
            "dataset_identity_sha256": "b" * 64,
            "feature_set": feature_set,
            "periods": periods,
        }
    }

    with pytest.raises(RuntimeError, match="derive-observed"):
        Worker._queue_fin_strategy_policy_evaluations(
            worker,
            "run-1",
            job,
            {"strategy_proposals": [artifact]},
            {
                "strategy_proposal_artifacts": [
                    {
                        "strategy_version_id": "a" * 32,
                        "compiled_artifact_id": "compiled-artifact-1",
                    }
                ]
            },
        )

    isolated = artifact["strategy_proposal"]["data_contract"]["research_periods"]
    assert isolated != periods
    assert observed["periods"] == periods


def _job_payload(feature_set: dict, periods: dict[str, str]) -> dict:
    incumbent_id = "a" * 32
    return {
        "payload": {
            "scenario": "fin_strategy",
            "research_run_id": "run-1",
            "loop_n": 1,
            "feature_set": feature_set,
            "dataset_identity_sha256": "b" * 64,
            "periods": periods,
            "strategy_horizon_profile": "short_1_5d",
            "incumbent_strategy": {
                "id": incumbent_id,
                "horizon_profile": "short_1_5d",
            },
        }
    }


def test_worker_archives_strategy_proposal_and_compiled_ir_as_research_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periods = {
        "train_start": "2008-01-02",
        "train_end": "2018-12-28",
        "valid_start": "2019-01-02",
        "valid_end": "2022-12-30",
        "test_start": "2023-01-03",
        "test_end": "2025-12-31",
    }
    feature_set = get_feature_set("governed-baseline")
    job = _job_payload(feature_set, periods)
    artifact = _compiled_short_artifact(
        monkeypatch,
        feature_set=feature_set,
        periods=periods,
        dataset_identity_sha256="b" * 64,
        incumbent_id="a" * 32,
    )
    recorder = _ArtifactRecorder()
    strategies = _StrategyRecorder()
    worker = SimpleNamespace(
        settings=SimpleNamespace(data_root=tmp_path),
        rdagent_candidates=recorder,
        strategies=strategies,
    )

    result = Worker._archive_fin_strategy_artifacts(
        worker,
        "run-1",
        job,
        {"strategy_proposals": [artifact]},
        sanitized_result_artifact_id="sanitized-1",
        sanitized_result_sha256="c" * 64,
    )

    assert result["strategy_research_status"] == "compiled_research_only"
    assert result["strategy_proposal_count"] == 1
    assert result["capital_eligible"] is False
    assert result["simulation_eligible"] is False
    assert result["recommendation_eligible"] is False
    assert result["strategy_proposal_artifacts"][0]["strategy_version_id"] == (
        "candidate-version"
    )
    assert result["strategy_proposal_artifacts"][0]["strategy_lifecycle"] == (
        "research_candidate"
    )
    assert len(strategies.created) == 1
    config = strategies.created[0]["config"]
    assert config["source_research_artifact_id"] == "artifact-2"
    assert config["horizon_profile"] == "short_1_5d"
    assert config["execution_lag_bars"] == 1
    assert [call["artifact_type"] for call in recorder.calls] == [
        "fin_strategy_proposal",
        "fin_strategy_compiled_artifact",
        "fin_strategy_audit_manifest",
    ]
    assert all(call["metadata"]["capital_eligible"] is False for call in recorder.calls)
    assert not hasattr(worker, "simulations")
    assert not hasattr(worker, "recommendations")


def test_worker_rejects_strategy_artifact_with_a_different_dataset_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periods = {
        "train_start": "2008-01-02",
        "train_end": "2018-12-28",
        "valid_start": "2019-01-02",
        "valid_end": "2022-12-30",
        "test_start": "2023-01-03",
        "test_end": "2025-12-31",
    }
    feature_set = get_feature_set("governed-baseline")
    job = _job_payload(feature_set, periods)
    artifact = _compiled_short_artifact(
        monkeypatch,
        feature_set=feature_set,
        periods=periods,
        dataset_identity_sha256="d" * 64,
        incumbent_id="a" * 32,
    )
    recorder = _ArtifactRecorder()
    worker = SimpleNamespace(
        settings=SimpleNamespace(data_root=tmp_path),
        rdagent_candidates=recorder,
    )

    with pytest.raises(ValueError, match="input binding disagrees"):
        Worker._archive_fin_strategy_artifacts(
            worker,
            "run-1",
            job,
            {"strategy_proposals": [artifact]},
            sanitized_result_artifact_id="sanitized-1",
            sanitized_result_sha256="c" * 64,
        )

    assert recorder.calls == []


def test_worker_rejects_strategy_archive_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periods = {
        "train_start": "2008-01-02",
        "train_end": "2018-12-28",
        "valid_start": "2019-01-02",
        "valid_end": "2022-12-30",
        "test_start": "2023-01-03",
        "test_end": "2025-12-31",
    }
    feature_set = get_feature_set("governed-baseline")
    job = _job_payload(feature_set, periods)
    artifact = _compiled_short_artifact(
        monkeypatch,
        feature_set=feature_set,
        periods=periods,
        dataset_identity_sha256="b" * 64,
        incumbent_id="a" * 32,
    )
    recorder = _ArtifactRecorder()
    worker = SimpleNamespace(
        settings=SimpleNamespace(data_root=tmp_path),
        rdagent_candidates=recorder,
    )

    with pytest.raises(ValueError, match="escapes its governed artifact root"):
        Worker._archive_fin_strategy_artifacts(
            worker,
            "../outside",
            job,
            {"strategy_proposals": [artifact]},
            sanitized_result_artifact_id="sanitized-1",
            sanitized_result_sha256="c" * 64,
        )

    assert recorder.calls == []
