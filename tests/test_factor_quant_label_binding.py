from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from quant_platform.factor_evaluation_recovery import (
    RecoverySafetyError,
    validate_factor_evaluation_result_contract,
)
from quant_platform.model_research_governance import canonical_sha256
from quant_platform.research_horizon import (
    LONG_1_3Y,
    SHORT_1_5D,
    SWING_1_6M,
    research_horizon_contract,
)
from quant_platform.research_label_binding import (
    resolve_research_label_binding,
    validate_research_label_binding,
)
from quant_platform.worker import LocalJobWorker

pytestmark = pytest.mark.no_database


def _payload(
    profile: str,
    *,
    selected: int | None = None,
    feature_set: bool = True,
) -> dict[str, Any]:
    horizon = research_horizon_contract(profile)
    periods = {
        "train_start": "2010-01-04",
        "train_end": "2018-12-28",
        "valid_start": "2019-01-02",
        "valid_end": "2022-12-30",
        "test_start": "2024-01-03",
        "test_end": "2026-08-20",
    }
    frozen_features = (
        {
            "id": "governed-features-v1",
            "definition_sha256": "f" * 64,
            "features": {"alpha": {"expression": "$close"}},
        }
        if feature_set
        else None
    )
    window = {
        "contract_version": "research-window-v1",
        "horizon_profile": profile,
        "horizon_contract_sha256": horizon.sha256,
        "dataset_name": "cn-governed-day",
        "dataset_identity_sha256": "d" * 64,
        "feature_set_id": frozen_features["id"] if frozen_features else None,
        "feature_set_sha256": (
            frozen_features["definition_sha256"] if frozen_features else None
        ),
        "label_horizons_sessions": list(horizon.label_horizons_sessions),
        "purge_sessions": horizon.purge_sessions,
        "embargo_sessions": horizon.embargo_sessions,
        "label_maturity_enforced": True,
        "periods": periods,
    }
    value: dict[str, Any] = {
        "horizon_profile": profile,
        "dataset": window["dataset_name"],
        "dataset_identity_sha256": window["dataset_identity_sha256"],
        "periods": periods,
        "feature_set": frozen_features,
        "research_window_contract": window,
        "research_window_contract_sha256": canonical_sha256(window),
    }
    if selected is not None:
        value["label_horizon_sessions"] = selected
    return value


@pytest.mark.parametrize(
    ("profile", "expected"),
    ((SHORT_1_5D, 5), (SWING_1_6M, 126), (LONG_1_3Y, 252)),
)
def test_active_factor_and_quant_labels_default_to_the_longest_window_label(
    profile: str, expected: int
) -> None:
    binding = resolve_research_label_binding(_payload(profile))

    assert binding is not None
    assert binding["label_horizon_sessions"] == expected
    assert binding["label_expression"] == (
        f"Ref($close,-{expected + 1})/Ref($close,-1)-1"
    )
    assert validate_research_label_binding(binding) == binding


def test_active_label_selection_accepts_only_a_member_of_the_verified_window() -> None:
    binding = resolve_research_label_binding(_payload(SWING_1_6M, selected=63))

    assert binding is not None
    assert binding["label_horizon_sessions"] == 63
    with pytest.raises(ValueError, match="not allowed"):
        resolve_research_label_binding(_payload(SWING_1_6M, selected=2))
    with pytest.raises(ValueError, match="no verified research window"):
        resolve_research_label_binding(
            {"horizon_profile": LONG_1_3Y, "label_horizon_sessions": 252}
        )


def test_label_binding_rejects_dataset_period_and_digest_substitution() -> None:
    payload = _payload(LONG_1_3Y)
    payload["dataset_identity_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="dataset identity differs"):
        resolve_research_label_binding(payload)

    payload = _payload(LONG_1_3Y)
    payload["periods"] = {**payload["periods"], "test_end": "2026-08-21"}
    with pytest.raises(ValueError, match="periods differ"):
        resolve_research_label_binding(payload)

    payload = _payload(LONG_1_3Y)
    payload["research_window_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest is missing or invalid"):
        resolve_research_label_binding(payload)


class _CaptureResearch:
    def __init__(self) -> None:
        self.added: list[dict[str, Any]] = []
        self.attached: list[tuple[str, str]] = []

    def add_candidate(self, run_id: str, **values: Any) -> dict[str, Any]:
        row = {"id": f"candidate-{len(self.added) + 1}", **values}
        self.added.append({"run_id": run_id, **row})
        return row

    def attach_job(self, run_id: str, job_id: str) -> None:
        self.attached.append((run_id, job_id))

    def list_candidates(self, **_values: Any) -> list[dict[str, Any]]:
        return []


class _CaptureJobs:
    def __init__(self) -> None:
        self.created: dict[str, Any] | None = None

    def create(
        self,
        kind: str,
        payload: dict[str, Any],
        log_path: Path,
        **_values: Any,
    ) -> dict[str, Any]:
        self.created = {"id": "factor-job-1", "kind": kind, "payload": payload}
        return self.created


def test_worker_overrides_untrusted_rdagent_label_and_freezes_factor_job(
    tmp_path: Path,
) -> None:
    payload = {
        **_payload(SWING_1_6M),
        "research_run_id": "run-1",
        "dataset_path": str(tmp_path / "qlib"),
        "evaluation_profiles": [],
    }
    code_path = tmp_path / "factor.py"
    values_path = tmp_path / "values.h5"
    code_path.write_text("def factor_feature(data):\n    return data\n", encoding="utf-8")
    values_path.write_bytes(b"values")
    research = _CaptureResearch()
    jobs = _CaptureJobs()
    worker = object.__new__(LocalJobWorker)
    worker.research = research
    worker.store = jobs
    worker.settings = SimpleNamespace(data_root=tmp_path)
    worker.factor_library = SimpleNamespace()
    job = {"kind": "rdagent_factor", "payload": payload}
    result = {
        "candidates": [
            {
                "name": "swing-factor",
                "description": "factor",
                "variables": {},
                "formulation": None,
                "code_path": str(code_path),
                "values_path": str(values_path),
                "rdagent_decision": True,
                # This is deliberately hostile/stale and must not be trusted.
                "label_horizon_days": 1,
            }
        ]
    }

    candidates = worker._import_rdagent_candidates("run-1", job, result)
    binding = resolve_research_label_binding(payload)
    assert binding is not None
    assert candidates[0]["label_horizon_days"] == 126
    assert candidates[0]["variables"]["rdagent_reported_label_horizon_days"] == 1
    assert candidates[0]["variables"]["research_label_binding_sha256"] == binding[
        "binding_sha256"
    ]

    worker._queue_factor_evaluation(job, candidates)
    assert jobs.created is not None
    frozen = jobs.created["payload"]
    assert frozen["label_horizon_sessions"] == 126
    assert frozen["candidates"][0]["label_horizon_days"] == 126
    assert frozen["research_label_binding"] == binding


def test_factor_result_contract_rejects_label_evidence_substitution() -> None:
    payload = _payload(SHORT_1_5D, selected=3, feature_set=False)
    binding = resolve_research_label_binding(payload)
    assert binding is not None
    candidate = {
        "id": "candidate-1",
        "label_horizon_days": 3,
    }
    job = {
        "payload": {
            **payload,
            "candidates": [candidate],
            "research_label_binding": binding,
            "research_label_binding_sha256": binding["binding_sha256"],
        }
    }
    result = {
        "status": "ok",
        "qlib_workflow": {"run_id": "workflow-1"},
        "research_label_binding": binding,
        "research_label_binding_sha256": binding["binding_sha256"],
        "evaluations": [
            {
                "candidate_id": "candidate-1",
                "status": "ok",
                "periods": payload["periods"],
                "metrics": {},
                "recomputed_values_path": "recomputed.h5",
                "recomputed_values_sha256": "a" * 64,
                "recompute_evidence": {
                    "label_horizon_days": 3,
                    "research_label_binding_sha256": binding["binding_sha256"],
                    "research_window_contract_sha256": binding[
                        "research_window_contract_sha256"
                    ],
                    "horizon_profile": SHORT_1_5D,
                },
            }
        ],
    }

    assert validate_factor_evaluation_result_contract(job, result) == (1, 0)
    result["evaluations"][0]["recompute_evidence"]["label_horizon_days"] = 1
    with pytest.raises(RecoverySafetyError, match="another label contract"):
        validate_factor_evaluation_result_contract(job, result)


def test_quant_evaluator_uses_the_verified_label_instead_of_one_day_literal() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "evaluate_quant_bundle.py"
    ).read_text(encoding="utf-8")

    assert "validate_research_label_binding" in source
    assert 'label_horizon_sessions = int(manifest.get("label_horizon_sessions") or 1)' in source
    assert "label_horizon_sessions + 1" in source
    assert 'fields=["Ref($close, -2)/Ref($close, -1)-1"]' not in source


def _frozen_model_prediction(binding: dict[str, Any]) -> dict[str, Any]:
    label_contract = {
        "contract_version": "model-label-contract-v1",
        "horizon_profile": binding["horizon_profile"],
        "legacy": False,
        "allowed_label_horizons_sessions": binding[
            "allowed_label_horizons_sessions"
        ],
        "label_horizon_sessions": binding["label_horizon_sessions"],
        "label_reference_offset_sessions": binding[
            "label_reference_offset_sessions"
        ],
        "label_expression": binding["label_expression"],
        "purge_sessions": binding["purge_sessions"],
        "embargo_sessions": binding["embargo_sessions"],
        "research_window_contract_sha256": binding[
            "research_window_contract_sha256"
        ],
    }
    return {
        "kind": "model",
        "profiles": {
            "recent_3y": {
                "seeds": {
                    "11": {
                        "model_label_contract": label_contract,
                        "model_label_contract_sha256": canonical_sha256(label_contract),
                    }
                }
            }
        },
    }


def test_fin_quant_incumbent_must_have_the_same_bound_prediction_label() -> None:
    binding = resolve_research_label_binding(_payload(SWING_1_6M))
    assert binding is not None
    frozen = _frozen_model_prediction(binding)

    LocalJobWorker._require_prediction_label_matches_binding(frozen, binding)

    wrong_contract = frozen["profiles"]["recent_3y"]["seeds"]["11"][
        "model_label_contract"
    ]
    wrong_contract["label_horizon_sessions"] = 2
    frozen["profiles"]["recent_3y"]["seeds"]["11"][
        "model_label_contract_sha256"
    ] = canonical_sha256(wrong_contract)
    with pytest.raises(ValueError, match="another label horizon"):
        LocalJobWorker._require_prediction_label_matches_binding(frozen, binding)
