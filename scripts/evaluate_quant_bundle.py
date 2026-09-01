from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_data.qlib_builder import qlib_research_field_catalog, verify_qlib_output_manifest
from quant_platform.factor_library import compile_qlib_expression
from quant_platform.factor_recompute import (
    compare_submitted_values,
    execute_factor_code,
    normalize_factor_input,
    require_exact_factor_index,
    sha256_file,
    validate_factor_prefix_invariance,
)
from quant_platform.feature_set_registry import resolve_feature_set
from quant_platform.horizon_factor_bundle import validate_horizon_factor_bundle
from quant_platform.model_ensemble import equal_rank_predictions
from quant_platform.model_recompute import (
    GOVERNED_MODEL_ENGINES,
    ModelResourceLimitError,
    execute_model_candidate,
    governed_checkpoint_filename,
    verify_governed_checkpoint,
)
from quant_platform.model_research_governance import (
    QUANT_BUNDLE_CONTRACT_VERSION,
    REQUIRED_MODEL_SEEDS,
    REQUIRED_QUANT_ABLATIONS,
    REQUIRED_RESEARCH_PROFILES,
    build_run_multiple_testing_evidence,
    canonical_sha256,
    file_sha256,
    is_sha256,
    paired_moving_block_bootstrap,
    validate_quant_bundle_evidence,
    verify_model_prediction_artifact,
)
from quant_platform.qlib_workflow import (
    qlib_workflow_run,
    qlib_workflow_tracking_uri,
)
from quant_platform.rdagent_dataset_view import prepare_rdagent_dataset_view
from quant_platform.research_label_binding import validate_research_label_binding


class UnsupportedQuantBaseline(ValueError):
    """A frozen incumbent feature contract cannot be evaluated without substitution."""


def _equal_except_paths(
    frozen: Any,
    runtime: Any,
    *,
    allowed_path_keys: set[str],
) -> bool:
    if isinstance(frozen, dict) and isinstance(runtime, dict):
        if set(frozen) != set(runtime):
            return False
        return all(
            True
            if key in allowed_path_keys
            else _equal_except_paths(
                frozen[key], runtime[key], allowed_path_keys=allowed_path_keys
            )
            for key in frozen
        )
    if isinstance(frozen, list) and isinstance(runtime, list):
        return len(frozen) == len(runtime) and all(
            _equal_except_paths(left, right, allowed_path_keys=allowed_path_keys)
            for left, right in zip(frozen, runtime, strict=True)
        )
    return frozen == runtime


def _runtime_baseline(bundle: dict[str, Any]) -> dict[str, Any]:
    frozen = bundle.get("baseline_prediction_champion")
    runtime = bundle.get("baseline_prediction_runtime")
    if not isinstance(frozen, dict) or not isinstance(runtime, dict):
        raise ValueError("quant incumbent runtime mapping is missing")
    identity_keys = (
        "contract_version",
        "kind",
        "candidate_id",
        "candidate_manifest_sha256",
        "admission_evidence_sha256",
        "dataset",
        "dataset_identity_sha256",
        "pre_final_end",
        "final_oos_start",
        "final_oos_end",
        "feature_set_id",
        "feature_set_definition_sha256",
        "quant_retraining_supported",
        "selection_evidence_sha256",
    )
    if any(runtime.get(key) != frozen.get(key) for key in identity_keys):
        raise ValueError("quant incumbent runtime identity changed")
    if frozen.get("kind") == "model":
        frozen_model = dict(frozen.get("model") or {})
        runtime_model = dict(runtime.get("model") or {})
        for key in frozen_model:
            if key != "code_path" and runtime_model.get(key) != frozen_model.get(key):
                raise ValueError("quant incumbent runtime model recipe changed")
        for profile_id, frozen_profile in frozen.get("profiles", {}).items():
            runtime_profile = runtime.get("profiles", {}).get(profile_id)
            if not isinstance(runtime_profile, dict) or runtime_profile.get(
                "periods"
            ) != frozen_profile.get("periods"):
                raise ValueError("quant incumbent runtime periods changed")
            for seed, frozen_cell in frozen_profile.get("seeds", {}).items():
                runtime_cell = runtime_profile.get("seeds", {}).get(seed)
                if not isinstance(runtime_cell, dict) or any(
                    runtime_cell.get(key) != frozen_cell.get(key)
                    for key in ("predictions_sha256", "portfolio_report_sha256")
                ):
                    raise ValueError("quant incumbent runtime artifacts changed")
    elif frozen.get("kind") == "ensemble":
        if not _equal_except_paths(
            frozen,
            runtime,
            allowed_path_keys={
                "code_path",
                "predictions_path",
                "checkpoint_path",
                "portfolio_report_path",
            },
        ):
            raise ValueError("quant incumbent ensemble runtime contract changed")
    return runtime


def _calendar_between(provider: Path, start: str, end: str) -> list[str]:
    calendar = [
        value.strip()
        for value in (provider / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    selected = [value for value in calendar if start <= value <= end]
    if not selected or selected[0] != start or selected[-1] != end:
        raise ValueError("quant ablation window does not match the Qlib trading calendar")
    return selected


def _frozen_candidate_model_engine(model: dict[str, Any]) -> str:
    hyperparameters = model.get("model_hyperparameters") or {}
    if not isinstance(hyperparameters, dict):
        raise ValueError("quant bundle model hyperparameters are invalid")
    direct = str(model.get("model_engine") or "").strip()
    nested = str(hyperparameters.get("model_engine") or "").strip()
    if direct and nested and direct != nested:
        raise ValueError("quant bundle model engine changed inside the frozen recipe")
    engine = direct or nested
    if not engine:
        raise ValueError("quant bundle model has no frozen model_engine")
    if engine not in GOVERNED_MODEL_ENGINES:
        raise ValueError("quant bundle model requests an ungoverned model engine")
    return engine


def _ablation_model_config(
    name: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    if name not in REQUIRED_QUANT_ABLATIONS:
        raise ValueError(f"unknown quant ablation {name!r}")
    if name == "factor_only":
        baseline = _runtime_baseline(bundle)
        if not isinstance(baseline, dict):
            raise ValueError("quant factor-only ablation has no frozen incumbent")
        if baseline.get("kind") == "ensemble":
            raise UnsupportedQuantBaseline(
                "ensemble_factor_only_requires_member_dispatch"
            )
        if baseline.get("kind") != "model" or not isinstance(
            baseline.get("model"), dict
        ):
            raise ValueError("quant factor-only incumbent identity is invalid")
        model = dict(baseline["model"])
        code_path = Path(str(model.get("code_path") or ""))
        if not code_path.is_file() or file_sha256(code_path) != str(
            model.get("code_sha256") or ""
        ):
            raise ValueError("quant factor-only incumbent model code changed")
        return {
            "code_path": code_path,
            "code_sha256": str(model["code_sha256"]),
            "model_type": str(model.get("model_type") or "Tabular"),
            "model_engine": _frozen_candidate_model_engine(model),
            "training_hyperparameters": dict(
                model.get("training_hyperparameters") or {}
            ),
        }
    model = bundle["model"]
    code_path = Path(model["code_path"])
    code_sha256 = file_sha256(code_path)
    if code_sha256 != model["code_sha256"]:
        raise ValueError("quant bundle model code changed after proposal")
    return {
        "code_path": code_path,
        "code_sha256": code_sha256,
        "model_type": str(model.get("model_type") or "Tabular"),
        "model_engine": _frozen_candidate_model_engine(model),
        "training_hyperparameters": dict(
            model.get("training_hyperparameters") or {}
        ),
    }


def _verify_runtime_model_profiles(
    profiles: dict[str, Any], *, expected_model_engine: str
) -> None:
    if set(profiles) != set(REQUIRED_RESEARCH_PROFILES):
        raise ValueError("quant incumbent model profile grid is incomplete")
    for profile_id in REQUIRED_RESEARCH_PROFILES:
        profile = profiles[profile_id]
        periods = dict(profile.get("periods") or {})
        seeds = profile.get("seeds")
        if not isinstance(seeds, dict) or {int(seed) for seed in seeds} != set(
            REQUIRED_MODEL_SEEDS
        ):
            raise ValueError("quant incumbent model seed grid is incomplete")
        for seed in REQUIRED_MODEL_SEEDS:
            cell = seeds[str(seed)]
            for path_key, digest_key in (
                ("predictions_path", "predictions_sha256"),
                ("checkpoint_path", "checkpoint_sha256"),
                ("portfolio_report_path", "portfolio_report_sha256"),
            ):
                path = Path(str(cell.get(path_key) or ""))
                if (
                    not path.is_file()
                    or file_sha256(path) != str(cell.get(digest_key) or "")
                ):
                    raise ValueError(
                        f"quant incumbent model {profile_id}/{seed} artifact changed"
                    )
            if str(cell.get("model_engine") or "") != expected_model_engine:
                raise ValueError("quant incumbent model engine changed across cells")
            verify_governed_checkpoint(
                Path(str(cell["checkpoint_path"])),
                model_engine=expected_model_engine,
                checkpoint_format=str(cell.get("checkpoint_format") or ""),
                expected_sha256=str(cell["checkpoint_sha256"]),
            )
            coverage = cell.get("coverage")
            if (
                not isinstance(coverage, dict)
                or coverage.get("coverage_gate_passed") is not True
                or coverage.get("artifact_sha256") != cell["predictions_sha256"]
                or coverage.get("test_start") != periods.get("valid_start")
                or coverage.get("test_end") != periods.get("valid_end")
            ):
                raise ValueError("quant incumbent model coverage proof changed")


def _verify_runtime_ensemble_profiles(
    profiles: dict[str, Any], *, expected_member_ids: set[str]
) -> None:
    if set(profiles) != set(REQUIRED_RESEARCH_PROFILES):
        raise ValueError("quant incumbent ensemble profile grid is incomplete")
    for profile_id in REQUIRED_RESEARCH_PROFILES:
        profile = profiles[profile_id]
        periods = dict(profile.get("periods") or {})
        seeds = profile.get("seeds")
        if not isinstance(seeds, dict) or {int(seed) for seed in seeds} != set(
            REQUIRED_MODEL_SEEDS
        ):
            raise ValueError("quant incumbent ensemble seed grid is incomplete")
        for seed in REQUIRED_MODEL_SEEDS:
            cell = seeds[str(seed)]
            for path_key, digest_key in (
                ("predictions_path", "predictions_sha256"),
                ("portfolio_report_path", "portfolio_report_sha256"),
            ):
                path = Path(str(cell.get(path_key) or ""))
                if (
                    not path.is_file()
                    or file_sha256(path) != str(cell.get(digest_key) or "")
                ):
                    raise ValueError(
                        f"quant incumbent ensemble {profile_id}/{seed} artifact changed"
                    )
            coverage = cell.get("coverage")
            if (
                not isinstance(coverage, dict)
                or coverage.get("coverage_gate_passed") is not True
                or coverage.get("artifact_sha256") != cell["predictions_sha256"]
                or coverage.get("test_start") != periods.get("valid_start")
                or coverage.get("test_end") != periods.get("valid_end")
            ):
                raise ValueError("quant incumbent ensemble coverage proof changed")
            members = cell.get("member_prediction_artifacts")
            member_ids = {
                str(item.get("model_candidate_id") or "")
                for item in members or []
                if isinstance(item, dict)
            }
            if (
                not isinstance(members, list)
                or member_ids != expected_member_ids
                or len(members) != len(expected_member_ids)
            ):
                raise ValueError("quant incumbent ensemble member evidence is incomplete")
            for member in members:
                path = Path(str(member.get("predictions_path") or ""))
                if (
                    not path.is_file()
                    or file_sha256(path)
                    != str(member.get("predictions_sha256") or "")
                ):
                    raise ValueError(
                        "quant incumbent ensemble member predictions changed"
                    )
            combination = cell.get("combination_evidence")
            if (
                not isinstance(combination, dict)
                or combination.get("combiner") != "equal_rank"
                or combination.get("stacking") is not False
                or set(str(item) for item in combination.get("member_ids") or [])
                != expected_member_ids
                or canonical_sha256(
                    {
                        key: value
                        for key, value in combination.items()
                        if key != "evidence_sha256"
                    }
                )
                != combination.get("evidence_sha256")
            ):
                raise ValueError("quant incumbent ensemble combination proof changed")


def _validate_baseline_prediction(
    *,
    baseline: Any,
    runtime_baseline: Any,
    manifest_baseline: Any,
    dataset_identity_sha256: str,
    feature_set_definition_sha256: str,
) -> dict[str, Any]:
    if not isinstance(baseline, dict) or not isinstance(manifest_baseline, dict):
        raise ValueError("quant evaluation has no frozen incumbent")
    if baseline != manifest_baseline:
        raise ValueError("quant candidate incumbent differs from the job contract")
    expected = canonical_sha256(
        {key: value for key, value in baseline.items() if key != "evidence_sha256"}
    )
    if (
        baseline.get("contract_version") != "fin-quant-baseline-prediction-v1"
        or baseline.get("kind") not in {"model", "ensemble"}
        or baseline.get("dataset_identity_sha256") != dataset_identity_sha256
        or baseline.get("evidence_sha256") != expected
        or not is_sha256(baseline.get("candidate_manifest_sha256"))
        or not is_sha256(baseline.get("admission_evidence_sha256"))
        or not is_sha256(baseline.get("selection_evidence_sha256"))
    ):
        raise ValueError("quant incumbent immutable identity is invalid")
    if baseline["kind"] == "ensemble":
        components = baseline.get("components")
        if (
            baseline.get("combiner") != "equal_rank"
            or baseline.get("stacking") is not False
            or baseline.get("quant_retraining_supported") is not True
            or baseline.get("member_retraining_contract_version")
            != "fin-quant-ensemble-member-retraining-v1"
            or not isinstance(components, list)
            or not 2 <= len(components) <= 3
            or not isinstance(baseline.get("profiles"), dict)
        ):
            raise ValueError("quant ensemble incumbent retraining contract is invalid")
        runtime = _runtime_baseline(
            {
                "baseline_prediction_champion": baseline,
                "baseline_prediction_runtime": runtime_baseline,
            }
        )
        component_ids: set[str] = set()
        for component in runtime["components"]:
            component_id = str(component.get("model_candidate_id") or "")
            model = dict(component.get("model") or {})
            feature_set_id = str(component.get("feature_set_id") or "")
            if (
                not component_id
                or component_id in component_ids
                or model.get("candidate_id") != component_id
                or component.get("weight") != 1.0 / len(components)
                or not is_sha256(component.get("candidate_manifest_sha256"))
                or not is_sha256(component.get("admission_evidence_sha256"))
                or not is_sha256(component.get("prediction_grid_sha256"))
            ):
                raise ValueError("quant ensemble incumbent member identity is invalid")
            component_ids.add(component_id)
            try:
                feature_set = resolve_feature_set(
                    feature_set_id, dict(component.get("feature_set") or {})
                )
            except ValueError as exc:
                raise UnsupportedQuantBaseline(
                    f"ensemble_member_feature_contract_unavailable:{component_id}"
                ) from exc
            if (
                feature_set["definition_sha256"]
                != component.get("feature_set_definition_sha256")
            ):
                raise ValueError("quant ensemble member feature contract changed")
            code_path = Path(str(model.get("code_path") or ""))
            if (
                not code_path.is_file()
                or file_sha256(code_path) != str(model.get("code_sha256") or "")
                or not is_sha256(model.get("recipe_sha256"))
            ):
                raise ValueError("quant ensemble incumbent model artifact changed")
            _frozen_candidate_model_engine(model)
            _verify_runtime_model_profiles(
                dict(component.get("profiles") or {}),
                expected_model_engine=_frozen_candidate_model_engine(model),
            )
        _verify_runtime_ensemble_profiles(
            dict(runtime.get("profiles") or {}), expected_member_ids=component_ids
        )
        return dict(baseline)
    if (
        baseline.get("feature_set_definition_sha256")
        != feature_set_definition_sha256
        or baseline.get("quant_retraining_supported") is not True
        or not isinstance(baseline.get("model"), dict)
        or not isinstance(baseline.get("profiles"), dict)
    ):
        raise ValueError("quant model incumbent contract is invalid")
    runtime = _runtime_baseline(
        {
            "baseline_prediction_champion": baseline,
            "baseline_prediction_runtime": runtime_baseline,
        }
    )
    model = dict(runtime["model"])
    code_path = Path(str(model.get("code_path") or ""))
    if (
        not code_path.is_file()
        or file_sha256(code_path) != str(model.get("code_sha256") or "")
        or not is_sha256(model.get("recipe_sha256"))
    ):
        raise ValueError("quant incumbent model artifact changed")
    return dict(baseline)


def _resource_limit_evidence(
    *,
    workspace: Path,
    error: ModelResourceLimitError,
) -> dict[str, Any]:
    manifest_path = workspace / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("resource-limited model cell has no immutable runtime manifest")
    runtime_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resource_policy = runtime_manifest.get("resource_policy")
    execution_environment = runtime_manifest.get("execution_environment")
    execution_environment_sha256 = runtime_manifest.get(
        "execution_environment_sha256"
    )
    if not isinstance(resource_policy, dict) or not isinstance(
        execution_environment, dict
    ):
        raise ValueError("resource-limited model cell has incomplete policy evidence")
    if canonical_sha256(execution_environment) != execution_environment_sha256:
        raise ValueError("resource-limited model cell changed its execution environment")
    evidence = {
        "contract_version": "model-resource-limit-evidence-v1",
        "status": "resource_blocked",
        "error": str(error),
        "input_manifest_sha256": canonical_sha256(runtime_manifest),
        "resource_policy": resource_policy,
        "execution_environment": execution_environment,
        "execution_environment_sha256": execution_environment_sha256,
        "final_oos_opened": False,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def _failed_evaluation_summary(evaluations: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{item.get('candidate_id')}: "
        f"{item.get('error', 'quant evaluation failed')}"
        for item in evaluations
        if item.get("status") == "failed"
    )[:3000]


def _frozen_incumbent_return_series(runtime_baseline: dict[str, Any]) -> pd.Series:
    baseline_seed_returns: list[pd.Series] = []
    baseline_seeds = runtime_baseline["profiles"]["recent_3y"]["seeds"]
    for seed in REQUIRED_MODEL_SEEDS:
        baseline_cell = baseline_seeds[str(seed)]
        baseline_path = Path(str(baseline_cell["portfolio_report_path"]))
        if (
            not baseline_path.is_file()
            or file_sha256(baseline_path)
            != baseline_cell["portfolio_report_sha256"]
        ):
            raise ValueError("frozen fin_quant incumbent return artifact changed")
        baseline_report = pd.read_parquet(baseline_path)
        if not {"return", "bench", "cost"}.issubset(baseline_report.columns):
            raise ValueError("frozen fin_quant incumbent return report is incomplete")
        baseline_series = (
            pd.to_numeric(baseline_report["return"], errors="coerce")
            - pd.to_numeric(baseline_report["bench"], errors="coerce")
            - pd.to_numeric(baseline_report["cost"], errors="coerce")
        ).rename(str(seed))
        baseline_series.index = pd.to_datetime(
            baseline_series.index
        ).tz_localize(None)
        baseline_seed_returns.append(baseline_series)
    baseline_frame = pd.concat(
        baseline_seed_returns, axis=1, join="inner"
    ).dropna()
    if (
        baseline_frame.shape[1] != len(REQUIRED_MODEL_SEEDS)
        or len(baseline_frame) < 40
    ):
        raise ValueError("frozen fin_quant incumbent seed return matrix is incomplete")
    return baseline_frame.mean(axis=1)


def _finalize_candidate_multiple_testing(
    *,
    item: dict[str, Any],
    multiple: dict[str, Any],
    candidate_returns: dict[tuple[str, str], pd.Series],
    incumbent_returns: dict[str, pd.Series],
    dataset_identity_sha256: str,
) -> None:
    if item.get("status") != "passed":
        return
    evidence = item["evidence"]
    candidate_id = str(item["candidate_id"])
    baseline_id = str(evidence["baseline_prediction_champion"]["candidate_id"])
    aligned = pd.concat(
        [
            candidate_returns[(candidate_id, "joint")].rename("joint"),
            incumbent_returns[baseline_id].rename("incumbent"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    comparison = paired_moving_block_bootstrap(
        aligned["joint"],
        aligned["incumbent"],
        block_size=min(20, len(aligned)),
        samples=2000,
        seed=0,
    )
    delta_trial_name = f"{candidate_id}:joint_vs_incumbent"
    trial_index = {
        str(name): index for index, name in enumerate(multiple["trial_names"])
    }
    delta_index = trial_index[delta_trial_name]
    raw_p_value = float(multiple["raw_p_values"][delta_index])
    adjusted_p_value = float(multiple["holm_adjusted_p_values"][delta_index])
    family_mean_difference = float(multiple["trial_daily_means"][delta_index])
    pbo_eligible = delta_trial_name in set(multiple["eligible_trial_names"])
    comparison.update(
        {
            "contract_version": "fin-quant-incumbent-comparison-v2",
            "candidate_id": candidate_id,
            "incumbent_candidate_id": baseline_id,
            "profile_id": "recent_3y",
            "seed_aggregation": "equal_mean_fixed_seeds",
            "multiple_testing_trial_name": delta_trial_name,
            "raw_one_sided_p_value": raw_p_value,
            "holm_adjusted_one_sided_p_value": adjusted_p_value,
            "family_observed_mean_difference": family_mean_difference,
            "maximum_one_sided_p_value": 0.05,
            "pbo_eligible": pbo_eligible,
            "passed": (
                family_mean_difference > 0.0
                and adjusted_p_value <= 0.05
                and pbo_eligible
            ),
            "final_oos_opened": False,
        }
    )
    comparison["evidence_sha256"] = canonical_sha256(comparison)
    evidence["incumbent_comparison"] = comparison
    evidence["multiple_testing"] = multiple
    evidence["bundle_sha256"] = canonical_sha256(evidence)
    item["evidence_sha256"] = evidence["bundle_sha256"]
    if comparison["passed"] is not True:
        item["status"] = "failed"
        item["error"] = (
            f"quant joint {candidate_id} did not beat the frozen incumbent "
            "after the shared Holm/PBO correction"
        )
        return
    validate_quant_bundle_evidence(
        evidence,
        dataset_identity_sha256=dataset_identity_sha256,
    )


def _finalize_candidate_multiple_testing_batch(
    *,
    evaluations: list[dict[str, Any]],
    multiple: dict[str, Any],
    candidate_returns: dict[tuple[str, str], pd.Series],
    incumbent_returns: dict[str, pd.Series],
    dataset_identity_sha256: str,
) -> None:
    for item in evaluations:
        try:
            _finalize_candidate_multiple_testing(
                item=item,
                multiple=multiple,
                candidate_returns=candidate_returns,
                incumbent_returns=incumbent_returns,
                dataset_identity_sha256=dataset_identity_sha256,
            )
        except Exception as exc:
            if item.get("status") != "resource_blocked":
                item["status"] = "failed"
                item["error"] = (
                    "candidate-specific quant Holm/PBO evidence failed: "
                    f"{exc}"
                )


def _expression_values(
    data_api: Any,
    instruments: Any,
    expression: str,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    compiled = compile_qlib_expression(expression)
    frame = (
        data_api.features(
            instruments,
            [compiled.expression],
            start_time=start,
            end_time=end,
            freq="day",
        )
        .swaplevel()
        .sort_index()
    )
    frame.columns = ["factor"]
    return frame


def _expression_prefix_checks(
    data_api: Any,
    instruments: Any,
    expression: str,
    values: pd.DataFrame,
    *,
    start: str,
) -> dict[str, Any]:
    dates = pd.DatetimeIndex(values.index.get_level_values("datetime").unique()).sort_values()
    positions = sorted({len(dates) // 4, len(dates) // 2, (3 * len(dates)) // 4})
    checks: list[dict[str, Any]] = []
    for position in positions:
        cutoff = dates[min(position, len(dates) - 1)]
        prefix = _expression_values(
            data_api,
            instruments,
            expression,
            start=start,
            end=cutoff.date().isoformat(),
        )
        expected = values.loc[values.index.get_level_values("datetime") <= cutoff]
        prefix = require_exact_factor_index(
            prefix, expected, context="quant Qlib expression PIT prefix"
        )
        if not np.allclose(
            prefix.iloc[:, 0].to_numpy(dtype=float),
            expected.iloc[:, 0].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError("quant Qlib expression changes under a PIT prefix cutoff")
        checks.append({"cutoff": cutoff.date().isoformat(), "rows": len(prefix)})
    return {
        "contract_version": "qlib-expression-prefix-invariance-v1",
        "passed": True,
        "checks": checks,
    }


def _freeze_factor_values(
    *,
    bundle: dict[str, Any],
    view: Path,
    periods: dict[str, str],
    output: Path,
    universe: str,
) -> tuple[Path, list[dict[str, Any]]]:
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(view), region="cn")
    output.mkdir(parents=True, exist_ok=True)
    declared_fields = {
        str(field)
        for factor in bundle["factors"]
        for field in (factor.get("required_fields") or [])
    }
    unknown_fields = declared_fields - set(qlib_research_field_catalog())
    if unknown_fields:
        raise ValueError(
            "quant factor fields are outside the governed catalog: "
            + ", ".join(sorted(unknown_fields))
        )
    factor_input = normalize_factor_input(
        D.features(
            D.instruments(universe),
            [
                f"${field}"
                for field in sorted(
                    {"open", "close", "high", "low", "volume", "factor"}
                    | declared_fields
                )
            ],
            start_time=periods["train_start"],
            end_time=periods["valid_end"],
            freq="day",
        )
        .swaplevel()
        .sort_index()
    )
    input_path = output / "daily_pv.h5"
    factor_input.to_hdf(input_path, key="data", mode="w")
    values: list[pd.Series] = []
    evidence: list[dict[str, Any]] = []
    instruments = D.instruments(universe)
    for index, factor in enumerate(bundle["factors"], start=1):
        factor_id = str(factor["candidate_id"])
        root = output / "factors" / f"{index:03d}-{factor_id}"
        if factor.get("implementation_kind") == "qlib_expression":
            compiled = compile_qlib_expression(str(factor["expression"]))
            if set(compiled.required_fields) != set(factor.get("required_fields") or []):
                raise ValueError("quant factor expression fields changed after proposal")
            recomputed = _expression_values(
                D,
                instruments,
                compiled.expression,
                start=periods["train_start"],
                end=periods["valid_end"],
            )
            execution = {
                "executor_version": "qlib-expression-recompute-v1",
                "factor_definition_id": factor.get("factor_definition_id"),
                "expression_sha256": compiled.expression_sha256,
            }
            pit = _expression_prefix_checks(
                D,
                instruments,
                compiled.expression,
                recomputed,
                start=periods["train_start"],
            )
        else:
            recomputed, execution = execute_factor_code(
                code_path=Path(factor["code_path"]),
                input_path=input_path,
                workspace=root / "full",
                timeout_seconds=300,
            )
            pit = validate_factor_prefix_invariance(
                code_path=Path(factor["code_path"]),
                input_path=input_path,
                full_values=recomputed,
                workspace_root=root / "prefix-checks",
                cutpoint_count=3,
            )
        recomputed = require_exact_factor_index(
            recomputed,
            factor_input,
            context=f"quant bundle factor {factor_id}",
        )
        submitted_path = factor.get("submitted_values_path")
        submitted = compare_submitted_values(
            Path(submitted_path) if submitted_path else None,
            recomputed,
        )
        if submitted_path and not submitted.get("exact_match"):
            raise ValueError(f"quant factor {factor_id} submitted values do not recompute")
        values.append(recomputed.iloc[:, 0].rename(f"factor_{index:03d}"))
        item = {
            "candidate_id": factor_id,
            "code_sha256": factor["code_sha256"],
            "execution": execution,
            "pit_invariance": pit,
            "submitted_comparison": submitted,
        }
        item["evidence_sha256"] = canonical_sha256(item)
        evidence.append(item)
    frame = pd.concat(values, axis=1, join="inner").sort_index()
    if not frame.index.equals(factor_input.index):
        raise ValueError("quant bundle factor combination changed the immutable input index")
    path = output / "combined_factors.parquet"
    frame.to_parquet(path)
    return path, evidence


def _execute_single_model_cell(
    *,
    name: str,
    bundle: dict[str, Any],
    feature_set: dict[str, Any],
    config: dict[str, Any],
    execution_candidate_id: str,
    view: Path,
    factor_values_path: Path,
    periods: dict[str, Any],
    seed: int,
    resource_stage: str,
    workspace: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    use_factors = name in {"factor_only", "joint"}
    execution_manifest = {
        "candidate_id": execution_candidate_id,
        "code_sha256": config["code_sha256"],
        "model_type": config["model_type"],
        "model_engine": config["model_engine"],
        "training_hyperparameters": config["training_hyperparameters"],
        "resource_stage": resource_stage,
        "feature_set": feature_set,
        "additional_factor_count": len(bundle["factors"]) if use_factors else 0,
        "periods": periods,
        "seed": seed,
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "research_window_contract": manifest.get("research_window_contract"),
        "research_window_contract_sha256": manifest.get(
            "research_window_contract_sha256"
        ),
        "label_horizon_sessions": manifest.get("label_horizon_sessions"),
        "universe": manifest.get("universe", "cn_all"),
        "benchmark": manifest.get("benchmark", "SH000300"),
        "account": manifest.get("account", 100_000_000),
        "topk": manifest.get("topk", 50),
        "n_drop": manifest.get("n_drop", 5),
        "open_cost": manifest.get("open_cost", 0.0005),
        "close_cost": manifest.get("close_cost", 0.0015),
        "min_cost": manifest.get("min_cost", 5.0),
        "final_oos_opened": False,
    }
    try:
        model_result, execution = execute_model_candidate(
            code_path=config["code_path"],
            provider_path=view,
            additional_factors_path=factor_values_path if use_factors else None,
            manifest=execution_manifest,
            workspace=workspace,
            runner_path=Path(__file__).resolve().with_name("model_sandbox_runner.py"),
            timeout_seconds=int(manifest.get("model_timeout_seconds", 7200)),
        )
    except ModelResourceLimitError as exc:
        resource_evidence = _resource_limit_evidence(workspace=workspace, error=exc)
        return (
            {
                "status": "resource_blocked",
                "error": str(exc),
                "model_engine": config["model_engine"],
                "resource_stage": resource_stage,
                "resource_policy": resource_evidence["resource_policy"],
                "execution_evidence": resource_evidence,
                "execution_evidence_sha256": resource_evidence["evidence_sha256"],
                "execution_environment_sha256": resource_evidence[
                    "execution_environment_sha256"
                ],
            },
            str(resource_evidence["execution_environment_sha256"]),
        )
    days = _calendar_between(view, periods["valid_start"], periods["valid_end"])
    predictions_path = workspace / "output" / "predictions.parquet"
    coverage = verify_model_prediction_artifact(
        predictions_path,
        expected_sha256=model_result["predictions_sha256"],
        test_start=periods["valid_start"],
        test_end=periods["valid_end"],
        trading_days=days,
    )
    cell = {
        "status": "passed",
        "metrics": model_result["metrics"],
        "latest_prediction_date": model_result["latest_prediction_date"],
        "predictions_path": str(predictions_path),
        "predictions_sha256": model_result["predictions_sha256"],
        "checkpoint_path": str(
            workspace
            / "output"
            / governed_checkpoint_filename(str(config["model_engine"]))
        ),
        "checkpoint_sha256": model_result["checkpoint_sha256"],
        "checkpoint_format": model_result["checkpoint_format"],
        "portfolio_report_path": str(workspace / "output" / "portfolio_report.parquet"),
        "portfolio_report_sha256": model_result["portfolio_report_sha256"],
        "execution_evidence": execution,
        "execution_evidence_sha256": execution["evidence_sha256"],
        "execution_environment_sha256": execution[
            "execution_environment_sha256"
        ],
        "coverage": coverage,
        "model_engine": config["model_engine"],
        "resource_stage": resource_stage,
        "resource_policy": model_result["resource_policy"],
    }
    return cell, str(execution["execution_environment_sha256"])


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _normalized_labels(value: pd.DataFrame) -> pd.Series:
    if not isinstance(value.index, pd.MultiIndex) or set(value.index.names) != {
        "datetime",
        "instrument",
    }:
        raise ValueError("Qlib labels do not use the governed model index")
    frame = value.copy()
    if frame.index.names != ["datetime", "instrument"]:
        frame = frame.reorder_levels(["datetime", "instrument"])
    dates = pd.to_datetime(frame.index.get_level_values("datetime"), errors="coerce")
    if dates.isna().any():
        raise ValueError("Qlib labels contain invalid dates")
    frame.index = pd.MultiIndex.from_arrays(
        [dates.tz_localize(None), frame.index.get_level_values("instrument").astype(str)],
        names=["datetime", "instrument"],
    )
    return pd.to_numeric(frame.iloc[:, 0], errors="coerce").rename("label").sort_index()


_QLIB_PROVIDER: str | None = None


def _evaluate_equal_rank_portfolio(
    *,
    predictions: pd.DataFrame,
    view: Path,
    periods: dict[str, Any],
    workspace: Path,
    manifest: dict[str, Any],
    experiment_name: str,
) -> tuple[dict[str, Any], Path]:
    global _QLIB_PROVIDER

    import qlib
    from qlib.contrib.evaluate import risk_analysis
    from qlib.data import D
    from qlib.workflow.record_temp import PortAnaRecord

    provider_value = str(view.resolve())
    if _QLIB_PROVIDER != provider_value:
        qlib.init(provider_uri=provider_value, region="cn")
        _QLIB_PROVIDER = provider_value
    label_horizon_sessions = int(manifest.get("label_horizon_sessions") or 1)
    labels = _normalized_labels(
        D.features(
            instruments=str(manifest.get("universe") or "cn_all"),
            fields=[
                f"Ref($close, -{label_horizon_sessions + 1})/Ref($close, -1)-1"
            ],
            start_time=periods["valid_start"],
            end_time=periods["valid_end"],
            freq="day",
        )
    )
    aligned = pd.concat(
        [predictions["score"].rename("score"), labels], axis=1
    ).dropna()
    if aligned.empty:
        raise ValueError("ensemble factor-only predictions have no aligned labels")
    daily_ic = aligned.groupby(level="datetime").apply(
        lambda frame: frame["score"].corr(frame["label"]),
        include_groups=False,
    )
    daily_rank_ic = aligned.groupby(level="datetime").apply(
        lambda frame: frame["score"].corr(frame["label"], method="spearman"),
        include_groups=False,
    )
    with qlib_workflow_run(
        run_kind="quant-bundle-portfolio",
        run_id=experiment_name,
        tracking_uri=qlib_workflow_tracking_uri(),
        dataset_identity_sha256=str(manifest["dataset_identity_sha256"]),
    ) as workflow:
        recorder = workflow.get_recorder()
        recorder.save_objects(
            **{
                "pred.pkl": predictions[["score"]],
                "label.pkl": labels.to_frame("label"),
            }
        )
        record = PortAnaRecord(
            recorder,
            config={
                "strategy": {
                    "class": "TopkDropoutStrategy",
                    "module_path": "qlib.contrib.strategy",
                    "kwargs": {
                        "signal": "<PRED>",
                        "topk": int(manifest.get("topk", 50)),
                        "n_drop": int(manifest.get("n_drop", 5)),
                    },
                },
                "backtest": {
                    "start_time": periods["valid_start"],
                    "end_time": periods["valid_end"],
                    "account": float(manifest.get("account", 100_000_000)),
                    "benchmark": str(manifest.get("benchmark") or "SH000300"),
                    "exchange_kwargs": {
                        "freq": "day",
                        "limit_threshold": 0.095,
                        "deal_price": "close",
                        "open_cost": float(manifest.get("open_cost", 0.0005)),
                        "close_cost": float(manifest.get("close_cost", 0.0015)),
                        "min_cost": float(manifest.get("min_cost", 5.0)),
                    },
                },
            },
            risk_analysis_freq="day",
        )
        generated = record.generate()
        if not isinstance(generated, dict):
            raise RuntimeError("Qlib quant-bundle portfolio record generation was skipped")
        record.check(include_self=True, parents=False)
        report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    excess = report["return"] - report["bench"] - report["cost"]
    risk = risk_analysis(excess, freq="day")["risk"]
    metrics = {
        "ic": _finite(daily_ic.mean()),
        "icir": _finite(daily_ic.mean() / daily_ic.std()),
        "rank_ic": _finite(daily_rank_ic.mean()),
        "rank_icir": _finite(daily_rank_ic.mean() / daily_rank_ic.std()),
        "information_ratio": _finite(risk.get("information_ratio")),
        "annualized_excess_return_with_cost": _finite(risk.get("annualized_return")),
        "max_drawdown": _finite(risk.get("max_drawdown")),
        "total_cost": _finite(report["cost"].sum()),
        "average_turnover": _finite(
            report.get("turnover", pd.Series(dtype=float)).mean()
        ),
    }
    output = workspace / "output"
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "portfolio_report.parquet"
    report.to_parquet(report_path)
    aligned.to_parquet(output / "signals_and_labels.parquet")
    return metrics, report_path


def _execute_ensemble_factor_only_cell(
    *,
    bundle: dict[str, Any],
    view: Path,
    factor_values_path: Path,
    periods: dict[str, Any],
    seed: int,
    resource_stage: str,
    workspace: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    baseline = _runtime_baseline(bundle)
    if baseline.get("kind") != "ensemble":
        raise ValueError("ensemble factor-only dispatcher received another incumbent")
    members: list[tuple[str, pd.DataFrame]] = []
    member_artifacts: list[dict[str, Any]] = []
    environments: set[str] = set()
    for index, component in enumerate(baseline["components"], start=1):
        member_id = str(component["model_candidate_id"])
        model = dict(component["model"])
        config = {
            "code_path": Path(str(model["code_path"])),
            "code_sha256": str(model["code_sha256"]),
            "model_type": str(model.get("model_type") or "Tabular"),
            "model_engine": _frozen_candidate_model_engine(model),
            "training_hyperparameters": dict(
                model.get("training_hyperparameters") or {}
            ),
        }
        try:
            feature_set = resolve_feature_set(
                str(component["feature_set_id"]), dict(component["feature_set"])
            )
        except ValueError as exc:
            raise UnsupportedQuantBaseline(
                f"ensemble_member_feature_contract_unavailable:{member_id}"
            ) from exc
        if feature_set["definition_sha256"] != component.get(
            "feature_set_definition_sha256"
        ):
            raise UnsupportedQuantBaseline(
                f"ensemble_member_feature_contract_unavailable:{member_id}"
            )
        member_cell, environment_sha256 = _execute_single_model_cell(
            name="factor_only",
            bundle=bundle,
            feature_set=feature_set,
            config=config,
            execution_candidate_id=(
                f"{bundle['id']}-factor_only-member-{index:02d}-{member_id}"
            ),
            view=view,
            factor_values_path=factor_values_path,
            periods=periods,
            seed=seed,
            resource_stage=resource_stage,
            workspace=workspace / "members" / f"{index:02d}-{member_id}",
            manifest=manifest,
        )
        environments.add(environment_sha256)
        member_artifact = {
            "model_candidate_id": member_id,
            "model_family": str(component.get("model_family") or ""),
            "weight": float(component["weight"]),
            "feature_set_id": str(component["feature_set_id"]),
            "feature_set_definition_sha256": str(
                component["feature_set_definition_sha256"]
            ),
            **member_cell,
        }
        member_artifacts.append(member_artifact)
        if member_cell["status"] == "resource_blocked":
            aggregate = {
                "status": "resource_blocked",
                "error": (
                    f"ensemble member {member_id} exceeded the governed budget: "
                    f"{member_cell['error']}"
                ),
                "prediction_component_kind": "ensemble",
                "combiner": "equal_rank",
                "stacking": False,
                "member_artifacts": member_artifacts,
                "resource_stage": resource_stage,
                "resource_policy": {
                    "member_count": len(baseline["components"]),
                    "execution": "sequential_cpu_only",
                },
            }
            return aggregate, environment_sha256
        predictions_path = Path(str(member_cell["predictions_path"]))
        members.append((member_id, pd.read_parquet(predictions_path)))
    if len(environments) != 1:
        raise ValueError("ensemble factor-only members used different environments")
    combined, combination = equal_rank_predictions(members)
    metrics, report_path = _evaluate_equal_rank_portfolio(
        predictions=combined,
        view=view,
        periods=periods,
        workspace=workspace,
        manifest=manifest,
        experiment_name="quantlab-fin-quant-ensemble-factor-only",
    )
    predictions_path = workspace / "output" / "predictions.parquet"
    combined.to_parquet(predictions_path)
    predictions_sha256 = file_sha256(predictions_path)
    coverage = verify_model_prediction_artifact(
        predictions_path,
        expected_sha256=predictions_sha256,
        test_start=periods["valid_start"],
        test_end=periods["valid_end"],
        trading_days=_calendar_between(
            view, periods["valid_start"], periods["valid_end"]
        ),
    )
    execution = {
        "contract_version": "fin-quant-ensemble-member-retraining-cell-v1",
        "member_retraining_contract_version": baseline[
            "member_retraining_contract_version"
        ],
        "member_ids": [item[0] for item in members],
        "member_execution_evidence_sha256": {
            item["model_candidate_id"]: item["execution_evidence_sha256"]
            for item in member_artifacts
        },
        "combination_evidence": combination,
        "combiner": "equal_rank",
        "stacking": False,
        "execution": "sequential_cpu_only",
        "final_oos_opened": False,
    }
    execution["evidence_sha256"] = canonical_sha256(execution)
    environment_sha256 = next(iter(environments))
    return (
        {
            "status": "passed",
            "prediction_component_kind": "ensemble",
            "combiner": "equal_rank",
            "stacking": False,
            "metrics": metrics,
            "latest_prediction_date": periods["valid_end"],
            "predictions_path": str(predictions_path),
            "predictions_sha256": predictions_sha256,
            "portfolio_report_path": str(report_path),
            "portfolio_report_sha256": file_sha256(report_path),
            "execution_evidence": execution,
            "execution_evidence_sha256": execution["evidence_sha256"],
            "execution_environment_sha256": environment_sha256,
            "coverage": coverage,
            "combination_evidence": combination,
            "member_artifacts": member_artifacts,
            "resource_stage": resource_stage,
            "resource_policy": {
                "member_count": len(member_artifacts),
                "execution": "sequential_cpu_only",
            },
        },
        environment_sha256,
    )


def _execute_model_cell(
    *,
    name: str,
    bundle: dict[str, Any],
    feature_set: dict[str, Any],
    view: Path,
    factor_values_path: Path,
    periods: dict[str, Any],
    seed: int,
    resource_stage: str,
    workspace: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if name == "factor_only" and _runtime_baseline(bundle).get("kind") == "ensemble":
        return _execute_ensemble_factor_only_cell(
            bundle=bundle,
            view=view,
            factor_values_path=factor_values_path,
            periods=periods,
            seed=seed,
            resource_stage=resource_stage,
            workspace=workspace,
            manifest=manifest,
        )
    config = _ablation_model_config(name, bundle)
    return _execute_single_model_cell(
        name=name,
        bundle=bundle,
        feature_set=feature_set,
        config=config,
        execution_candidate_id=f"{bundle['id']}-{name}",
        view=view,
        factor_values_path=factor_values_path,
        periods=periods,
        seed=seed,
        resource_stage=resource_stage,
        workspace=workspace,
        manifest=manifest,
    )


def _run_resource_screen(
    *,
    bundle: dict[str, Any],
    feature_set: dict[str, Any],
    profiles: list[dict[str, Any]],
    view: Path,
    factor_values_path: Path,
    output: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    profile = next(
        item for item in profiles if str(item.get("id")) == "balanced_5y"
    )
    periods = dict(profile["periods"])
    seed = int(REQUIRED_MODEL_SEEDS[0])
    try:
        cell, environment_sha256 = _execute_model_cell(
            name="joint",
            bundle=bundle,
            feature_set=feature_set,
            view=view,
            factor_values_path=factor_values_path,
            periods=periods,
            seed=seed,
            resource_stage="screening",
            workspace=output / "resource-screen" / "balanced_5y" / f"seed-{seed}",
            manifest=manifest,
        )
    except Exception as exc:
        raise ValueError(
            f"quant bundle resource screen balanced_5y/seed-{seed} failed: {exc}"
        ) from exc
    return {
        **cell,
        "ablation": "joint",
        "profile_id": "balanced_5y",
        "seed": seed,
        "periods": periods,
        "execution_environment_sha256": environment_sha256,
        "date_segments_modified": False,
        "universe_modified": False,
    }


def _run_ablation(
    *,
    name: str,
    bundle: dict[str, Any],
    feature_set: dict[str, Any],
    profiles: list[dict[str, Any]],
    view: Path,
    factor_values_path: Path,
    output: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "passed",
        "source": "independent_qlib_recompute",
        "experiment_family_id": bundle["experiment_family_id"],
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "final_oos_opened": False,
        "profiles": {},
    }
    execution_environments: set[str] = set()
    for profile in profiles:
        profile_id = str(profile["id"])
        periods = dict(profile["periods"])
        seed_results: dict[str, Any] = {}
        resource_block: str | None = None
        for seed in REQUIRED_MODEL_SEEDS:
            workspace = output / "ablations" / name / profile_id / f"seed-{seed}"
            try:
                cell, environment_sha256 = _execute_model_cell(
                    name=name,
                    bundle=bundle,
                    feature_set=feature_set,
                    view=view,
                    factor_values_path=factor_values_path,
                    periods=periods,
                    seed=seed,
                    resource_stage="full_validation",
                    workspace=workspace,
                    manifest=manifest,
                )
            except UnsupportedQuantBaseline:
                raise
            except Exception as exc:
                raise ValueError(
                    f"quant ablation {name}/{profile_id}/seed-{seed} failed: {exc}"
                ) from exc
            seed_results[str(seed)] = cell
            execution_environments.add(environment_sha256)
            if cell["status"] == "resource_blocked":
                resource_block = f"{name}/{profile_id}/seed-{seed}: {cell['error']}"
                break
        result["profiles"][profile_id] = {
            "periods": periods,
            "seeds": seed_results,
        }
        if resource_block is not None:
            result["status"] = "resource_blocked"
            result["reason_code"] = "full_validation_resource_limit"
            result["error"] = resource_block
            break
    if len(execution_environments) != 1:
        raise ValueError(f"quant ablation {name} used inconsistent execution environments")
    result["execution_environment_sha256"] = next(iter(execution_environments))
    result["evidence_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    provider = Path(args.provider_uri).resolve()
    manifest: dict[str, Any] = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    raw_label_binding = manifest.get("research_label_binding")
    label_binding = (
        validate_research_label_binding(raw_label_binding)
        if raw_label_binding is not None
        else None
    )
    if label_binding is not None:
        if (
            manifest.get("research_label_binding_sha256")
            != label_binding["binding_sha256"]
            or manifest.get("research_window_contract")
            != label_binding["research_window_contract"]
            or manifest.get("research_window_contract_sha256")
            != label_binding["research_window_contract_sha256"]
            or manifest.get("dataset_identity_sha256")
            != label_binding["dataset_identity_sha256"]
            or manifest.get("periods") != label_binding["periods"]
            or int(manifest.get("label_horizon_sessions") or 0)
            != int(label_binding["label_horizon_sessions"])
            or any(
                candidate.get("research_label_binding") != label_binding
                or candidate.get("research_label_binding_sha256")
                != label_binding["binding_sha256"]
                for candidate in manifest.get("candidates") or []
            )
        ):
            raise ValueError("quant evaluation label binding changed in transit")
    provenance = json.loads((provider / "metadata" / "provenance.json").read_text(encoding="utf-8"))
    verify_qlib_output_manifest(provider, provenance)
    if provenance.get("dataset_identity_sha256") != manifest.get("dataset_identity_sha256"):
        raise ValueError("quant evaluation provider does not match the sealed dataset")
    research_trial_ids = dict(manifest.get("research_trial_ids") or {})
    expected_candidate_ids = {
        str(item.get("id") or "") for item in manifest.get("candidates") or []
    }
    if (
        not str(manifest.get("research_tournament_id") or "")
        or not str(manifest.get("parent_research_tournament_id") or "")
        or not is_sha256(manifest.get("research_tournament_manifest_sha256"))
        or set(research_trial_ids) != expected_candidate_ids
        or any(not str(value or "") for value in research_trial_ids.values())
        or manifest.get("research_screening_only") is not True
        or manifest.get("not_capital_confirmation") is not True
        or manifest.get("cross_cycle_fwer_claimed") is not False
        or manifest.get("final_oos_opened") is not False
    ):
        raise ValueError("fin_quant research trial ledger binding is invalid")
    profiles = manifest.get("evaluation_profiles") or []
    if {str(item.get("id")) for item in profiles} != {
        "recent_3y",
        "balanced_5y",
        "robust_10y",
    }:
        raise ValueError("quant evaluation requires the three governed profiles")
    valid_ends = {str(item["periods"]["valid_end"]) for item in profiles}
    if len(valid_ends) != 1:
        raise ValueError("quant profiles must share one pre-final cutoff")
    output = Path(args.output).resolve()
    root = output.parent / "independent-quant-evaluations"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    view = prepare_rdagent_dataset_view(
        provider,
        output.parent / "quant-dataset-view",
        cutoff=next(iter(valid_ends)),
    )
    evaluations: list[dict[str, Any]] = []
    for candidate in manifest.get("candidates") or []:
        candidate_root = root / str(candidate["id"])
        try:
            feature_set_id = str(
                candidate.get("feature_set_id")
                or manifest.get("feature_set_id")
                or "governed-baseline"
            )
            feature_set = resolve_feature_set(
                feature_set_id,
                candidate.get("feature_set") or manifest.get("feature_set"),
            )
            horizon_factor_bundle = None
            if label_binding is not None:
                horizon_factor_bundle = validate_horizon_factor_bundle(
                    candidate.get("horizon_factor_bundle") or {}
                )
                if (
                    candidate.get("horizon_factor_bundle_sha256")
                    != horizon_factor_bundle["bundle_sha256"]
                    or horizon_factor_bundle["horizon_profile"]
                    != label_binding["horizon_profile"]
                    or horizon_factor_bundle["label_horizon_sessions"]
                    != label_binding["label_horizon_sessions"]
                    or horizon_factor_bundle["research_label_binding_sha256"]
                    != label_binding["binding_sha256"]
                    or horizon_factor_bundle["base_feature_set"]["id"]
                    != feature_set["id"]
                    or horizon_factor_bundle["base_feature_set"][
                        "definition_sha256"
                    ]
                    != feature_set["definition_sha256"]
                    or horizon_factor_bundle["incremental_factors"]
                    != [
                        {
                            "candidate_id": str(factor.get("candidate_id") or ""),
                            "code_sha256": str(factor.get("code_sha256") or ""),
                        }
                        for factor in candidate.get("factors") or []
                    ]
                ):
                    raise ValueError(
                        "quant evaluation changed the horizon factor bundle"
                    )
            elif (
                candidate.get("horizon_factor_bundle") is not None
                or candidate.get("horizon_factor_bundle_sha256") is not None
            ):
                raise ValueError(
                    "legacy quant evaluation cannot claim a horizon factor bundle"
                )
            baseline = _validate_baseline_prediction(
                baseline=candidate.get("baseline_prediction_champion"),
                runtime_baseline=candidate.get("baseline_prediction_runtime"),
                manifest_baseline=manifest.get("baseline_prediction_champion"),
                dataset_identity_sha256=str(manifest["dataset_identity_sha256"]),
                feature_set_definition_sha256=str(feature_set["definition_sha256"]),
            )
            factor_periods = {
                "train_start": min(
                    str(profile["periods"]["train_start"]) for profile in profiles
                ),
                "valid_end": max(
                    str(profile["periods"]["valid_end"]) for profile in profiles
                ),
            }
            factor_values, factor_evidence = _freeze_factor_values(
                bundle=candidate,
                view=view,
                periods=factor_periods,
                output=candidate_root,
                universe=str(manifest.get("universe") or "cn_all"),
            )
            resource_screen = _run_resource_screen(
                bundle=candidate,
                feature_set=feature_set,
                profiles=profiles,
                view=view,
                factor_values_path=factor_values,
                output=candidate_root,
                manifest=manifest,
            )
            ablations: dict[str, dict[str, Any]] = {}
            resource_block: str | None = None
            if resource_screen["status"] != "resource_blocked":
                for name in REQUIRED_QUANT_ABLATIONS:
                    ablation = _run_ablation(
                        name=name,
                        bundle=candidate,
                        feature_set=feature_set,
                        profiles=profiles,
                        view=view,
                        factor_values_path=factor_values,
                        output=candidate_root,
                        manifest=manifest,
                    )
                    ablations[name] = ablation
                    if ablation["status"] == "resource_blocked":
                        resource_block = str(ablation["error"])
                        break
            else:
                resource_block = str(resource_screen["error"])
            execution_environments = {
                str(resource_screen["execution_environment_sha256"]),
                *(
                    str(item["execution_environment_sha256"])
                    for item in ablations.values()
                ),
            }
            if len(execution_environments) != 1:
                raise ValueError(
                    "quant screening and ablations used inconsistent execution environments"
                )
            model_engine = _frozen_candidate_model_engine(candidate["model"])
            evidence = {
                "contract_version": QUANT_BUNDLE_CONTRACT_VERSION,
                "id": candidate["id"],
                "dataset_identity_sha256": manifest["dataset_identity_sha256"],
                **(
                    {
                        "horizon_profile": label_binding["horizon_profile"],
                        "label_horizon_sessions": label_binding[
                            "label_horizon_sessions"
                        ],
                        "research_label_binding_sha256": label_binding[
                            "binding_sha256"
                        ],
                        "research_window_contract_sha256": label_binding[
                            "research_window_contract_sha256"
                        ],
                        "horizon_factor_bundle": horizon_factor_bundle,
                        "horizon_factor_bundle_sha256": horizon_factor_bundle[
                            "bundle_sha256"
                        ],
                    }
                    if label_binding is not None
                    else {}
                ),
                "feature_set_definition_sha256": feature_set["definition_sha256"],
                "experiment_family_id": candidate["experiment_family_id"],
                "baseline_prediction_champion": baseline,
                "baseline_prediction_champion_sha256": baseline[
                    "evidence_sha256"
                ],
                "factors": [
                    {
                        "candidate_id": factor["candidate_id"],
                        "code_sha256": factor["code_sha256"],
                    }
                    for factor in candidate["factors"]
                ],
                "factor_recompute_evidence": factor_evidence,
                "combined_factor_values_sha256": sha256_file(factor_values),
                "model": {
                    "code_sha256": candidate["model"]["code_sha256"],
                    "recipe_sha256": candidate["model"]["recipe_sha256"],
                    "model_engine": model_engine,
                },
                "resource_screen": resource_screen,
                "ablations": ablations,
                "execution_environment_sha256": next(
                    iter(execution_environments)
                ),
                "final_oos_opened": False,
            }
            if resource_block is not None:
                evaluations.append(
                    {
                        "candidate_id": candidate["id"],
                        "status": "resource_blocked",
                        "reason_code": (
                            "screening_resource_limit"
                            if resource_screen["status"] == "resource_blocked"
                            else "full_validation_resource_limit"
                        ),
                        "error": resource_block,
                        "evidence": evidence,
                    }
                )
                continue
            evaluations.append(
                {
                    "candidate_id": candidate["id"],
                    "status": "passed",
                    "evidence": evidence,
                }
            )
        except UnsupportedQuantBaseline as exc:
            evaluations.append(
                {
                    "candidate_id": candidate.get("id"),
                    "status": "resource_blocked",
                    "reason_code": "ensemble_member_feature_contract_unavailable",
                    "error": str(exc),
                    "evidence": {
                        "contract_version": "quant-baseline-resource-block-v1",
                        "candidate_id": candidate.get("id"),
                        "reason_code": (
                            "ensemble_member_feature_contract_unavailable"
                        ),
                        "final_oos_opened": False,
                    },
                }
            )
        except Exception as exc:
            evaluations.append(
                {"candidate_id": candidate.get("id"), "status": "failed", "error": str(exc)}
            )
    multiple: dict[str, Any] | None = None
    try:
        expected_candidates = manifest.get("candidates") or []
        if len(evaluations) != len(expected_candidates):
            raise ValueError(
                "the pre-registered quant run has an incomplete candidate count: "
                f"expected {len(expected_candidates)}, received {len(evaluations)}"
            )
        trial_series: list[tuple[dict[str, str], pd.Series]] = []
        forced_raw_p_values: dict[str, float] = {}
        candidate_returns: dict[tuple[str, str], pd.Series] = {}
        incumbent_returns: dict[str, pd.Series] = {}
        candidate_baseline_ids: dict[str, str] = {}
        baseline_errors: dict[str, str] = {}
        candidates_by_id = {
            str(value.get("id") or ""): value for value in expected_candidates
        }
        evaluations_by_id = {
            str(value.get("candidate_id") or ""): value for value in evaluations
        }

        # Verify all frozen incumbents first.  Their real return index is the
        # only admissible basis for a failed candidate's p=1 placeholder.
        for candidate_id, candidate in candidates_by_id.items():
            frozen_baseline = dict(candidate.get("baseline_prediction_champion") or {})
            baseline_id = str(frozen_baseline.get("candidate_id") or "")
            if not baseline_id:
                baseline_errors[candidate_id] = "frozen incumbent identity is missing"
                continue
            candidate_baseline_ids[candidate_id] = baseline_id
            if baseline_id in incumbent_returns:
                continue
            try:
                incumbent_returns[baseline_id] = _frozen_incumbent_return_series(
                    _runtime_baseline(candidate)
                )
            except Exception as exc:
                baseline_errors[candidate_id] = str(exc)

        if not incumbent_returns:
            raise ValueError(
                "no verified frozen incumbent return series is available for the "
                "pre-registered fin_quant family"
            )
        reference_returns = next(iter(incumbent_returns.values()))
        zero_placeholder = pd.Series(0.0, index=reference_returns.index)

        baseline_ids = sorted(set(candidate_baseline_ids.values()))
        for baseline_id in baseline_ids:
            trial_name = f"incumbent:{baseline_id}"
            returns = incumbent_returns.get(baseline_id)
            if returns is None:
                returns = zero_placeholder
                forced_raw_p_values[trial_name] = 1.0
            trial_series.append(
                (
                    {
                        "name": trial_name,
                        "candidate_id": baseline_id,
                        "kind": "fin_quant_incumbent",
                        "ablation": "incumbent",
                    },
                    returns,
                )
            )

        for candidate_id in candidates_by_id:
            item = evaluations_by_id[candidate_id]
            baseline_id = candidate_baseline_ids.get(candidate_id, "")
            candidate_failed = item.get("status") != "passed" or bool(
                baseline_errors.get(candidate_id)
            )
            candidate_error = str(item.get("error") or "")
            if baseline_errors.get(candidate_id):
                candidate_error = (
                    "frozen incumbent evidence failed: " + baseline_errors[candidate_id]
                )
            if not candidate_failed:
                try:
                    for ablation in REQUIRED_QUANT_ABLATIONS:
                        seed_returns: list[pd.Series] = []
                        seeds = item["evidence"]["ablations"][ablation]["profiles"][
                            "recent_3y"
                        ]["seeds"]
                        for seed in REQUIRED_MODEL_SEEDS:
                            seed_result = seeds[str(seed)]
                            report_path = Path(
                                str(seed_result["portfolio_report_path"])
                            )
                            if (
                                not report_path.is_file()
                                or file_sha256(report_path)
                                != seed_result["portfolio_report_sha256"]
                            ):
                                raise ValueError(
                                    "portfolio report changed for "
                                    f"{ablation}/seed-{seed}"
                                )
                            report = pd.read_parquet(report_path)
                            if not {"return", "bench", "cost"}.issubset(
                                report.columns
                            ):
                                raise ValueError(
                                    "portfolio report is incomplete for "
                                    f"{ablation}/seed-{seed}"
                                )
                            series = (
                                pd.to_numeric(report["return"], errors="coerce")
                                - pd.to_numeric(report["bench"], errors="coerce")
                                - pd.to_numeric(report["cost"], errors="coerce")
                            ).rename(str(seed))
                            series.index = pd.to_datetime(series.index).tz_localize(
                                None
                            )
                            seed_returns.append(series)
                        frame = pd.concat(
                            seed_returns, axis=1, join="inner"
                        ).dropna()
                        if (
                            frame.shape[1] != len(REQUIRED_MODEL_SEEDS)
                            or len(frame) < 40
                        ):
                            raise ValueError(
                                f"seed return matrix is incomplete for {ablation}"
                            )
                        candidate_returns[(candidate_id, ablation)] = frame.mean(
                            axis=1
                        )
                except Exception as exc:
                    candidate_failed = True
                    candidate_error = f"quant run-level evidence failed: {exc}"

            if candidate_failed:
                if item.get("status") != "resource_blocked":
                    item["status"] = "failed"
                    item["error"] = candidate_error or "quant evaluation failed"
                for ablation in REQUIRED_QUANT_ABLATIONS:
                    candidate_returns[(candidate_id, ablation)] = zero_placeholder

            for ablation in REQUIRED_QUANT_ABLATIONS:
                trial_name = f"{candidate_id}:{ablation}"
                trial_series.append(
                    (
                        {
                            "name": trial_name,
                            "candidate_id": candidate_id,
                            "kind": "quant_bundle",
                            "ablation": ablation,
                        },
                        candidate_returns[(candidate_id, ablation)],
                    )
                )
                if candidate_failed:
                    forced_raw_p_values[trial_name] = 1.0

            delta_trial_name = f"{candidate_id}:joint_vs_incumbent"
            if candidate_failed or baseline_id not in incumbent_returns:
                delta_returns = zero_placeholder
                forced_raw_p_values[delta_trial_name] = 1.0
            else:
                aligned = pd.concat(
                    [
                        candidate_returns[(candidate_id, "joint")].rename("joint"),
                        incumbent_returns[baseline_id].rename("incumbent"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                if len(aligned) < 40:
                    item["status"] = "failed"
                    item["error"] = "joint/incumbent paired return matrix is incomplete"
                    for ablation in REQUIRED_QUANT_ABLATIONS:
                        forced_raw_p_values[f"{candidate_id}:{ablation}"] = 1.0
                    delta_returns = zero_placeholder
                    forced_raw_p_values[delta_trial_name] = 1.0
                else:
                    delta_returns = aligned["joint"] - aligned["incumbent"]
            trial_series.append(
                (
                    {
                        "name": delta_trial_name,
                        "candidate_id": candidate_id,
                        "kind": "fin_quant_joint_delta",
                        "ablation": "joint_vs_incumbent",
                    },
                    delta_returns,
                )
            )

        if evaluations:
            multiple = build_run_multiple_testing_evidence(
                research_run_id=str(manifest["research_run_id"]),
                trial_series=trial_series,
                output=root / "run-level-multiple-testing",
                forced_raw_p_values=forced_raw_p_values,
            )
            _finalize_candidate_multiple_testing_batch(
                evaluations=evaluations,
                multiple=multiple,
                candidate_returns=candidate_returns,
                incumbent_returns=incumbent_returns,
                dataset_identity_sha256=str(manifest["dataset_identity_sha256"]),
            )
    except Exception as exc:
        error = f"run-level quant Holm/PBO gate failed: {exc}"
        evaluations = [
            item
            if item.get("status") == "resource_blocked"
            else {
                "candidate_id": str(item.get("candidate_id") or ""),
                "status": "failed",
                "error": error,
            }
            for item in evaluations
        ]
    receipt = {
        "contract_version": "fin-quant-research-ledger-receipt-v1",
        "research_tournament_id": str(manifest["research_tournament_id"]),
        "parent_research_tournament_id": str(
            manifest["parent_research_tournament_id"]
        ),
        "research_tournament_manifest_sha256": str(
            manifest["research_tournament_manifest_sha256"]
        ),
        "research_trial_ids": research_trial_ids,
        "candidate_statuses": {
            str(item.get("candidate_id") or ""): str(item.get("status") or "")
            for item in evaluations
        },
        "run_multiple_testing_evidence_sha256": (
            str(multiple.get("evidence_sha256") or "") if multiple else None
        ),
        "failed_and_rejected_trials_retained": True,
        "failed_candidate_raw_p_value": 1.0,
        "research_screening_only": True,
        "not_capital_confirmation": True,
        "cross_cycle_fwer_claimed": False,
        "final_oos_opened": False,
        "research_label_binding_sha256": (
            label_binding["binding_sha256"] if label_binding is not None else None
        ),
    }
    receipt["evidence_sha256"] = canonical_sha256(receipt)
    result = {
        "status": "ok",
        "evaluations": evaluations,
        "multiple_testing": multiple,
        "research_trial_ledger_receipt": receipt,
        "research_screening_only": True,
        "not_capital_confirmation": True,
        "cross_cycle_fwer_claimed": False,
        "final_oos_opened": False,
        "resource_blocked_count": sum(
            item.get("status") == "resource_blocked" for item in evaluations
        ),
        **(
            {
                "research_label_binding": label_binding,
                "research_label_binding_sha256": label_binding[
                    "binding_sha256"
                ],
            }
            if label_binding is not None
            else {}
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
