from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import select

from quant_data.config import Settings
from quant_data.database import (
    model_candidates,
    open_database,
    research_run_artifacts,
    research_runs,
)

from .feature_set_registry import get_feature_set
from .job_store import JobStore
from .rdagent_candidate_store import RDAGentCandidateStore
from .research_horizon import primary_label_policy_contract
from .research_label_binding import resolve_research_label_binding
from .research_store import ResearchStore
from .research_tournament import MODEL_FAMILIES


def _safe_scope_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


class PlatformModelTournamentService:
    """Register the platform model baselines in the normal independent gate.

    These candidates do not receive privileged scores.  They are immutable
    model candidates attached to a dedicated research run and are evaluated by
    the exact same ``model_evaluate`` worker used for RD-Agent proposals.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = open_database(settings.database_url)
        self.jobs = JobStore(settings.database_url)
        self.research = ResearchStore(settings.database_url)
        self.candidates = RDAGentCandidateStore(settings.database_url)
        self.project_root = Path(__file__).resolve().parents[2]

    def ensure_lane(
        self,
        *,
        cycle_id: str,
        tournament_id: str,
        dataset: Mapping[str, Any],
        stage: str,
        feature_set_id: str,
        periods: Mapping[str, str],
        evaluation_profiles: Sequence[Mapping[str, Any]],
        horizon_profile: str,
        label_horizon_sessions: int,
        primary_label_policy: Mapping[str, Any],
        research_window_contract: Mapping[str, Any],
        research_window_contract_sha256: str,
        trials: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if stage not in {"feature_screen", "model_full"}:
            raise ValueError("platform model tournament stage is invalid")
        if not trials:
            raise ValueError("platform model lane has no preregistered trials")
        feature_set = get_feature_set(feature_set_id)
        stage_profiles = [
            dict(item)
            for item in evaluation_profiles
            if stage != "feature_screen" or str(item.get("id") or "") == "recent_3y"
        ]
        expected_profile_count = 1 if stage == "feature_screen" else 3
        if len(stage_profiles) != expected_profile_count:
            raise ValueError("platform model tournament profile contract is incomplete")
        identity = str((dataset.get("provenance") or {}).get("dataset_identity_sha256") or "")
        lineage_id = str(dataset.get("lineage_id") or "")
        if len(identity) != 64 or len(lineage_id) != 64:
            raise ValueError("platform model tournament requires sealed dataset lineage")
        policy = primary_label_policy_contract()
        if dict(primary_label_policy) != policy:
            raise ValueError("platform model tournament primary-label policy changed")
        label_binding = resolve_research_label_binding(
            {
                "horizon_profile": horizon_profile,
                "dataset": str(dataset["name"]),
                "dataset_identity_sha256": identity,
                "periods": dict(periods),
                "feature_set": feature_set,
                "research_window_contract": dict(research_window_contract),
                "research_window_contract_sha256": research_window_contract_sha256,
                "label_horizon_sessions": label_horizon_sessions,
            }
        )
        if label_binding is None:
            raise ValueError("platform model tournament has no active label binding")
        requested_by = f"autopilot:{cycle_id}"
        run_kind = (
            f"platform_model_{stage}_{_safe_scope_token(feature_set_id)}_"
            f"{horizon_profile}"
        )
        with self.engine.connect() as connection:
            existing_run = connection.execute(
                select(research_runs).where(
                    research_runs.c.kind == run_kind,
                    research_runs.c.dataset == str(dataset["name"]),
                    research_runs.c.requested_by == requested_by,
                )
            ).first()
        if existing_run is None:
            run = self.research.create_run(
                kind=run_kind,
                objective=(
                    "Evaluate preregistered CPU model baselines on the frozen "
                    f"feature set {feature_set_id}."
                ),
                dataset=str(dataset["name"]),
                requested_by=requested_by,
                budget={
                    "cpu_only": True,
                    "qlib_evaluation_concurrency_cap": 3,
                    "service_resource_reserve": 0.25,
                },
                config={
                    "contract_version": "platform-model-tournament-run-v2-horizon",
                    "autopilot_cycle_id": cycle_id,
                    "tournament_stage": stage,
                    "feature_set_id": feature_set_id,
                    "feature_set_definition_sha256": feature_set["definition_sha256"],
                    "dataset_identity_sha256": identity,
                    "periods": dict(periods),
                    "evaluation_profiles": stage_profiles,
                    "horizon_profile": horizon_profile,
                    "label_horizon_sessions": label_horizon_sessions,
                    "research_window_contract": dict(research_window_contract),
                    "research_window_contract_sha256": research_window_contract_sha256,
                    "research_label_binding": label_binding,
                    "research_label_binding_sha256": label_binding["binding_sha256"],
                    "primary_label_policy": policy,
                    "primary_label_policy_sha256": policy["policy_sha256"],
                },
                artifact_path=self.settings.data_root / "artifacts" / "model-tournaments",
            )
        else:
            run = self.research.get_run(str(existing_run.id))
            config = dict(existing_run.config_json or {})
            if (
                config.get("contract_version")
                != "platform-model-tournament-run-v2-horizon"
                or config.get("autopilot_cycle_id") != cycle_id
                or config.get("tournament_stage") != stage
                or config.get("feature_set_definition_sha256")
                != feature_set["definition_sha256"]
                or config.get("dataset_identity_sha256") != identity
                or config.get("periods") != dict(periods)
                or config.get("evaluation_profiles") != stage_profiles
                or config.get("research_label_binding") != label_binding
                or config.get("primary_label_policy") != policy
            ):
                raise ValueError("existing platform model tournament run changed contract")
            if str(existing_run.status) in {"failed", "cancelled"}:
                raise ValueError("failed platform model tournament requires explicit retry")

        stub_path = self.project_root / "scripts" / "baseline_model_stub.py"
        if not stub_path.is_file():
            raise ValueError("governed platform model stub is unavailable")
        with self.engine.connect() as connection:
            artifact_row = connection.execute(
                select(research_run_artifacts).where(
                    research_run_artifacts.c.research_run_id == run["id"],
                    research_run_artifacts.c.artifact_type
                    == "platform_model_baseline_stub",
                )
            ).first()
        if artifact_row is None:
            artifact = self.candidates.register_run_artifact(
                research_run_id=str(run["id"]),
                artifact_type="platform_model_baseline_stub",
                storage_path=stub_path,
                producer="quantlab",
                actor="autopilot",
                contract_version="platform-model-baseline-stub-v1",
                metadata={"feature_set_id": feature_set_id},
            )
        else:
            artifact = self.candidates.get_run_artifact(str(artifact_row.id), verify=True)

        with self.engine.connect() as connection:
            existing_candidates = {
                str(item.name): item
                for item in connection.execute(
                    select(model_candidates).where(
                        model_candidates.c.research_run_id == run["id"]
                    )
                )
            }
        candidate_payloads: list[dict[str, Any]] = []
        bindings: list[dict[str, str]] = []
        expected_names: set[str] = set()
        for trial in trials:
            family = str(trial.get("model_family") or "")
            model_spec = MODEL_FAMILIES.get(family)
            if model_spec is None:
                raise ValueError(f"unknown platform model family {family!r}")
            trial_feature_set = str(trial.get("feature_set_id") or "")
            if trial_feature_set != feature_set_id:
                raise ValueError("platform model trial belongs to another feature lane")
            name = f"platform-{stage}-{feature_set_id}-{family}"
            expected_names.add(name)
            row = existing_candidates.get(name)
            model_type = "TimeSeries" if family in {"gru", "transformer"} else "Tabular"
            engine = str(model_spec["engine"])
            training_hyperparameters = {
                "n_epochs": 12,
                "early_stop": 3,
                "batch_size": 256,
            } if model_type == "TimeSeries" else {}
            if row is None:
                candidate = self.candidates.create_model_candidate(
                    research_run_id=str(run["id"]),
                    name=name,
                    description=(
                        f"Preregistered {family} CPU baseline on {feature_set_id}; "
                        "no RD-Agent score or final OOS is used."
                    ),
                    model_type=model_type,
                    code_artifact_id=str(artifact["id"]),
                    architecture={
                        key: value for key, value in model_spec.items() if key != "engine"
                    },
                    model_hyperparameters={
                        "model_engine": engine,
                        "governed_spec": dict(model_spec),
                    },
                    training_hyperparameters=training_hyperparameters,
                    feature_set_id=feature_set_id,
                    dataset=str(dataset["name"]),
                    dataset_identity_sha256=identity,
                    dataset_lineage_id=lineage_id,
                    pre_final_end=date.fromisoformat(str(periods["valid_end"])),
                    final_oos_start=date.fromisoformat(str(periods["test_start"])),
                    final_oos_end=date.fromisoformat(str(periods["test_end"])),
                    research_label_binding=label_binding,
                    rdagent_decision=None,
                    rdagent_feedback=None,
                )
                candidate_id = str(candidate["id"])
                code_sha256 = str(candidate["code_sha256"])
            else:
                candidate = self.candidates.get_model_candidate(str(row.id), verify=True)
                recipe = dict((candidate.get("manifest_json") or {}).get("recipe") or {})
                if (
                    str(row.model_type) != model_type
                    or (recipe.get("model_hyperparameters") or {}).get("model_engine")
                    != engine
                    or str(row.feature_set_definition_sha256)
                    != str(feature_set["definition_sha256"])
                    or str(row.dataset_identity_sha256) != identity
                ):
                    raise ValueError("existing platform model candidate changed contract")
                candidate_id = str(row.id)
                code_sha256 = str(row.code_sha256)
            candidate_payloads.append(
                {
                    "id": candidate_id,
                    "code_path": str(stub_path.resolve()),
                    "code_sha256": code_sha256,
                    "model_type": model_type,
                    "model_engine": engine,
                    "training_hyperparameters": training_hyperparameters,
                }
            )
            bindings.append(
                {
                    "candidate_id": candidate_id,
                    "model_family": family,
                    "trial_id": str(trial["id"]),
                }
            )
        if set(existing_candidates) - expected_names:
            raise ValueError("platform model tournament run contains unregistered candidates")

        if run.get("job_id"):
            job = self.jobs.get(str(run["job_id"]))
        else:
            payload = {
                "research_run_id": str(run["id"]),
                "dataset": str(dataset["name"]),
                "dataset_path": str(dataset["path"]),
                "dataset_identity_sha256": identity,
                "evaluation_stage": stage,
                "evaluation_profiles": stage_profiles,
                "feature_set_id": feature_set_id,
                "feature_set_definition_sha256": feature_set["definition_sha256"],
                "feature_set": feature_set,
                "candidates": candidate_payloads,
                "research_tournament_id": tournament_id,
                "candidate_bindings": bindings,
                "universe": "cn_all",
                "benchmark": "SH000300",
                "horizon_profile": horizon_profile,
                "label_horizon_sessions": label_horizon_sessions,
                "research_window_contract": dict(research_window_contract),
                "research_window_contract_sha256": research_window_contract_sha256,
                "research_label_binding": label_binding,
                "research_label_binding_sha256": label_binding["binding_sha256"],
                "primary_label_policy": policy,
                "primary_label_policy_sha256": policy["policy_sha256"],
            }
            job = self.jobs.create(
                "model_evaluate",
                payload,
                self.settings.data_root
                / "platform"
                / "logs"
                / (
                    f"platform-model-{stage}-{cycle_id}-"
                    f"{_safe_scope_token(feature_set_id)}.log"
                ),
                dedupe_active_kind=False,
                idempotency_key=(
                    f"platform-model:{cycle_id}:{stage}:{feature_set_id}"
                ),
            )
            self.research.attach_job(str(run["id"]), str(job["id"]))
        return {
            "run": self.research.get_run(str(run["id"])),
            "job": job,
            "bindings": bindings,
            "stage": stage,
            "feature_set_id": feature_set_id,
        }

    def ensure_champion_revalidation_lane(
        self,
        *,
        cycle_id: str,
        tournament_id: str,
        dataset: Mapping[str, Any],
        periods: Mapping[str, str],
        evaluation_profiles: Sequence[Mapping[str, Any]],
        horizon_profile: str,
        label_horizon_sessions: int,
        primary_label_policy: Mapping[str, Any],
        research_window_contract: Mapping[str, Any],
        research_window_contract_sha256: str,
        source_components: Sequence[Mapping[str, Any]],
        trials: Sequence[Mapping[str, Any]],
        source_manifest_sha256: str,
    ) -> dict[str, Any]:
        """Retrain an already frozen winner on one new Qlib identity.

        This is deliberately not another tournament: callers provide every
        source component and its exact recipe hash.  The only mutable values
        are current-vintage data boundaries and the new immutable candidate
        IDs generated for that evaluation.
        """

        identity = str((dataset.get("provenance") or {}).get("dataset_identity_sha256") or "")
        lineage_id = str(dataset.get("lineage_id") or "")
        if len(identity) != 64 or len(lineage_id) != 64:
            raise ValueError("champion revalidation requires sealed dataset lineage")
        policy = primary_label_policy_contract()
        if dict(primary_label_policy) != policy:
            raise ValueError("champion revalidation primary-label policy changed")
        if len(source_components) != len(trials) or not source_components:
            raise ValueError("champion revalidation components and trials differ")
        profiles = [dict(item) for item in evaluation_profiles]
        if {str(item.get("id") or "") for item in profiles} != {
            "recent_3y",
            "balanced_5y",
            "robust_10y",
        }:
            raise ValueError("champion revalidation requires the full three-window grid")
        source_ids = sorted(
            str(item.get("source_model_candidate_id") or "")
            for item in source_components
        )
        if not all(source_ids):
            raise ValueError("champion revalidation source model identity is missing")
        lane_token = _safe_scope_token(",".join(source_ids))
        requested_by = f"autopilot:{cycle_id}"
        # A frozen ensemble may deliberately contain models trained on
        # different feature sets.  Qlib's evaluator accepts one feature set
        # per job, so each same-feature group is its own immutable lane while
        # all lanes remain bound to one revalidation tournament.
        feature_ids = {str(item["feature_set_id"]) for item in source_components}
        if len(feature_ids) != 1:
            raise ValueError("champion revalidation lane mixes feature sets")
        feature_set = get_feature_set(next(iter(feature_ids)))
        label_binding = resolve_research_label_binding(
            {
                "horizon_profile": horizon_profile,
                "dataset": str(dataset["name"]),
                "dataset_identity_sha256": identity,
                "periods": dict(periods),
                "feature_set": feature_set,
                "research_window_contract": dict(research_window_contract),
                "research_window_contract_sha256": research_window_contract_sha256,
                "label_horizon_sessions": label_horizon_sessions,
            }
        )
        if label_binding is None:
            raise ValueError("champion revalidation has no active label binding")
        run_kind = (
            f"champion_current_identity_revalidation_{lane_token}_"
            f"{horizon_profile}"
        )
        with self.engine.connect() as connection:
            existing_run = connection.execute(
                select(research_runs).where(
                    research_runs.c.kind == run_kind,
                    research_runs.c.dataset == str(dataset["name"]),
                    research_runs.c.requested_by == requested_by,
                )
            ).first()
        config = {
            "contract_version": "champion-current-identity-revalidation-run-v2-horizon",
            "autopilot_cycle_id": cycle_id,
            "research_tournament_id": tournament_id,
            "dataset_identity_sha256": identity,
            "source_manifest_sha256": source_manifest_sha256,
            "source_model_candidate_ids": source_ids,
            "periods": dict(periods),
            "evaluation_profiles": profiles,
            "horizon_profile": horizon_profile,
            "label_horizon_sessions": label_horizon_sessions,
            "research_window_contract": dict(research_window_contract),
            "research_window_contract_sha256": research_window_contract_sha256,
            "research_label_binding": label_binding,
            "research_label_binding_sha256": label_binding["binding_sha256"],
            "primary_label_policy": policy,
            "primary_label_policy_sha256": policy["policy_sha256"],
            "fixed_recipe_only": True,
            "final_oos_opened": False,
            "research_screening_only": True,
            "not_capital_confirmation": True,
        }
        if existing_run is None:
            run = self.research.create_run(
                kind=run_kind,
                objective=(
                    "Revalidate the prior frozen prediction champion on one new "
                    "Qlib data identity without changing model recipe or features."
                ),
                dataset=str(dataset["name"]),
                requested_by=requested_by,
                budget={
                    "cpu_only": True,
                    "qlib_evaluation_concurrency_cap": 3,
                    "service_resource_reserve": 0.25,
                },
                config=config,
                artifact_path=self.settings.data_root / "artifacts" / "model-tournaments",
            )
        else:
            run = self.research.get_run(str(existing_run.id))
            if dict(existing_run.config_json or {}) != config:
                raise ValueError("existing champion revalidation run changed contract")
            if str(existing_run.status) in {"failed", "cancelled"}:
                raise ValueError("failed champion revalidation requires explicit retry")

        source_by_id = {
            str(item.get("source_model_candidate_id") or ""): dict(item)
            for item in source_components
        }
        trial_by_source = {
            str((item.get("spec") or {}).get("source_model_candidate_id") or ""): dict(item)
            for item in trials
        }
        if set(source_by_id) != set(trial_by_source) or "" in source_by_id:
            raise ValueError("champion revalidation trial bindings are incomplete")
        with self.engine.connect() as connection:
            existing_candidates = {
                str(item.name): item
                for item in connection.execute(
                    select(model_candidates).where(model_candidates.c.research_run_id == run["id"])
                )
            }
        candidate_payloads: list[dict[str, Any]] = []
        bindings: list[dict[str, str]] = []
        expected_names: set[str] = set()
        for index, source_id in enumerate(sorted(source_by_id), start=1):
            source = source_by_id[source_id]
            trial = trial_by_source[source_id]
            old = self.candidates.get_model_candidate(source_id, verify=True)
            old_manifest = dict(old.get("manifest_json") or {})
            feature = dict(old.get("base_features_manifest_json") or {})
            feature_set_id = str(feature.get("feature_set_id") or "")
            feature_set = get_feature_set(feature_set_id)
            if (
                str(old.get("status")) != "research_admitted"
                or str(old.get("manifest_sha256")) != str(source["source_model_manifest_sha256"])
                or str(old_manifest.get("recipe_sha256") or "")
                != str(source["source_recipe_sha256"])
                or str(old.get("code_sha256")) != str(source["source_code_sha256"])
                or str(feature_set.get("definition_sha256"))
                != str(source["feature_set_definition_sha256"])
                or feature_set_id != str(source["feature_set_id"])
            ):
                raise ValueError("frozen champion component changed or is unavailable")
            artifact_type = f"champion_revalidation_model_code_{index}"
            with self.engine.connect() as connection:
                artifact_row = connection.execute(
                    select(research_run_artifacts).where(
                        research_run_artifacts.c.research_run_id == run["id"],
                        research_run_artifacts.c.artifact_type == artifact_type,
                    )
                ).first()
            if artifact_row is None:
                source_artifact = self.candidates.get_run_artifact(
                    str(old["code_artifact_id"]), verify=True
                )
                artifact = self.candidates.register_run_artifact(
                    research_run_id=str(run["id"]),
                    artifact_type=artifact_type,
                    storage_path=str(source_artifact["storage_path"]),
                    producer="quantlab_champion_revalidation",
                    actor="autopilot",
                    contract_version="champion-revalidation-model-code-v1",
                    metadata={
                        "source_model_candidate_id": source_id,
                        "source_code_artifact_id": str(old["code_artifact_id"]),
                        "source_code_sha256": str(old["code_sha256"]),
                    },
                )
            else:
                artifact = self.candidates.get_run_artifact(str(artifact_row.id), verify=True)
            if str(artifact.get("content_sha256")) != str(old.get("code_sha256")):
                raise ValueError("champion revalidation code artifact changed")
            name = f"champion-revalidation-{index}-{_safe_scope_token(source_id)}"
            expected_names.add(name)
            row = existing_candidates.get(name)
            if row is None:
                candidate = self.candidates.create_model_candidate(
                    research_run_id=str(run["id"]),
                    name=name,
                    description=(
                        "Current-identity retraining of a frozen champion component; "
                        "recipe and feature set are copied byte-for-byte from its source."
                    ),
                    model_type=str(old["model_type"]),
                    code_artifact_id=str(artifact["id"]),
                    architecture=dict(old.get("architecture_json") or {}),
                    model_hyperparameters=dict(old.get("model_hyperparameters_json") or {}),
                    training_hyperparameters=dict(old.get("training_hyperparameters_json") or {}),
                    feature_set_id=feature_set_id,
                    dataset=str(dataset["name"]),
                    dataset_identity_sha256=identity,
                    dataset_lineage_id=lineage_id,
                    pre_final_end=date.fromisoformat(str(periods["valid_end"])),
                    final_oos_start=date.fromisoformat(str(periods["test_start"])),
                    final_oos_end=date.fromisoformat(str(periods["test_end"])),
                    research_label_binding=label_binding,
                    rdagent_decision=None,
                    rdagent_feedback="fixed champion current-identity revalidation",
                )
            else:
                candidate = self.candidates.get_model_candidate(str(row.id), verify=True)
                if (
                    str(candidate.get("dataset_identity_sha256")) != identity
                    or str(candidate.get("dataset_lineage_id") or "") != lineage_id
                    or str(candidate.get("feature_set_definition_sha256"))
                    != str(source["feature_set_definition_sha256"])
                ):
                    raise ValueError("existing champion revalidation candidate changed contract")
            new_manifest = dict(candidate.get("manifest_json") or {})
            if (
                str(new_manifest.get("recipe_sha256") or "")
                != str(source["source_recipe_sha256"])
                or str(candidate.get("code_sha256")) != str(source["source_code_sha256"])
            ):
                raise ValueError("champion revalidation recipe was not copied exactly")
            candidate_payloads.append(
                {
                    "id": str(candidate["id"]),
                    "code_path": str(artifact["storage_path"]),
                    "code_sha256": str(candidate["code_sha256"]),
                    "model_type": str(candidate["model_type"]),
                    "model_engine": str(
                        (candidate.get("model_hyperparameters_json") or {}).get("model_engine")
                        or "rdagent_pytorch"
                    ),
                    "training_hyperparameters": dict(
                        candidate.get("training_hyperparameters_json") or {}
                    ),
                }
            )
            bindings.append(
                {
                    "source_model_candidate_id": source_id,
                    "candidate_id": str(candidate["id"]),
                    "model_family": str(source["model_family"]),
                    "feature_set_id": feature_set_id,
                    "trial_id": str(trial["id"]),
                }
            )
        if set(existing_candidates) - expected_names:
            raise ValueError("champion revalidation run contains unregistered candidates")
        if run.get("job_id"):
            job = self.jobs.get(str(run["job_id"]))
        else:
            # The controller creates one lane per feature set for a mixed
            # feature ensemble.  A lane itself must stay single-feature: the
            # evaluator's DataHandler is intentionally not allowed to merge
            # feature definitions behind the frozen ensemble contract.
            job = self.jobs.create(
                "model_evaluate",
                {
                    "research_run_id": str(run["id"]),
                    "dataset": str(dataset["name"]),
                    "dataset_path": str(dataset["path"]),
                    "dataset_identity_sha256": identity,
                    "evaluation_stage": "model_full",
                    "evaluation_profiles": profiles,
                    "feature_set_id": feature_set["id"],
                    "feature_set_definition_sha256": feature_set["definition_sha256"],
                    "feature_set": feature_set,
                    "candidates": candidate_payloads,
                    "research_tournament_id": tournament_id,
                    "candidate_bindings": bindings,
                    "champion_revalidation": True,
                    "universe": "cn_all",
                    "benchmark": "SH000300",
                    "horizon_profile": horizon_profile,
                    "label_horizon_sessions": label_horizon_sessions,
                    "research_window_contract": dict(research_window_contract),
                    "research_window_contract_sha256": research_window_contract_sha256,
                    "research_label_binding": label_binding,
                    "research_label_binding_sha256": label_binding["binding_sha256"],
                    "primary_label_policy": policy,
                    "primary_label_policy_sha256": policy["policy_sha256"],
                },
                self.settings.data_root
                / "platform"
                / "logs"
                / f"champion-revalidation-{cycle_id}-{lane_token}.log",
                dedupe_active_kind=False,
                idempotency_key=(
                    f"champion-revalidation:{cycle_id}:{source_manifest_sha256}:{lane_token}"
                ),
            )
            self.research.attach_job(str(run["id"]), str(job["id"]))
        return {
            "run": self.research.get_run(str(run["id"])),
            "job": job,
            "bindings": bindings,
            "candidate_payloads": candidate_payloads,
        }
