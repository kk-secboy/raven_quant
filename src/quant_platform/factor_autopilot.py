from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import select, update

from quant_data.config import Settings
from quant_data.database import (
    factor_candidates,
    jobs,
    research_runs,
)

from .cost_model import CostModelConfig
from .factor_library_store import (
    FactorLibraryStore,
    validate_factor_definition_immutability,
    validate_incremental_evidence,
    validate_sota_roll_forward,
)
from .feature_set_registry import get_feature_set, register_feature_set
from .job_store import JobStore
from .research_automation import factor_spearman, resolve_research_window_contract
from .research_horizon import (
    SHORT_1_5D,
    primary_label_horizon_sessions,
    primary_label_policy_contract,
)
from .research_store import FactorGatePolicy, ResearchStore
from .services import list_qlib_datasets

FACTOR_SOTA_AUTOPILOT_CONTRACT_VERSION = "factor-sota-autopilot-v2-dual-admission"
FACTOR_SOTA_MODEL_CONTRACT_VERSION = "factor-sota-lightgbm-ablation-v1"
FACTOR_SOTA_MAX_CANDIDATES = 6
REQUIRED_PROFILES = ("recent_3y", "balanced_5y", "robust_10y")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def factor_sota_admission_path(
    candidate_status: str, layers: dict[str, dict[str, Any]]
) -> str | None:
    """Classify only candidates whose three governed hard gates are intact."""

    if set(layers) != set(REQUIRED_PROFILES) or any(
        item.get("hard_status") != "passed" for item in layers.values()
    ):
        return None
    if candidate_status == "profile_pending" and all(
        item.get("effect_status") == "passed" for item in layers.values()
    ):
        return "standalone"
    if (
        candidate_status == "incremental_pending"
        and layers["recent_3y"].get("effect_status") == "failed"
    ):
        return "incremental"
    return None


def resolve_sota_roll_forward(
    current_dataset: dict[str, Any],
    active_versions: list[dict[str, Any]],
    dataset_catalog: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve an exact or same-lineage monotonic SOTA predecessor.

    Missing provenance blocks resolution instead of silently replacing a
    previously governed fourth feature set with Alpha158.
    """

    provenance = dict(current_dataset.get("provenance") or {})
    identity = str(provenance.get("dataset_identity_sha256") or "")
    lineage_id = str(
        current_dataset.get("lineage_id") or provenance.get("dataset_lineage_id") or ""
    )
    end_text = str(current_dataset.get("end_date") or "")
    try:
        current_end = date.fromisoformat(end_text)
    except ValueError as exc:
        raise ValueError("current SOTA dataset has no valid end date") from exc
    if (
        len(identity) != 64
        or not lineage_id
        or current_dataset.get("lineage_verified") is not True
    ):
        raise ValueError("current SOTA dataset lineage is not verified")

    active = [item for item in active_versions if item.get("status") == "active"]
    exact = [
        item
        for item in active
        if str(item.get("dataset_identity_sha256") or "") == identity
    ]
    if len(exact) > 1:
        raise ValueError("multiple active SOTA versions claim the current dataset identity")
    if exact:
        return {
            "contract_version": "sota-roll-forward-v1",
            "mode": "exact",
            "predecessor_id": str(exact[0]["id"]),
            "source_dataset": str(exact[0]["dataset"]),
            "source_dataset_identity_sha256": identity,
            "source_end_date": end_text,
            "target_dataset": str(current_dataset["name"]),
            "target_dataset_identity_sha256": identity,
            "target_end_date": end_text,
            "dataset_lineage_id": lineage_id,
            "lineage_verified": True,
        }

    catalog_by_name = {
        str(item.get("name") or ""): item
        for item in dataset_catalog
        if str(item.get("name") or "")
    }
    candidates: list[tuple[date, dict[str, Any], str]] = []
    unresolved: list[str] = []
    for version in active:
        source_identity = str(version.get("dataset_identity_sha256") or "")
        source = catalog_by_name.get(str(version.get("dataset") or ""))
        source_lineage = ""
        source_end_text = ""
        source_verified = False
        if source is not None and str(
            (source.get("provenance") or {}).get("dataset_identity_sha256") or ""
        ) == source_identity:
            source_lineage = str(
                source.get("lineage_id")
                or (source.get("provenance") or {}).get("dataset_lineage_id")
                or ""
            )
            source_end_text = str(source.get("end_date") or "")
            source_verified = source.get("lineage_verified") is True
        else:
            evidence = dict(version.get("evidence") or {})
            if str(evidence.get("dataset_identity_sha256") or "") == source_identity:
                source_lineage = str(evidence.get("dataset_lineage_id") or "")
                source_end_text = str(evidence.get("dataset_end_date") or "")
                source_verified = evidence.get("dataset_lineage_verified") is True
        try:
            source_end = date.fromisoformat(source_end_text)
        except ValueError:
            unresolved.append(str(version.get("id") or ""))
            continue
        if not source_lineage or not source_verified:
            unresolved.append(str(version.get("id") or ""))
            continue
        if source_lineage != lineage_id:
            continue
        if source_end >= current_end:
            raise ValueError(
                "same-lineage SOTA dataset end date did not increase monotonically"
            )
        candidates.append((source_end, version, source_identity))
    if not candidates:
        if unresolved:
            raise ValueError(
                "active SOTA provenance is unavailable; refusing Alpha158 fallback"
            )
        return None
    latest_end = max(item[0] for item in candidates)
    latest = [item for item in candidates if item[0] == latest_end]
    if len(latest) != 1:
        raise ValueError("same-lineage SOTA predecessor is ambiguous")
    source_end, version, source_identity = latest[0]
    return {
        "contract_version": "sota-roll-forward-v1",
        "mode": "roll_forward",
        "predecessor_id": str(version["id"]),
        "source_dataset": str(version["dataset"]),
        "source_dataset_identity_sha256": source_identity,
        "source_end_date": source_end.isoformat(),
        "target_dataset": str(current_dataset["name"]),
        "target_dataset_identity_sha256": identity,
        "target_end_date": current_end.isoformat(),
        "dataset_lineage_id": lineage_id,
        "lineage_verified": True,
    }


def validate_factor_sota_result_contract(
    payload: dict[str, Any], result: dict[str, Any]
) -> None:
    """Validate the complete trial family, including negative outcomes."""

    if result.get("contract_version") != "factor-sota-paired-frozen-model-v2":
        raise ValueError("factor SOTA result contract version is invalid")
    if result.get("status") not in {"passed", "failed"}:
        raise ValueError("factor SOTA result status is invalid")
    if result.get("dataset_identity_sha256") != payload.get(
        "dataset_identity_sha256"
    ):
        raise ValueError("factor SOTA result uses another dataset identity")
    if result.get("predecessor_id") != payload.get("predecessor_id"):
        raise ValueError("factor SOTA result uses another frozen predecessor")
    if payload.get("final_oos_opened") is not False:
        raise ValueError("factor SOTA manifest must keep final OOS closed")
    validate_sota_roll_forward(
        payload.get("predecessor_roll_forward"),
        predecessor_id=(
            str(payload["predecessor_id"])
            if payload.get("predecessor_id") is not None
            else None
        ),
        target_dataset=str(payload.get("dataset") or ""),
        target_dataset_identity_sha256=str(
            payload.get("dataset_identity_sha256") or ""
        ),
        target_lineage_id=str(payload.get("dataset_lineage_id") or ""),
        target_end_date=str(payload.get("dataset_end_date") or ""),
    )
    expected = [
        str(item.get("factor_candidate_id") or "")
        for item in payload.get("candidates") or []
    ]
    if any(
        item.get("admission_path") not in {"standalone", "incremental"}
        for item in payload.get("candidates") or []
    ):
        raise ValueError("factor SOTA manifest has an invalid admission path")
    trials = result.get("trials")
    if (
        not expected
        or len(expected) != len(set(expected))
        or not isinstance(trials, list)
        or len(trials) != len(expected)
        or int(result.get("attempted_hypotheses") or -1) != len(expected)
    ):
        raise ValueError("factor SOTA result omitted preregistered trials")
    observed = [str(item.get("factor_candidate_id") or "") for item in trials]
    if sorted(observed) != sorted(expected) or len(observed) != len(set(observed)):
        raise ValueError("factor SOTA trial identities disagree with the manifest")
    allowed_statuses = {"accepted", "rejected", "failed"}
    if any(
        not isinstance(item, dict) or item.get("status") not in allowed_statuses
        for item in trials
    ):
        raise ValueError("factor SOTA result contains an invalid trial outcome")
    accepted = result.get("accepted")
    if not isinstance(accepted, list) or len(accepted) > 1:
        raise ValueError("factor SOTA may atomically accept at most one candidate")
    accepted_ids = [str(item.get("factor_candidate_id") or "") for item in accepted]
    trial_accepted = [
        str(item["factor_candidate_id"])
        for item in trials
        if item.get("status") == "accepted"
    ]
    if accepted_ids != trial_accepted:
        raise ValueError("factor SOTA accepted member disagrees with its trial ledger")
    if (result["status"] == "passed") is not bool(accepted):
        raise ValueError("factor SOTA terminal status disagrees with accepted members")
    frozen = result.get("frozen_model")
    if not isinstance(frozen, dict) or canonical_sha256(frozen) != payload.get(
        "frozen_model_sha256"
    ):
        raise ValueError("factor SOTA result changed the frozen model")
    if accepted:
        member = accepted[0]
        candidate_id = str(member.get("factor_candidate_id") or "")
        proposals = {
            str(item.get("factor_candidate_id") or ""): item
            for item in payload.get("candidates") or []
        }
        proposal = proposals.get(candidate_id)
        incremental = member.get("incremental_evidence")
        if proposal is None or not isinstance(incremental, dict):
            raise ValueError("factor SOTA accepted an unregistered candidate")
        validate_incremental_evidence(incremental)
        if (
            incremental.get("frozen_model") != frozen
            or incremental.get("factor_candidate_id") != candidate_id
            or incremental.get("candidate_code_sha256")
            != proposal.get("candidate_code_sha256")
            or incremental.get("candidate_values_sha256")
            != proposal.get("candidate_values_sha256")
            or incremental.get("evaluation_ids")
            != proposal.get("profile_evaluation_ids")
            or member.get("similarity_cluster_id")
            != proposal.get("similarity_cluster_id")
            or member.get("admission_path") != proposal.get("admission_path")
        ):
            raise ValueError("factor SOTA accepted evidence changed after preregistration")
        members = result.get("members")
        if not isinstance(members, list):
            raise ValueError("factor SOTA accepted result has no immutable membership")
        baseline = list(payload.get("baseline_members") or [])
        replaced_ids = {
            str(item.get("factor_candidate_id") or "")
            for item in baseline
            if str(item.get("similarity_cluster_id") or "")
            == str(proposal.get("similarity_cluster_id") or "")
        }
        expected_member_ids = {
            str(item.get("factor_candidate_id") or "") for item in baseline
        } - replaced_ids | {candidate_id}
        actual_member_ids = [
            str(item.get("factor_candidate_id") or "")
            for item in members
            if isinstance(item, dict)
        ]
        if (
            len(actual_member_ids) != len(members)
            or len(actual_member_ids) != len(set(actual_member_ids))
            or set(actual_member_ids) != expected_member_ids
            or member.get("action") != ("replaced" if replaced_ids else "added")
            or member.get("replaced_factor_candidate_id")
            != (next(iter(replaced_ids)) if replaced_ids else None)
        ):
            raise ValueError("factor SOTA result changed the frozen baseline membership")


def validate_promoted_factor_sota_admission(
    candidate: dict[str, Any],
    *,
    admission_path: str,
    paired_evidence: dict[str, Any],
) -> None:
    """Verify an idempotent SOTA retry against the persisted admission path.

    Standalone factors keep their three-profile consensus as their candidate
    admission record. The paired add/replace evidence remains attached to the
    immutable SOTA member instead of being rewritten as incremental admission.
    """

    if admission_path not in {"standalone", "incremental"}:
        raise ValueError("factor SOTA candidate has an invalid admission path")
    validate_incremental_evidence(paired_evidence)
    if (
        candidate.get("status") != "promoted"
        or candidate.get("admission_path") != admission_path
        or paired_evidence.get("factor_candidate_id") != candidate.get("id")
        or paired_evidence.get("candidate_code_sha256")
        != candidate.get("code_sha256")
        or paired_evidence.get("candidate_values_sha256")
        != candidate.get("values_sha256")
    ):
        raise ValueError("promoted factor does not match the SOTA admission path")
    if admission_path == "incremental":
        if (
            candidate.get("profile_consensus") is not None
            or candidate.get("incremental_evidence") != paired_evidence
            or candidate.get("incremental_evidence_sha256")
            != canonical_sha256(paired_evidence)
        ):
            raise ValueError("promoted factor does not match the SOTA increment")
        return

    consensus = candidate.get("profile_consensus")
    if (
        not isinstance(consensus, dict)
        or consensus.get("status") != "passed"
        or candidate.get("profile_consensus_sha256") != canonical_sha256(consensus)
        or consensus.get("candidate_id") != candidate.get("id")
        or consensus.get("candidate_code_sha256") != candidate.get("code_sha256")
        or consensus.get("candidate_values_sha256") != candidate.get("values_sha256")
        or consensus.get("dataset_identity_sha256")
        != paired_evidence.get("dataset_identity_sha256")
        or consensus.get("evaluation_ids") != paired_evidence.get("evaluation_ids")
        or consensus.get("profile_periods") != paired_evidence.get("profile_periods")
        or candidate.get("incremental_evidence") is not None
        or candidate.get("incremental_evidence_sha256") is not None
    ):
        raise ValueError("promoted standalone factor does not match its profile consensus")


class FactorAutopilotService:
    """Bridge independent factor gates into one recoverable Autopilot branch."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.research = ResearchStore(settings.database_url)
        self.engine = self.research.engine
        self.jobs = JobStore(settings.database_url)
        self.library = FactorLibraryStore(self.engine)
        self.project_root = Path(__file__).resolve().parents[2]

    def ensure_incremental_lane(
        self,
        *,
        cycle: dict[str, Any],
        dataset: dict[str, Any],
        horizon_profile: str = SHORT_1_5D,
    ) -> dict[str, Any] | None:
        label_horizon_sessions = primary_label_horizon_sessions(horizon_profile)
        identity = str((dataset.get("provenance") or {}).get("dataset_identity_sha256") or "")
        lineage_id = str(
            dataset.get("lineage_id")
            or (dataset.get("provenance") or {}).get("dataset_lineage_id")
            or ""
        )
        dataset_end_date = str(dataset.get("end_date") or "")
        if len(identity) != 64:
            raise ValueError("factor SOTA autopilot requires a sealed dataset identity")
        if (
            not lineage_id
            or dataset.get("lineage_verified") is not True
            or not dataset_end_date
        ):
            raise ValueError("factor SOTA autopilot requires verified dataset lineage")
        policy = primary_label_policy_contract()
        if (
            cycle.get("horizon_profile") != horizon_profile
            or cycle.get("primary_label_policy_sha256") != policy["policy_sha256"]
        ):
            raise ValueError("factor SOTA cycle horizon binding changed")
        active_run = self._active_incremental_run(horizon_profile)
        if active_run is not None:
            return self._recover_active_lane(
                active_run,
                cycle_id=str(cycle["id"]),
                horizon_profile=horizon_profile,
                label_horizon_sessions=label_horizon_sessions,
            )
        candidates = self._eligible_candidates(
            str(cycle["id"]),
            identity,
            horizon_profile=horizon_profile,
            label_horizon_sessions=label_horizon_sessions,
        )
        if not candidates:
            return None
        baseline = self._frozen_baseline(
            dataset,
            identity,
            horizon_profile=horizon_profile,
        )
        attempted, prior = self._multiplicity_reference(
            identity=identity,
            frozen_model_sha256=baseline["frozen_model_sha256"],
        )
        candidates = [item for item in candidates if str(item["id"]) not in attempted]
        if not candidates:
            return None
        candidates = candidates[:FACTOR_SOTA_MAX_CANDIDATES]
        self._assign_candidate_clusters(candidates, baseline, identity)
        proposals = [self._proposal(item, baseline) for item in candidates]
        hypothesis_count = int(prior["attempted_hypotheses"]) + len(proposals)
        family_id = (
            f"factor-sota:{horizon_profile}:{identity}:"
            f"{baseline['frozen_model_sha256']}:"
            f"{baseline.get('predecessor_id') or 'bootstrap'}"
        )
        calendar = (
            (Path(str(dataset["path"])) / "calendars" / "day.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        feature_set = dict(baseline["frozen_model"]["feature_set"])
        periods, resolution = resolve_research_window_contract(
            dataset,
            calendar,
            horizon_profile=horizon_profile,
            feature_set=feature_set,
        )
        candidate_set_sha256 = canonical_sha256(
            {
                "candidates": [
                    {
                        "factor_candidate_id": str(item["factor_candidate_id"]),
                        "admission_path": str(item["admission_path"]),
                    }
                    for item in proposals
                ],
                "horizon_profile": horizon_profile,
                "label_horizon_sessions": label_horizon_sessions,
                "frozen_model_sha256": baseline["frozen_model_sha256"],
                "predecessor_roll_forward": baseline["predecessor_roll_forward"],
                "prior_hypotheses": prior["attempted_hypotheses"],
            }
        )
        run = self.research.create_run(
            kind=f"factor_sota_increment:{horizon_profile}",
            objective=(
                "Test hard-gate-passing standalone and complementary factors as paired "
                "additions or replacements against one frozen LightGBM recipe without "
                "opening final OOS."
            ),
            dataset=str(dataset["name"]),
            requested_by="autopilot",
            budget={
                "candidate_limit": FACTOR_SOTA_MAX_CANDIDATES,
                "cpu_only": True,
                "bootstrap_samples": 2000,
            },
            config={
                "contract_version": FACTOR_SOTA_AUTOPILOT_CONTRACT_VERSION,
                "autopilot_cycle_id": str(cycle["id"]),
                "horizon_profile": horizon_profile,
                "label_horizon_sessions": label_horizon_sessions,
                "primary_label_policy": policy,
                "primary_label_policy_sha256": policy["policy_sha256"],
                "dataset_identity_sha256": identity,
                "dataset_lineage_id": lineage_id,
                "dataset_end_date": dataset_end_date,
                "candidate_set_sha256": candidate_set_sha256,
                "candidate_ids": [
                    str(item["factor_candidate_id"]) for item in proposals
                ],
                "candidate_admission_paths": {
                    str(item["factor_candidate_id"]): str(item["admission_path"])
                    for item in proposals
                },
                "frozen_model_sha256": baseline["frozen_model_sha256"],
                "predecessor_roll_forward": baseline["predecessor_roll_forward"],
                "evaluation_profiles": resolution["evaluation_profiles"],
                "periods": periods,
                "research_window_contract": resolution[
                    "research_window_contract"
                ],
                "research_window_contract_sha256": resolution[
                    "research_window_contract_sha256"
                ],
                "final_oos_opened": False,
            },
            artifact_path=self.settings.data_root / "artifacts" / "factor-sota-evaluations",
        )
        payload = {
            "research_run_id": str(run["id"]),
            "autopilot_cycle_id": str(cycle["id"]),
            "horizon_profile": horizon_profile,
            "label_horizon_sessions": label_horizon_sessions,
            "primary_label_policy": policy,
            "primary_label_policy_sha256": policy["policy_sha256"],
            "evaluation_scope_id": str(run["id"]),
            "dataset": str(dataset["name"]),
            "dataset_path": str(dataset["path"]),
            "dataset_identity_sha256": identity,
            "dataset_lineage_id": lineage_id,
            "dataset_lineage_verified": True,
            "dataset_end_date": dataset_end_date,
            "universe": "cn_all",
            "evaluation_profiles": resolution["evaluation_profiles"],
            "periods": periods,
            "feature_set": feature_set,
            "research_window_contract": resolution[
                "research_window_contract"
            ],
            "research_window_contract_sha256": resolution[
                "research_window_contract_sha256"
            ],
            "baseline_members": baseline["baseline_members"],
            "candidates": proposals,
            "predecessor_id": baseline.get("predecessor_id"),
            "predecessor_roll_forward": baseline["predecessor_roll_forward"],
            "frozen_model": baseline["frozen_model"],
            "frozen_model_sha256": baseline["frozen_model_sha256"],
            "generate_ablation_predictions": True,
            "experiment_family_id": family_id,
            "experiment_hypothesis_count": hypothesis_count,
            "multiplicity_reference": {
                "rank_ic_p_values": prior["rank_ic_p_values"],
                "cost_return_p_values": prior["cost_return_p_values"],
            },
            "bootstrap_samples": 2000,
            "cost_model": CostModelConfig().to_dict(),
            "cost_reference_order_value": 100_000.0,
            "min_daily_instruments": 50,
            "final_oos_opened": False,
            "research_screening_only": True,
            "not_capital_confirmation": True,
        }
        try:
            job = self.jobs.create(
                "factor_sota_evaluate",
                payload,
                self.settings.data_root
                / "platform"
                / "logs"
                / f"factor-sota-autopilot-{run['id']}.log",
                dedupe_active_kind=False,
                idempotency_key=(
                    f"autopilot-factor-sota:{cycle['id']}:{horizon_profile}:"
                    f"{candidate_set_sha256}:"
                    f"{FACTOR_SOTA_AUTOPILOT_CONTRACT_VERSION}"
                ),
            )
            self.research.attach_job(str(run["id"]), str(job["id"]))
        except Exception as exc:
            self.research.mark_run(
                str(run["id"]),
                "failed",
                actor="autopilot",
                error=f"factor SOTA enqueue failed: {exc}",
            )
            raise
        return {
            "run": self.research.get_run(str(run["id"])),
            "job": job,
            "scope_key": candidate_set_sha256,
            "candidate_ids": [str(item["factor_candidate_id"]) for item in proposals],
            "frozen_model_sha256": baseline["frozen_model_sha256"],
        }

    def _active_incremental_run(
        self, horizon_profile: str
    ) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(research_runs).where(
                    research_runs.c.kind
                    == f"factor_sota_increment:{horizon_profile}",
                    research_runs.c.status.in_(("queued", "running", "evaluating")),
                )
            ).first()
        return dict(row._mapping) if row is not None else None

    def _recover_active_lane(
        self,
        run: dict[str, Any],
        *,
        cycle_id: str,
        horizon_profile: str,
        label_horizon_sessions: int,
    ) -> dict[str, Any] | None:
        """Attach an orphaned active run to its Autopilot branch on retry."""

        config = dict(run.get("config_json") or {})
        policy = primary_label_policy_contract()
        if (
            str(config.get("autopilot_cycle_id") or "") != cycle_id
            or str(config.get("horizon_profile") or "") != horizon_profile
            or int(config.get("label_horizon_sessions") or 0)
            != label_horizon_sessions
            or config.get("primary_label_policy") != policy
            or config.get("primary_label_policy_sha256")
            != policy["policy_sha256"]
        ):
            # Factor ablations are intentionally globally serial.  Another
            # cycle's active frozen baseline must finish before this one starts.
            return None
        job_id = str(run.get("job_id") or "")
        if not job_id:
            raise ValueError("active factor SOTA run has no durable job binding")
        job = self.jobs.get(job_id)
        payload = dict(job.get("payload") or {})
        if (
            job.get("kind") != "factor_sota_evaluate"
            or str(payload.get("research_run_id") or "") != str(run["id"])
            or str(payload.get("autopilot_cycle_id") or "") != cycle_id
            or str(payload.get("horizon_profile") or "") != horizon_profile
            or int(payload.get("label_horizon_sessions") or 0)
            != label_horizon_sessions
            or payload.get("primary_label_policy") != policy
            or payload.get("primary_label_policy_sha256")
            != policy["policy_sha256"]
            or str(payload.get("dataset_identity_sha256") or "")
            != str(config.get("dataset_identity_sha256") or "")
            or str(payload.get("dataset_lineage_id") or "")
            != str(config.get("dataset_lineage_id") or "")
            or str(payload.get("dataset_end_date") or "")
            != str(config.get("dataset_end_date") or "")
            or payload.get("predecessor_roll_forward")
            != config.get("predecessor_roll_forward")
        ):
            raise ValueError("active factor SOTA run has an invalid job binding")
        candidate_ids = [
            str(item.get("factor_candidate_id") or "")
            for item in payload.get("candidates") or []
        ]
        if (
            not candidate_ids
            or "" in candidate_ids
            or len(candidate_ids) != len(set(candidate_ids))
            or config.get("candidate_ids", candidate_ids) != candidate_ids
        ):
            raise ValueError("active factor SOTA candidate set changed after enqueue")
        candidate_admission_paths = {
            str(item.get("factor_candidate_id") or ""): str(
                item.get("admission_path") or ""
            )
            for item in payload.get("candidates") or []
        }
        if (
            set(candidate_admission_paths) != set(candidate_ids)
            or set(candidate_admission_paths.values())
            - {"standalone", "incremental"}
            or config.get("candidate_admission_paths", candidate_admission_paths)
            != candidate_admission_paths
        ):
            raise ValueError("active factor SOTA admission paths changed after enqueue")
        scope_key = str(config.get("candidate_set_sha256") or "")
        if len(scope_key) != 64:
            raise ValueError("active factor SOTA scope is unsealed")
        return {
            "run": self.research.get_run(str(run["id"])),
            "job": job,
            "scope_key": scope_key,
            "candidate_ids": candidate_ids,
            "frozen_model_sha256": str(payload["frozen_model_sha256"]),
        }

    def _eligible_candidates(
        self,
        cycle_id: str,
        identity: str,
        *,
        horizon_profile: str,
        label_horizon_sessions: int,
    ) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(factor_candidates, research_runs.c.config_json)
                .join(
                    research_runs,
                    research_runs.c.id == factor_candidates.c.research_run_id,
                )
                .where(
                    research_runs.c.requested_by == "autopilot",
                    research_runs.c.dataset.is_not(None),
                    factor_candidates.c.status.in_(
                        ("profile_pending", "incremental_pending")
                    ),
                    factor_candidates.c.admission_path.is_(None),
                    factor_candidates.c.incremental_evidence_json.is_(None),
                )
                .order_by(factor_candidates.c.created_at, factor_candidates.c.id)
            ).all()
        eligible: list[dict[str, Any]] = []
        policy = primary_label_policy_contract()
        for row in rows:
            item = dict(row._mapping)
            config = dict(item.pop("config_json") or {})
            if (
                str(config.get("autopilot_cycle_id") or "") != cycle_id
                or config.get("scenario") not in {"fin_factor", "fin_factor_report"}
            ):
                continue
            candidate = self.research.get_candidate(str(item["id"]))
            variables = dict(candidate.get("variables") or {})
            if not (
                candidate.get("factor_definition_id")
                and candidate.get("economic_family")
                and candidate.get("code_sha256")
                and candidate.get("values_sha256")
                and int(candidate.get("label_horizon_days") or 0)
                == label_horizon_sessions
                and variables.get("horizon_profile") == horizon_profile
                and variables.get("primary_label_policy") == policy
                and variables.get("primary_label_policy_sha256")
                == policy["policy_sha256"]
            ):
                continue
            evaluations = self.research.list_evaluations(str(candidate["id"]))
            by_profile = {
                str((evaluation.get("metrics") or {}).get("research_profile", {}).get("id") or ""):
                evaluation
                for evaluation in evaluations
            }
            if set(by_profile) != set(REQUIRED_PROFILES):
                continue
            layers = {
                profile_id: FactorGatePolicy().evaluate_layers(
                    by_profile[profile_id]["metrics"]
                )
                for profile_id in REQUIRED_PROFILES
            }
            admission_path = factor_sota_admission_path(candidate["status"], layers)
            if admission_path is None:
                continue
            if any(
                str(value.get("dataset_identity_sha256") or "") != identity
                for value in by_profile.values()
            ):
                continue
            candidate["profile_evaluations"] = by_profile
            candidate["proposed_admission_path"] = admission_path
            if any(
                value.get("candidate_values_sha256")
                != candidate.get("values_sha256")
                for value in by_profile.values()
            ):
                continue
            eligible.append(candidate)
        return eligible

    def resolve_active_sota(
        self,
        dataset: dict[str, Any],
        *,
        horizon_profile: str = SHORT_1_5D,
    ) -> dict[str, Any] | None:
        label_horizon_sessions = primary_label_horizon_sessions(horizon_profile)
        versions = [
            item
            for item in self.library.list_sota(limit=1000)
            if item.get("status") == "active"
            and item.get("universe") == "cn_all"
            and int(item.get("label_horizon_days") or 0)
            == label_horizon_sessions
        ]
        if not versions:
            return None
        resolution = resolve_sota_roll_forward(
            dataset,
            versions,
            list_qlib_datasets(self.settings.data_root),
        )
        if resolution is None:
            return None
        version_id = str(resolution["predecessor_id"])
        detail = self.library.get_sota(version_id)
        if (
            int(detail.get("label_horizon_days") or 0)
            != label_horizon_sessions
            or detail.get("universe") != "cn_all"
            or detail.get("status") != "active"
            or canonical_sha256(detail["evidence"])
            != detail.get("evidence_sha256")
            or canonical_sha256(detail["policy"]) != detail.get("policy_sha256")
        ):
            raise ValueError("active SOTA version provenance changed in place")
        feature_set = self.library.horizon_champion_feature_set(
            version_id,
            horizon_profile=horizon_profile,
        )
        if canonical_sha256(
            {key: value for key, value in feature_set.items() if key != "definition_sha256"}
        ) != feature_set.get("definition_sha256"):
            raise ValueError("active SOTA feature-set definition changed in place")
        member_manifest: list[dict[str, Any]] = []
        for member in detail["members"]:
            candidate = self.research.get_candidate(str(member["factor_candidate_id"]))
            definition = self.library.get_definition(str(member["factor_definition_id"]))
            definition_sha256 = validate_factor_definition_immutability(definition)
            incremental = dict(member.get("incremental_evidence") or {})
            validate_incremental_evidence(incremental)
            if (
                candidate.get("factor_definition_id") != member["factor_definition_id"]
                or incremental.get("factor_candidate_id") != candidate.get("id")
                or incremental.get("candidate_code_sha256")
                != candidate.get("code_sha256")
                or incremental.get("candidate_values_sha256")
                != candidate.get("values_sha256")
                or canonical_sha256(incremental)
                != member.get("incremental_evidence_sha256")
            ):
                raise ValueError("active SOTA member definition or hash changed in place")
            member_manifest.append(
                {
                    "member_rank": int(member["member_rank"]),
                    "factor_candidate_id": str(member["factor_candidate_id"]),
                    "factor_definition_id": str(member["factor_definition_id"]),
                    "factor_definition_sha256": definition_sha256,
                    "candidate_code_sha256": str(candidate["code_sha256"]),
                    "candidate_values_sha256": str(candidate["values_sha256"]),
                    "incremental_evidence_sha256": str(
                        member["incremental_evidence_sha256"]
                    ),
                }
            )
        resolution = {
            **resolution,
            "horizon_profile": horizon_profile,
            "label_horizon_sessions": label_horizon_sessions,
            "source_sota_evidence_sha256": str(detail["evidence_sha256"]),
            "source_feature_set_definition_sha256": str(
                feature_set["sota_feature_set_sha256"]
            ),
            "source_horizon_champion_feature_set_definition_sha256": str(
                feature_set["definition_sha256"]
            ),
            "source_member_set_sha256": canonical_sha256(member_manifest),
        }
        return {
            "id": version_id,
            "horizon_profile": horizon_profile,
            "label_horizon_sessions": label_horizon_sessions,
            "detail": detail,
            "feature_set": register_feature_set(feature_set),
            "resolution": resolution,
        }

    def _frozen_baseline(
        self,
        dataset: dict[str, Any],
        identity: str,
        *,
        horizon_profile: str = SHORT_1_5D,
    ) -> dict[str, Any]:
        label_horizon_sessions = primary_label_horizon_sessions(horizon_profile)
        baseline_members: list[dict[str, Any]] = []
        predecessor_id: str | None = None
        feature_set = get_feature_set("qlib-alpha158")
        predecessor_roll_forward: dict[str, Any] | None = None
        resolved = self.resolve_active_sota(
            dataset,
            horizon_profile=horizon_profile,
        )
        if resolved is not None:
            predecessor_id = str(resolved["id"])
            predecessor_roll_forward = dict(resolved["resolution"])
            detail = dict(resolved["detail"])
            feature_set = dict(resolved["feature_set"])
            for member in detail["members"]:
                candidate = self.research.get_candidate(str(member["factor_candidate_id"]))
                if (
                    int(candidate.get("label_horizon_days") or 0)
                    != label_horizon_sessions
                ):
                    raise ValueError(
                        "active SOTA member label differs from its horizon champion"
                    )
                evaluation = self.research.get_evaluation(str(member["factor_evaluation_id"]))
                feature_name = (
                    f"SOTA_{int(member['member_rank']):03d}_"
                    f"{str(member['factor_definition_id'])[-8:]}"
                )
                baseline_members.append(
                    {
                        **member,
                        "feature_name": feature_name,
                        "values_path": candidate["values_path"],
                        "values_sha256": candidate["values_sha256"],
                        "direction": (
                            -1
                            if (evaluation.get("metrics") or {}).get("direction") == "inverted"
                            else 1
                        ),
                        "label_horizon_days": label_horizon_sessions,
                    }
                )
        stub = self.project_root / "scripts" / "baseline_model_stub.py"
        if not stub.is_file():
            raise ValueError("factor SOTA frozen LightGBM stub is unavailable")
        recipe = {
            "contract_version": FACTOR_SOTA_MODEL_CONTRACT_VERSION,
            "kind": "governed_lightgbm_factor_ablation",
            "model_type": "Tabular",
            "model_engine": "lightgbm_baseline",
            "training_hyperparameters": {},
            "seed": 11,
            "feature_set_definition_sha256": feature_set["definition_sha256"],
            "horizon_profile": horizon_profile,
            "label_horizon_sessions": label_horizon_sessions,
            "final_oos_opened": False,
        }
        artifact_identity = {
            "code_sha256": file_sha256(stub),
            "recipe": recipe,
            "predecessor_id": predecessor_id,
        }
        frozen_model = {
            "kind": recipe["kind"],
            "model_artifact_id": (
                f"platform-factor-ablation-{canonical_sha256(artifact_identity)[:24]}"
            ),
            "model_artifact_sha256": canonical_sha256(artifact_identity),
            "training_recipe_sha256": canonical_sha256(recipe),
            "feature_contract_sha256": feature_set["definition_sha256"],
            "code_path": str(stub.resolve()),
            "code_sha256": file_sha256(stub),
            "model_type": "Tabular",
            "model_engine": "lightgbm_baseline",
            "training_hyperparameters": {},
            "feature_set": feature_set,
            "seed": 11,
            "final_oos_opened": False,
        }
        return {
            "predecessor_id": predecessor_id,
            "horizon_profile": horizon_profile,
            "label_horizon_sessions": label_horizon_sessions,
            "predecessor_roll_forward": predecessor_roll_forward,
            "baseline_members": baseline_members,
            "frozen_model": frozen_model,
            "frozen_model_sha256": canonical_sha256(frozen_model),
        }

    def _assign_candidate_clusters(
        self,
        candidates: list[dict[str, Any]],
        baseline: dict[str, Any],
        identity: str,
    ) -> None:
        new_correlations: list[dict[str, Any]] = []
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                value = factor_spearman(left, right)
                new_correlations.append(
                    {
                        "left_candidate_id": str(left["id"]),
                        "right_candidate_id": str(right["id"]),
                        "mean_abs_spearman": value,
                        "evidence": {
                            "contract_version": "factor-daily-rank-correlation-v1",
                            "left_values_sha256": left["values_sha256"],
                            "right_values_sha256": right["values_sha256"],
                        },
                    }
                )
        assignments = self.library.assign_similarity_clusters(
            candidate_ids=[str(item["id"]) for item in candidates],
            correlations=new_correlations,
            dataset_identity_sha256=identity,
            profile_id="recent_3y",
        )
        component_assignments = dict(assignments)
        baseline_by_cluster = {
            str(item["similarity_cluster_id"]): item
            for item in baseline["baseline_members"]
        }
        for candidate in candidates:
            pinned: dict[str, dict[str, Any]] = {}
            max_correlation = 0.0
            for member in baseline["baseline_members"]:
                reference = self.research.get_candidate(str(member["factor_candidate_id"]))
                evaluation = self.research.get_evaluation(str(member["factor_evaluation_id"]))
                reference["values_path"] = evaluation["recomputed_values_path"]
                reference["values_sha256"] = evaluation["recomputed_values_sha256"]
                correlation = factor_spearman(candidate, reference)
                max_correlation = max(max_correlation, correlation)
                self.library.record_similarity(
                    left_candidate_id=str(candidate["id"]),
                    right_candidate_id=str(reference["id"]),
                    dataset_identity_sha256=identity,
                    profile_id="recent_3y",
                    mean_abs_spearman=correlation,
                    evidence={
                        "contract_version": "factor-daily-rank-correlation-v1",
                        "left_values_sha256": candidate["values_sha256"],
                        "right_values_sha256": reference["values_sha256"],
                    },
                )
                if correlation > self.library.policy.cluster_threshold:
                    pinned[str(member["similarity_cluster_id"])] = member
            if len(pinned) > 1:
                candidate["preflight_rejection_reason"] = (
                    "candidate bridges multiple frozen SOTA correlation clusters"
                )
            elif pinned:
                cluster_id, member = next(iter(pinned.items()))
                assignments[str(candidate["id"])] = cluster_id
                candidate["replaced_feature_name"] = member.get("feature_name")
            if max_correlation >= self.library.policy.near_duplicate_threshold:
                candidate["preflight_rejection_reason"] = (
                    "candidate is a near-duplicate of the frozen SOTA"
                )
        # New-new connected components must share a pinned cluster.  If one
        # member pins the component, copy that identity to its peers.
        component_pins: dict[str, set[str]] = {}
        for candidate in candidates:
            original = str(component_assignments[str(candidate["id"])])
            current = str(assignments[str(candidate["id"])])
            if current in baseline_by_cluster:
                component_pins.setdefault(original, set()).add(current)
        for candidate in candidates:
            component = str(component_assignments[str(candidate["id"])])
            pins = component_pins.get(component) or set()
            if len(pins) > 1:
                candidate["preflight_rejection_reason"] = (
                    "candidate component bridges multiple frozen SOTA clusters"
                )
            elif pins:
                assignments[str(candidate["id"])] = next(iter(pins))
        with self.engine.begin() as connection:
            for candidate in candidates:
                cluster_id = assignments[str(candidate["id"])]
                connection.execute(
                    update(factor_candidates)
                    .where(factor_candidates.c.id == str(candidate["id"]))
                    .values(similarity_cluster_id=cluster_id)
                )
                candidate["similarity_cluster_id"] = cluster_id

    @staticmethod
    def _proposal(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        evaluations = candidate["profile_evaluations"]
        return {
            "factor_candidate_id": str(candidate["id"]),
            "admission_path": str(candidate["proposed_admission_path"]),
            "factor_definition_id": str(candidate["factor_definition_id"]),
            "factor_evaluation_id": str(evaluations["recent_3y"]["id"]),
            "profile_evaluation_ids": {
                profile: str(evaluations[profile]["id"]) for profile in REQUIRED_PROFILES
            },
            "profile_evaluation_evidence_sha256": {
                profile: str(evaluations[profile]["evidence_sha256"])
                for profile in REQUIRED_PROFILES
            },
            "candidate_code_sha256": str(candidate["code_sha256"]),
            "candidate_values_sha256": str(candidate["values_sha256"]),
            "economic_family": str(candidate["economic_family"]),
            "family_tags": list(candidate.get("family_tags") or []),
            "similarity_cluster_id": str(candidate["similarity_cluster_id"]),
            "values_path": str(candidate["values_path"]),
            "direction": (
                -1
                if (evaluations["recent_3y"].get("metrics") or {}).get("direction")
                == "inverted"
                else 1
            ),
            "label_horizon_days": int(candidate.get("label_horizon_days") or 0),
            "replaced_feature_name": candidate.get("replaced_feature_name"),
            "preflight_rejection_reason": candidate.get("preflight_rejection_reason"),
            "frozen_model_sha256": baseline["frozen_model_sha256"],
        }

    def _multiplicity_reference(
        self, *, identity: str, frozen_model_sha256: str
    ) -> tuple[set[str], dict[str, Any]]:
        attempted: set[str] = set()
        rank_p: list[float] = []
        return_p: list[float] = []
        hypothesis_count = 0
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(jobs).where(jobs.c.kind == "factor_sota_evaluate")
            ).all()
        for row in rows:
            payload = dict(row.payload_json or {})
            if (
                payload.get("dataset_identity_sha256") != identity
                or payload.get("frozen_model_sha256") != frozen_model_sha256
            ):
                continue
            candidates = list(payload.get("candidates") or [])
            attempted.update(
                str(item.get("factor_candidate_id") or "") for item in candidates
            )
            progress = dict(row.progress_json or {})
            trials = list(progress.get("trials") or [])
            hypothesis_count += len(candidates)
            for trial in trials:
                value = trial.get("rank_ic_p_value")
                if isinstance(value, (int, float)):
                    rank_p.append(float(value))
                value = trial.get("cost_return_p_value")
                if isinstance(value, (int, float)):
                    return_p.append(float(value))
        attempted.discard("")
        return attempted, {
            "attempted_hypotheses": hypothesis_count,
            "rank_ic_p_values": rank_p,
            "cost_return_p_values": return_p,
        }
