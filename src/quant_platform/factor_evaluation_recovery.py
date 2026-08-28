from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

_JOB_ID = re.compile(r"[0-9a-f]{32}")
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PERIOD_KEYS = (
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "test_start",
    "test_end",
)


class RecoverySafetyError(ValueError):
    """The orphan result cannot be proven safe to import."""


class JobReader(Protocol):
    def get(self, job_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FactorEvaluationRecoveryInspection:
    job: dict[str, Any]
    result: dict[str, Any]
    result_path: Path
    result_sha256: str
    evaluation_count: int
    succeeded_count: int
    failed_count: int

    def public_report(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "job_id": str(self.job["id"]),
            "job_kind": str(self.job["kind"]),
            "job_status": str(self.job["status"]),
            "research_run_id": str(self.job["payload"]["research_run_id"]),
            "result_path": str(self.result_path),
            "result_sha256": self.result_sha256,
            "evaluation_count": self.evaluation_count,
            "succeeded_count": self.succeeded_count,
            "failed_count": self.failed_count,
            "would_finish_as": "failed" if self.failed_count else "succeeded",
            "apply_requires_result_sha256": self.result_sha256,
        }


def inspect_orphan_factor_evaluation(
    store: JobReader,
    *,
    data_root: Path,
    job_id: str,
    proc_root: Path = Path("/proc"),
) -> FactorEvaluationRecoveryInspection:
    """Read-only, fail-closed inspection for one orphan factor evaluation."""

    if not _JOB_ID.fullmatch(job_id):
        raise RecoverySafetyError("job id must be 32 lowercase hexadecimal characters")
    job = store.get(job_id)
    if str(job.get("id") or "") != job_id:
        raise RecoverySafetyError("job lookup returned a different identity")
    if job.get("kind") != "factor_evaluate":
        raise RecoverySafetyError("only factor_evaluate jobs can be recovered")
    if job.get("status") != "running":
        raise RecoverySafetyError("only a still-running orphan job can be recovered")
    payload = job.get("payload")
    if not isinstance(payload, dict):
        raise RecoverySafetyError("factor evaluation job payload is invalid")
    research_run_id = str(payload.get("research_run_id") or "")
    if not _safe_path_component(research_run_id):
        raise RecoverySafetyError("research run identity is unsafe for artifact resolution")

    result_path = _owned_result_path(data_root, research_run_id, job_id)
    active_pids = active_factor_evaluator_pids(result_path, proc_root=proc_root)
    if active_pids:
        raise RecoverySafetyError(
            "factor evaluator is still running: " + ", ".join(str(pid) for pid in active_pids)
        )

    raw, result_sha256 = _read_stable_result(result_path)
    result = _decode_complete_json(raw)
    succeeded_count, failed_count = validate_factor_evaluation_result_contract(job, result)
    return FactorEvaluationRecoveryInspection(
        job=job,
        result=result,
        result_path=result_path,
        result_sha256=result_sha256,
        evaluation_count=succeeded_count + failed_count,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
    )


def active_factor_evaluator_pids(
    result_path: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> tuple[int, ...]:
    """Return evaluator PIDs bound to this exact output, or fail if unprovable."""

    require_factor_worker_namespace(proc_root)
    expected = str(result_path.resolve(strict=True))
    matches: list[int] = []
    unreadable: list[int] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        raise RecoverySafetyError("/proc cannot be enumerated safely") from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if proc_root == Path("/proc") and pid == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        except OSError:
            unreadable.append(pid)
            continue
        arguments = [
            value.decode("utf-8", errors="replace")
            for value in raw.split(b"\0")
            if value
        ]
        if not any(Path(value).name == "evaluate_factor_batch.py" for value in arguments):
            continue
        for index, value in enumerate(arguments[:-1]):
            if value != "--output":
                continue
            try:
                candidate = str(Path(arguments[index + 1]).resolve(strict=False))
            except OSError:
                candidate = arguments[index + 1]
            if candidate == expected:
                matches.append(pid)
                break
    if unreadable:
        raise RecoverySafetyError(
            "/proc contains unreadable processes; subprocess exit cannot be proven"
        )
    return tuple(sorted(matches))


def require_factor_worker_namespace(proc_root: Path = Path("/proc")) -> None:
    """Prove inspection runs in the target Qlib worker PID namespace.

    A one-off sibling container has a different PID namespace and would report
    a false absence for a still-running evaluator. The recovery command must be
    copied into and executed with ``docker exec`` inside the existing worker
    container, whose PID 1 is ``quant-worker`` and whose immutable queue allow
    list includes ``factor_evaluate``.
    """

    if not proc_root.is_dir():
        raise RecoverySafetyError("/proc is unavailable; subprocess exit cannot be proven")
    try:
        command = [
            value.decode("utf-8", errors="replace")
            for value in (proc_root / "1" / "cmdline").read_bytes().split(b"\0")
            if value
        ]
        environment = {
            value.partition(b"=")[0].decode("utf-8", errors="replace"): value.partition(b"=")[
                2
            ].decode("utf-8", errors="replace")
            for value in (proc_root / "1" / "environ").read_bytes().split(b"\0")
            if b"=" in value
        }
    except OSError as exc:
        raise RecoverySafetyError("target worker PID namespace cannot be verified") from exc
    if not any(Path(value).name == "quant-worker" for value in command):
        raise RecoverySafetyError(
            "recovery must run inside the existing quant-worker PID namespace"
        )
    allowed_kinds = {
        value.strip()
        for value in environment.get("WORKER_JOB_KINDS", "").split(",")
        if value.strip()
    }
    if "factor_evaluate" not in allowed_kinds:
        raise RecoverySafetyError("target worker does not own the factor_evaluate queue")


def _owned_result_path(data_root: Path, research_run_id: str, job_id: str) -> Path:
    root = Path(data_root).absolute()
    artifact_root = root / "artifacts" / "factor-evaluations"
    expected = artifact_root / research_run_id / job_id / "result.json"
    for path in (artifact_root, expected.parent.parent, expected.parent, expected):
        if path.is_symlink():
            raise RecoverySafetyError("factor evaluation artifact path contains a symlink")
    try:
        artifact_root_resolved = artifact_root.resolve(strict=True)
        result_resolved = expected.resolve(strict=True)
        result_resolved.relative_to(artifact_root_resolved)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RecoverySafetyError(
            "factor evaluation result is outside its owned artifact root"
        ) from exc
    metadata = result_resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RecoverySafetyError("factor evaluation result is not a private regular file")
    return result_resolved


def _read_stable_result(path: Path) -> tuple[bytes, str]:
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        raise RecoverySafetyError("factor evaluation result cannot be read") from exc
    if not raw:
        raise RecoverySafetyError("factor evaluation result is empty")
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(raw) != after.st_size:
        raise RecoverySafetyError("factor evaluation result changed while being inspected")
    return raw, hashlib.sha256(raw).hexdigest()


def _decode_complete_json(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value, end = json.JSONDecoder().raw_decode(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoverySafetyError("factor evaluation result JSON is incomplete") from exc
    if text[end:].strip() or not isinstance(value, dict):
        raise RecoverySafetyError("factor evaluation result JSON is not one complete object")
    return value


def validate_factor_evaluation_result_contract(
    job: dict[str, Any], result: dict[str, Any]
) -> tuple[int, int]:
    """Require exactly one outcome for every frozen candidate/profile pair.

    Candidate counts alone are insufficient: a duplicated recent-window result
    must never stand in for a missing balanced or robust window.  This shared
    contract is used by the live worker, the ledger importer and guarded orphan
    recovery before any durable state is changed.
    """

    if not isinstance(result, dict):
        raise RecoverySafetyError("factor evaluation result is not an object")
    if result.get("status") != "ok" or not isinstance(result.get("qlib_workflow"), dict):
        raise RecoverySafetyError("factor evaluation result has no completed Qlib workflow")
    payload = job.get("payload")
    if not isinstance(payload, dict):
        raise RecoverySafetyError("factor evaluation job payload is invalid")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RecoverySafetyError("factor evaluation job has no frozen candidates")
    candidate_ids = [str(item.get("id") or "") for item in candidates if isinstance(item, dict)]
    if (
        len(candidate_ids) != len(candidates)
        or any(not candidate_id for candidate_id in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise RecoverySafetyError("factor evaluation job candidate identities are invalid")
    profiles = payload.get("evaluation_profiles")
    frozen_profiles: dict[str, dict[str, str]] = {}
    profile_by_periods: dict[tuple[str, ...], str] = {}
    if profiles:
        if not isinstance(profiles, list) or not 1 <= len(profiles) <= 3:
            raise RecoverySafetyError("factor evaluation profile count is invalid")
        for profile in profiles:
            if not isinstance(profile, dict):
                raise RecoverySafetyError("factor evaluation profile is invalid")
            profile_id = str(profile.get("id") or "")
            if not profile_id or profile_id in frozen_profiles:
                raise RecoverySafetyError("factor evaluation profile identities are invalid")
            normalized_periods = _normalize_periods(profile.get("periods"))
            period_key = tuple(normalized_periods[key] for key in _PERIOD_KEYS)
            if period_key in profile_by_periods:
                raise RecoverySafetyError(
                    "factor evaluation profiles do not have unique frozen periods"
                )
            frozen_profiles[profile_id] = normalized_periods
            profile_by_periods[period_key] = profile_id
    else:
        normalized_periods = _normalize_periods(payload.get("periods"))
        frozen_profiles["default"] = normalized_periods
        profile_by_periods[
            tuple(normalized_periods[key] for key in _PERIOD_KEYS)
        ] = "default"

    evaluations = result.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise RecoverySafetyError("factor evaluation result has no evaluations")
    expected_pairs = {
        (candidate_id, profile_id)
        for candidate_id in candidate_ids
        for profile_id in frozen_profiles
    }
    seen_pairs: set[tuple[str, str]] = set()
    succeeded = 0
    failed = 0
    for item in evaluations:
        if not isinstance(item, dict):
            raise RecoverySafetyError("factor evaluation result contains a non-object outcome")
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id not in candidate_ids:
            raise RecoverySafetyError("factor evaluation result contains an unknown candidate")
        status_value = item.get("status")
        if status_value not in {"ok", "failed"}:
            raise RecoverySafetyError("factor evaluation result contains an unsupported status")
        item_periods = _normalize_periods(item.get("periods") or payload.get("periods"))
        period_key = tuple(item_periods[key] for key in _PERIOD_KEYS)
        profile_id = profile_by_periods.get(period_key)
        if profile_id is None:
            raise RecoverySafetyError(
                "factor evaluation result contains an extra or altered frozen profile"
            )
        reported_profile_id = _reported_profile_id(item)
        if set(frozen_profiles) != {"default"}:
            if status_value == "ok" and reported_profile_id != profile_id:
                raise RecoverySafetyError(
                    "successful factor evaluation is not bound to its frozen profile"
                )
            if reported_profile_id and reported_profile_id != profile_id:
                raise RecoverySafetyError(
                    "factor evaluation reported a different frozen profile"
                )
        pair = (candidate_id, profile_id)
        if pair in seen_pairs:
            raise RecoverySafetyError(
                "factor evaluation result duplicates a candidate/profile outcome"
            )
        seen_pairs.add(pair)
        if status_value == "ok":
            if (
                not isinstance(item.get("metrics"), dict)
                or not isinstance(item.get("recomputed_values_path"), str)
                or not _SHA256.fullmatch(str(item.get("recomputed_values_sha256") or ""))
                or not isinstance(item.get("recompute_evidence"), dict)
            ):
                raise RecoverySafetyError("successful factor evaluation evidence is incomplete")
            succeeded += 1
        else:
            failed += 1
    if seen_pairs != expected_pairs:
        raise RecoverySafetyError(
            "factor evaluation result does not cover every frozen candidate/profile exactly once"
        )
    return succeeded, failed


def _normalize_periods(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise RecoverySafetyError("factor evaluation periods are missing")
    try:
        parsed = {key: date.fromisoformat(str(value[key])) for key in _PERIOD_KEYS}
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoverySafetyError("factor evaluation periods are invalid") from exc
    train_start, train_end, valid_start, valid_end, test_start, test_end = (
        parsed[key] for key in _PERIOD_KEYS
    )
    if not (
        train_start <= train_end < valid_start <= valid_end < test_start <= test_end
    ):
        raise RecoverySafetyError("factor evaluation periods are not strictly ordered")
    return {key: parsed[key].isoformat() for key in _PERIOD_KEYS}


def _reported_profile_id(item: dict[str, Any]) -> str:
    metrics = item.get("metrics")
    profile = metrics.get("research_profile") if isinstance(metrics, dict) else None
    if isinstance(profile, dict) and profile.get("id"):
        return str(profile["id"])
    return str(item.get("research_profile_id") or "")


def _safe_path_component(value: str) -> bool:
    return bool(_PATH_COMPONENT.fullmatch(value) and value not in {".", ".."})
