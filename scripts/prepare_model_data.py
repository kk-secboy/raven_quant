"""Trusted isolated Qlib preparation; this process never imports candidate model code."""
from __future__ import annotations

import argparse
import atexit
import errno
import json
import os
import sys
import time
from contextlib import suppress
from pathlib import Path

sys.path.insert(0, "/work")

from quant_platform.model_data_handler import prepare_memory_bounded_model_data  # noqa: E402
from quant_platform.model_data_request import (  # noqa: E402
    PREPARED_DATA_REQUEST_VERSION,
    file_digest,
)
from quant_platform.model_prepared_data import (  # noqa: E402
    canonical_key,
    manifest_sha256,
    write_prepared_data,
)
from quant_platform.upstream_versions import (  # noqa: E402
    require_upstream_runtime_identity,
    upstream_runtime_identity,
)

SUMMARY_VERSION = "model-prepared-data-summary-v1"


def hold_producer_lease(path: Path) -> None:
    """Keep staging alive even if the trusted parent dies before Docker exits."""
    import fcntl

    stream = path.open("rb")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
    except BaseException:
        stream.close()
        raise
    atexit.register(stream.close)


def producer_memory_snapshot() -> dict:
    """Read this producer's Linux process and complete container high-water marks."""
    def counter(path: str) -> int | None:
        try:
            value = int(Path(path).read_text(encoding="ascii").strip())
            return value if value >= 0 else None
        except (OSError, ValueError):
            return None

    rss = peak = None
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            key, _, value = line.partition(":")
            if key == "VmRSS":
                rss = int(value.split()[0]) * 1024
            elif key == "VmHWM":
                peak = int(value.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    current = counter("/sys/fs/cgroup/memory.current")
    cgroup_peak = counter("/sys/fs/cgroup/memory.peak")
    limit = counter("/sys/fs/cgroup/memory.max")
    version = 2
    if all(value is None for value in (current, cgroup_peak, limit)):
        current = counter("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        cgroup_peak = counter("/sys/fs/cgroup/memory/memory.max_usage_in_bytes")
        limit = counter("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        version = 1 if any(value is not None for value in (current, cgroup_peak, limit)) else None
    peaks = [value for value in (peak, cgroup_peak) if value is not None]
    return {
        "process_rss_bytes": rss, "process_peak_rss_bytes": peak,
        "cgroup_version": version, "cgroup_current_bytes": current,
        "cgroup_peak_bytes": cgroup_peak, "cgroup_limit_bytes": limit,
        "observed_memory_peak_bytes": max(peaks) if peaks else None,
    }


def _run(args, started: float, stage) -> None:
    hold_producer_lease(args.lease)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if request.get("contract_version") != PREPARED_DATA_REQUEST_VERSION:
        raise ValueError("prepared data request contract is invalid")
    identity = request["producer_identity"]
    for name in (
        "model_data_handler.py", "model_data_request.py", "model_prepared_data.py",
        "upstream_versions.py", "prepare_model_data.py",
    ):
        source = (Path("/work") / name if name == "prepare_model_data.py"
                  else Path("/work/quant_platform") / name)
        if file_digest(source) != identity.get(name):
            raise ValueError("prepared data producer source changed")
    provider = Path("/qlib")
    if file_digest(provider / "calendars" / "day.txt") != request["provider_calendar_sha256"]:
        raise ValueError("prepared data physical calendar changed")
    universe = request["universe"]
    if isinstance(universe, str) and file_digest(
        provider / "instruments" / f"{universe.lower()}.txt"
    ) != request["market_membership_sha256"]:
        raise ValueError("prepared data market membership changed")
    additional = Path("/work/additional_factors.parquet")
    if request.get("additional_factors_sha256") is not None:
        if file_digest(additional) != request["additional_factors_sha256"]:
            raise ValueError("prepared data additional factors changed")
    else:
        additional = None
    require_upstream_runtime_identity("qlib", upstream_runtime_identity("qlib"))
    import qlib

    qlib.init(provider_uri=str(provider), region="cn", kernels=1,
              expression_cache=None, dataset_cache=None,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": "/tmp/qlib-tracking", "default_exp_name": "prepare"}})
    preparation_started = time.monotonic()
    stage("qlib_initialized")

    prepared_at = None
    seal = None
    status = "prepared"
    try:
        data = prepare_memory_bounded_model_data(
            features=dict(request["features"]),
            label_expression=request["label_contract"]["label_expression"],
            instruments=universe, start_time=request["train_start"],
            end_time=request["load_end"], fit_end_time=request["train_end"],
            additional_factors_path=additional, on_stage=stage,
        )
        prepared_at = time.monotonic()
        stage("prepared_data_writing")
        output = Path(args.output)
        write_prepared_data(output, contract=request, data=data)
        seal = manifest_sha256(output)
    except OSError as exc:
        if exc.errno not in {errno.ENOSPC, errno.EDQUOT}:
            raise
        status = "storage_capacity_unavailable"
        raise
    finally:
        # Only successful preparation and capacity fallback have complete summaries.
        # Other failures retain their traceback and cannot be mistaken for a result.
        if seal is not None or status == "storage_capacity_unavailable":
            memory = producer_memory_snapshot()
            finished = time.monotonic()
            print(json.dumps({
                "contract_version": SUMMARY_VERSION, "status": status,
                "request_sha256": canonical_key(request), "manifest_sha256": seal,
                "prepare_seconds": (
                    prepared_at if prepared_at is not None else finished
                ) - preparation_started,
                "write_seconds": finished - prepared_at if prepared_at is not None else 0.0,
                "elapsed_seconds": finished - started, "memory": memory,
            }, allow_nan=False), flush=True)


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default="/work/request.json")
    parser.add_argument("--output", default="/output/entry")
    parser.add_argument("--lease", type=Path, default=Path("/prepared-producer.lease"))
    parser.add_argument("--audit", type=Path, default=Path("/audit/progress.jsonl"))
    args = parser.parse_args()
    # A fresh attempt gets its own progress file, independently of the model log.
    with args.audit.open("x", encoding="utf-8"):
        pass

    def stage(value: str, *, status: str = "running", echo: bool = True) -> None:
        record = {
            "contract_version": "model-prepared-data-progress-v1", "stage": value,
            "status": status, "elapsed_seconds": time.monotonic() - started,
            "memory": producer_memory_snapshot(),
        }
        encoded = json.dumps(record, allow_nan=False)
        with args.audit.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if echo:
            print(encoded, flush=True)

    stage("producer_started")
    try:
        _run(args, started, stage)
    except BaseException as exc:
        capacity = isinstance(exc, OSError) and exc.errno in {errno.ENOSPC, errno.EDQUOT}
        status = "capacity" if capacity else "failed"
        # Keep the original failure when the output disk itself is unavailable.
        with suppress(OSError):
            stage(f"producer_{status}", status=status, echo=False)
        raise
    else:
        # stdout's final line remains the separately validated result summary.
        stage("producer_completed", status="completed", echo=False)


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        if exc.errno not in {errno.ENOSPC, errno.EDQUOT}:
            raise
        print(json.dumps({"status": "storage_capacity_unavailable", "errno": exc.errno}),
              file=sys.stderr, flush=True)
        raise SystemExit(75) from exc
