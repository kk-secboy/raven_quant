from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from quant_data.config import Settings
from quant_data.supplemental_data import SUPPORTED_BUNDLES
from quant_platform.job_store import JobStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", choices=sorted(SUPPORTED_BUNDLES))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise ValueError("end must not be before start")
    settings = Settings.from_env()
    job = JobStore(settings.database_url).create(
        f"supplemental_{args.bundle}",
        {"bundle": args.bundle, "start": start.isoformat(), "end": end.isoformat()},
        Path("/data/platform/logs")
        / f"supplemental-{args.bundle}-{start:%Y%m%d}-{end:%Y%m%d}.log",
        idempotency_key=f"supplemental:{args.bundle}:{start.isoformat()}:{end.isoformat()}",
    )
    print(job["id"])


if __name__ == "__main__":
    main()
