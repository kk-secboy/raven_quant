from __future__ import annotations

import argparse
import json
from pathlib import Path

from _project import PROJECT_ROOT

from quant_platform.backup_restore import compose_context
from quant_platform.canonical_baseline import converge_canonical_baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or explicitly converge a mixed QuantLab Compose project "
            "onto one image-pinned canonical baseline"
        )
    )
    parser.add_argument("--project-name", default="quantlab-platform")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / "deploy" / ".env")
    parser.add_argument(
        "--compose-file", type=Path, default=PROJECT_ROOT / "deploy" / "compose.yaml"
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=("gpu",),
        default=[],
    )
    parser.add_argument(
        "--receipt-root",
        type=Path,
        default=Path("/opt/quantlab-backups/canonical-baselines"),
    )
    parser.add_argument("--release-id")
    parser.add_argument("--wait-timeout", type=int, default=600)
    parser.add_argument(
        "--confirm-convergence",
        action="store_true",
        help="Permit container recreation and canonical env/receipt writes.",
    )
    args = parser.parse_args()
    context = compose_context(
        args.project_name,
        args.env_file,
        args.compose_file,
        profiles=args.profile,
    )
    result = converge_canonical_baseline(
        context,
        args.receipt_root,
        confirmed=args.confirm_convergence,
        release_id=args.release_id,
        wait_timeout=args.wait_timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] in {"blocked"}:
        raise SystemExit(2)
    if result["status"] in {"rolled_back", "rollback_failed"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
