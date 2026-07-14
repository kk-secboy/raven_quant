from __future__ import annotations

import argparse
import json
from pathlib import Path

from _project import PROJECT_ROOT

from quant_platform.backup_restore import compose_context
from quant_platform.release_upgrade import run_release_upgrade


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, back up, upgrade, verify, and automatically roll back QuantLab"
    )
    parser.add_argument("--confirm-upgrade", action="store_true")
    parser.add_argument("--project-name", default="quantlab-platform")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / "deploy" / ".env")
    parser.add_argument(
        "--compose-file", type=Path, default=PROJECT_ROOT / "deploy" / "compose.yaml"
    )
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--retention-count", type=int, default=14)
    parser.add_argument("--rollback-image-retention", type=int, default=3)
    parser.add_argument("--minimum-free-gb", type=float, default=20.0)
    parser.add_argument("--wait-timeout", type=int, default=300)
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Refresh base images during the build; requires registry connectivity",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    context = compose_context(args.project_name, args.env_file, args.compose_file)
    result = run_release_upgrade(
        context,
        PROJECT_ROOT,
        args.backup_root,
        confirmed=args.confirm_upgrade,
        retention_count=args.retention_count,
        minimum_free_gb=max(1.0, args.minimum_free_gb),
        wait_timeout=args.wait_timeout,
        pull_images=args.pull,
        rollback_image_retention=args.rollback_image_retention,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    report = args.report
    if report is None:
        report = (
            PROJECT_ROOT
            / "artifacts"
            / "release-upgrades"
            / f"release-upgrade-{result['release_id']}.json"
        )
    report = report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(output + "\n", encoding="utf-8")
    print(output)
    if result["status"] != "succeeded":
        raise SystemExit(2 if result["status"] == "blocked" else 1)


if __name__ == "__main__":
    main()
