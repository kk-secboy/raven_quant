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
    parser.add_argument(
        "--profile",
        action="append",
        choices=("gpu",),
        default=[],
        help="Enable an optional production Compose profile (repeatable).",
    )
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument(
        "--reuse-backup",
        type=Path,
        help=(
            "Reuse only the verified backup that owns the current live rollback "
            "contract; arbitrary older backups are rejected"
        ),
    )
    parser.add_argument("--retention-count", type=int, default=14)
    parser.add_argument("--rollback-image-retention", type=int, default=3)
    parser.add_argument("--minimum-free-gb", type=float, default=20.0)
    parser.add_argument("--wait-timeout", type=int, default=300)
    parser.add_argument(
        "--drain-active-work",
        action="store_true",
        help=(
            "After full preflight and build, close admission and wait for accepted work "
            "to finish before a fresh backup; incompatible with --reuse-backup"
        ),
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Refresh base images during the build; requires registry connectivity",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--preserve-model-sandbox-image",
        help="Reuse an existing digest-pinned model image only after exact numeric-source checks",
    )
    parser.add_argument(
        "--preserve-model-sandbox-image-id",
        help="Expected immutable image ID already present in the target private Docker daemon",
    )
    parser.add_argument(
        "--stable-release-link",
        type=Path,
        default=Path("/opt/quantlab"),
        help="Atomically switch this operational symlink only after acceptance.",
    )
    parser.add_argument(
        "--skip-stable-link",
        action="store_true",
        help="Do not switch the operational symlink (intended only for drills).",
    )
    args = parser.parse_args()

    context = compose_context(
        args.project_name,
        args.env_file,
        args.compose_file,
        profiles=args.profile,
    )
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
        reuse_backup=args.reuse_backup,
        stable_release_link=(
            None if args.skip_stable_link else args.stable_release_link
        ),
        preserve_model_sandbox_image=args.preserve_model_sandbox_image,
        preserve_model_sandbox_image_id=args.preserve_model_sandbox_image_id,
        drain_active_work=args.drain_active_work,
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
