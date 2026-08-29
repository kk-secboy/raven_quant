from __future__ import annotations

import argparse
import json
from pathlib import Path

from _project import PROJECT_ROOT

from quant_platform.backup_restore import (
    assess_control_plane_backup_readiness,
    compose_context,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check only the prerequisites for a QuantLab v2 control-plane backup"
    )
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--minimum-free-gb", type=float, default=10.0)
    parser.add_argument("--project-name", default="quantlab-platform")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / "deploy" / ".env")
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=PROJECT_ROOT / "deploy" / "compose.yaml",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        context = compose_context(
            args.project_name,
            args.env_file,
            args.compose_file,
        )
        result = assess_control_plane_backup_readiness(
            context,
            args.backup_root,
            minimum_free_gb=args.minimum_free_gb,
        )
    except Exception as exc:
        result = {
            "status": "blocked",
            "backup_format_version": 2,
            "business_readiness_consulted": False,
            "immutable_data_copied": False,
            "checks": [
                {
                    "id": "backup_preflight",
                    "status": "block",
                    "evidence": f"{type(exc).__name__}: {exc}",
                }
            ],
        }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        report = args.report.resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
