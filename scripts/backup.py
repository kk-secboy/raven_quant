from __future__ import annotations

import argparse
from pathlib import Path

from _project import PROJECT_ROOT

from quant_platform.backup_restore import compose_context, create_backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Coordinated QuantLab PostgreSQL and /data backup")
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--retention-count", type=int, default=14)
    parser.add_argument("--project-name", default="quantlab-platform")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / "deploy" / ".env")
    parser.add_argument(
        "--compose-file", type=Path, default=PROJECT_ROOT / "deploy" / "compose.yaml"
    )
    args = parser.parse_args()
    context = compose_context(args.project_name, args.env_file, args.compose_file)
    print(create_backup(context, args.backup_root, retention_count=args.retention_count))


if __name__ == "__main__":
    main()
