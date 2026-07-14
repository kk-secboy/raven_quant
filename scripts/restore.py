from __future__ import annotations

import argparse
from pathlib import Path

from _project import PROJECT_ROOT

from quant_platform.backup_restore import compose_context, restore_backup


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore QuantLab PostgreSQL and /data")
    parser.add_argument("--backup-directory", type=Path, required=True)
    parser.add_argument("--confirm-restore", action="store_true")
    parser.add_argument("--project-name", default="quantlab-platform")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / "deploy" / ".env")
    parser.add_argument(
        "--compose-file", type=Path, default=PROJECT_ROOT / "deploy" / "compose.yaml"
    )
    args = parser.parse_args()
    context = compose_context(args.project_name, args.env_file, args.compose_file)
    revision = restore_backup(context, args.backup_directory, confirmed=args.confirm_restore)
    print(f"restored schema {revision} from {args.backup_directory.resolve()}")


if __name__ == "__main__":
    main()
