from __future__ import annotations

import argparse
import json
from pathlib import Path

from _project import PROJECT_ROOT

from quant_platform.backup_restore import compose_context
from quant_platform.release_preflight import assess_release


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed release preflight for a running QuantLab deployment"
    )
    parser.add_argument("--project-name", default="quantlab-platform")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / "deploy" / ".env")
    parser.add_argument(
        "--compose-file", type=Path, default=PROJECT_ROOT / "deploy" / "compose.yaml"
    )
    parser.add_argument("--minimum-free-gb", type=float, default=20.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    context = compose_context(args.project_name, args.env_file, args.compose_file)
    result = assess_release(
        context,
        PROJECT_ROOT,
        minimum_free_gb=max(1.0, args.minimum_free_gb),
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        report = args.report.resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(output + "\n", encoding="utf-8")
    print(output)
    if result["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
