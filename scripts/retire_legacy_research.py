from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_data.config import Settings
from quant_platform.legacy_research_retirement import LegacyResearchRetirement


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retire old research program/campaign orchestration without touching "
            "Autopilot jobs"
        )
    )
    parser.add_argument("--apply", action="store_true", help="apply the fail-closed plan")
    args = parser.parse_args()
    settings = Settings.from_env(Path.cwd() / ".env")
    retirement = LegacyResearchRetirement(settings.database_url)
    result = retirement.apply() if args.apply else retirement.plan()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
