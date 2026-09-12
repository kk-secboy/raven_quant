"""Plan or execute one audited pre-experiment fin_quant recovery successor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_data.config import Settings
from quant_platform.autopilot import AutopilotController
from quant_platform.fin_quant_handoff_recovery import FinQuantHandoffRecovery, plan_file
from quant_platform.research_tournament import canonical_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cycle")
    parser.add_argument("--restart-cycle", help="Restart unexported research on a repaired release")
    parser.add_argument("--event-key")
    parser.add_argument("--actor", default="codex-operator")
    parser.add_argument("--reason")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256", help="SHA256 of the reviewed plan file bytes")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    service = FinQuantHandoffRecovery(AutopilotController(Settings.from_env()))
    if args.execute:
        if args.plan is None or not args.plan_sha256:
            parser.error("execution requires --plan and --plan-sha256")
        plan = plan_file(args.plan, args.plan_sha256)
        result = service.execute(plan, expected_sha256=canonical_sha256(plan))
        print(json.dumps({"cycle_id": result["id"], "status": result["status"],
                          "source_cycle_id": plan["source_cycle_id"]}, ensure_ascii=False))
    else:
        if (bool(args.source_cycle) == bool(args.restart_cycle)
                or not args.event_key or not args.reason):
            parser.error("planning requires one source/restart cycle, --event-key and --reason")
        planner = service.plan_runtime_restart if args.restart_cycle else service.plan
        print(json.dumps(planner(args.restart_cycle or args.source_cycle,
                                 args.event_key, args.actor, args.reason),
                         sort_keys=True, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
