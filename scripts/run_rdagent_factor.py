#!/usr/bin/env python3
"""Run a bounded RD-Agent factor loop, then export a sanitized JSON result."""

from __future__ import annotations

import argparse
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--loop-n", required=True, type=int)
    parser.add_argument("--duration", required=True)
    args = parser.parse_args()
    env = os.environ.copy()
    env["LOG_TRACE_PATH"] = args.trace
    subprocess.run(
        [args.command, "fin_factor", "--loop-n", str(args.loop_n), "--all-duration", args.duration],
        env=env,
        check=True,
    )
    subprocess.run(
        [os.sys.executable, args.bridge, "export", "--trace", args.trace, "--output", args.result],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
