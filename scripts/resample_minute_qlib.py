#!/usr/bin/env python3
"""Resample native 1/5-minute staging with Qlib's A-share calendar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.qlib_minute_resample import resample_staging_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-frequency", required=True)
    parser.add_argument("--target-frequency", required=True)
    args = parser.parse_args()
    resample_staging_directory(
        Path(args.source),
        Path(args.output),
        source_frequency=args.source_frequency,
        target_frequency=args.target_frequency,
    )


if __name__ == "__main__":
    main()
