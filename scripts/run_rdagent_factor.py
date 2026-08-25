#!/usr/bin/env python3
"""Backward-compatible entry point for the governed fin_factor scenario."""

from run_rdagent_scenario import main

if __name__ == "__main__":
    main(default_scenario="fin_factor")
