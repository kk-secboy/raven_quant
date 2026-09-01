"""Shared helpers for the job_commands domain builders.

These functions were lifted verbatim out of ``worker.py`` so the per-domain
command builders can use them without importing the worker module itself.
``worker`` re-exports the ones tests and other modules still import from it.
"""

from __future__ import annotations



