from __future__ import annotations

import pytest

from quant_platform.terminal_state_reconciliation import (
    validate_complete_failed_progress,
)

pytestmark = pytest.mark.no_database


def test_complete_failed_progress_requires_every_preregistered_trial() -> None:
    progress = {
        "completed_count": 2,
        "trial_count": 2,
        "succeeded_count": 0,
        "failed_count": 2,
        "trials": [
            {
                "trial_index": 0,
                "status": "failed",
                "score": None,
                "warnings": [],
                "error": "first exact execution error",
            },
            {
                "trial_index": 1,
                "status": "failed",
                "score": None,
                "warnings": ["retained warning"],
                "error": "second exact execution error",
            },
        ],
    }

    terminals = validate_complete_failed_progress(progress, expected_indexes=[0, 1])

    assert [item["trial_index"] for item in terminals] == [0, 1]
    assert terminals[1]["warnings"] == ["retained warning"]
    assert terminals[1]["error"] == "second exact execution error"


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"completed_count": 1}, "counters"),
        ({"succeeded_count": 1, "failed_count": 1}, "counters"),
        (
            {
                "trials": [
                    {
                        "trial_index": 0,
                        "status": "failed",
                        "score": None,
                        "warnings": [],
                        "error": "first",
                    },
                    {
                        "trial_index": 0,
                        "status": "failed",
                        "score": None,
                        "warnings": [],
                        "error": "duplicate",
                    },
                ]
            },
            "exact failed terminal",
        ),
        (
            {
                "trials": [
                    {
                        "trial_index": 0,
                        "status": "failed",
                        "score": None,
                        "warnings": [],
                        "error": "first",
                    },
                    {
                        "trial_index": 1,
                        "status": "failed",
                        "score": None,
                        "warnings": [],
                        "error": "",
                    },
                ]
            },
            "exact failed terminal",
        ),
    ],
)
def test_complete_failed_progress_rejects_ambiguous_evidence(
    patch: dict, message: str
) -> None:
    progress = {
        "completed_count": 2,
        "trial_count": 2,
        "succeeded_count": 0,
        "failed_count": 2,
        "trials": [
            {
                "trial_index": 0,
                "status": "failed",
                "score": None,
                "warnings": [],
                "error": "first",
            },
            {
                "trial_index": 1,
                "status": "failed",
                "score": None,
                "warnings": [],
                "error": "second",
            },
        ],
    }
    progress.update(patch)

    with pytest.raises(ValueError, match=message):
        validate_complete_failed_progress(progress, expected_indexes=[0, 1])


def test_successful_progress_cannot_be_used_without_metrics() -> None:
    progress = {
        "completed_count": 1,
        "trial_count": 1,
        "succeeded_count": 1,
        "failed_count": 0,
        "trials": [
            {
                "trial_index": 0,
                "status": "succeeded",
                "score": 1.0,
                "warnings": [],
                "error": None,
            }
        ],
    }

    with pytest.raises(ValueError, match="counters"):
        validate_complete_failed_progress(progress, expected_indexes=[0])
