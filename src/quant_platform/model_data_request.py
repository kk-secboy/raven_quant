"""Content identity for model-independent, training-window-specific Qlib data."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

PREPARED_DATA_REQUEST_VERSION = "model-prepared-data-request-v1"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iso_date(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("prepared data dates must be real ISO dates")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("prepared data dates must be real ISO dates") from exc
    return value


def _validated_periods(
    manifest: Mapping[str, Any], provider: Path, label_contract: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    """Apply the runner's authorization and temporal guards before any preparation.

    Validation boundaries do not become cache identity fields: valid_start and
    prediction_start select model segments, but do not change prepared arrays.
    A pre-final provider deliberately omits future OOS dates; never synthesize
    that future calendar to decide its embargo here.
    """
    raw = manifest.get("periods")
    names = ("train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end")
    if not isinstance(raw, Mapping) or any(name not in raw for name in names):
        raise ValueError("prepared data periods are incomplete")
    periods = {name: _iso_date(raw[name]) for name in names}
    if any(periods[f"{part}_start"] > periods[f"{part}_end"]
           for part in ("train", "valid", "test")):
        raise ValueError("prepared data period boundaries are reversed")
    if periods["valid_end"] >= periods["test_start"]:
        raise ValueError("model validation reaches the sealed final OOS")
    segment = manifest.get("prediction_segment", "valid")
    if segment not in {"valid", "test"}:
        raise ValueError("prepared data prediction segment is invalid")
    inference = manifest.get("inference_only") is True
    retrain = manifest.get("live_retrain") is True
    if inference and retrain:
        raise ValueError("live inference and live retraining are mutually exclusive")
    if inference or retrain:
        if segment != "test" or manifest.get("final_oos_opened") is not False:
            raise ValueError("live model execution requires a non-OOS test segment")
        if periods["test_start"] != periods["test_end"]:
            raise ValueError("live model execution must produce exactly one signal date")
    elif segment == "test" and manifest.get("final_oos_opened") is not True:
        raise ValueError("formal model prediction requires an opened final OOS ledger")
    for name in ("purge_sessions", "embargo_sessions"):
        value = label_contract.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("prepared data purge and embargo must be nonnegative integers")
    purge, embargo = label_contract["purge_sessions"], label_contract["embargo_sessions"]
    calendar = [
        _iso_date(line.strip())
        for line in (provider / "calendars" / "day.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("prepared data trading calendar must be unique and ordered")
    positions = {day: index for index, day in enumerate(calendar)}
    try:
        train_end = positions[periods["train_end"]]
        valid_start = positions[periods["valid_start"]]
        valid_end = positions[periods["valid_end"]]
    except KeyError as exc:
        raise ValueError("model periods are outside the governed trading calendar") from exc
    if valid_start - train_end - 1 < purge:
        raise ValueError("model train/validation boundary does not purge the forward label horizon")
    if valid_end - purge < valid_start:
        raise ValueError("model validation window is shorter than its label purge")
    if periods["test_start"] in positions:
        if positions[periods["test_start"]] - valid_end - 1 < embargo:
            raise ValueError("model validation/final-OOS embargo is too short")
    return periods, segment


def prepared_data_request(
    manifest: Mapping[str, Any], *, provider: Path, label_contract: Mapping[str, Any],
    producer_identity: Mapping[str, str], additional_factors: Path | None = None,
) -> dict[str, Any]:
    """Exclude models/seeds; bind every input that can change prepared array values."""
    dataset = str(manifest.get("dataset_identity_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", dataset):
        raise ValueError("prepared data requires an immutable dataset identity")
    features = (manifest.get("feature_set") or {}).get("features")
    if (
        not isinstance(features, dict) or not 1 <= len(features) <= 512
        or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
               for k, v in features.items())
    ):
        raise ValueError("prepared data requires an ordered governed feature definition")
    periods, prediction_segment = _validated_periods(manifest, provider, label_contract)
    if not label_contract.get("label_expression"):
        raise ValueError("prepared data has no governed label expression")
    universe = manifest.get("universe", "cn_all")
    instrument_digest = None
    if isinstance(universe, str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", universe):
            raise ValueError("prepared data market identifier is invalid")
        instrument_file = provider / "instruments" / f"{universe.lower()}.txt"
        if not instrument_file.is_file():
            raise ValueError("prepared data market membership is unavailable")
        instrument_digest = file_digest(instrument_file)
    elif not isinstance(universe, (list, dict)):
        raise ValueError("prepared data universe is invalid")
    if not producer_identity or any(
        not isinstance(value, str) or not value for value in producer_identity.values()
    ):
        raise ValueError("prepared data producer identity is unavailable")
    return {
        "contract_version": PREPARED_DATA_REQUEST_VERSION,
        "dataset_identity_sha256": dataset,
        # The physical pre-final view differs from a full/formal provider even
        # when their originating dataset identity is the same. This calendar
        # pins availability of future labels, not just requested output dates.
        "provider_calendar_sha256": file_digest(provider / "calendars" / "day.txt"),
        "market_membership_sha256": instrument_digest,
        "features": [[name, expression] for name, expression in features.items()],
        "universe": universe,
        "label_contract": dict(label_contract),
        "additional_factors_sha256": (
            file_digest(additional_factors) if additional_factors is not None else None
        ),
        "train_start": periods["train_start"], "train_end": periods["train_end"],
        "load_end": periods[f"{prediction_segment}_end"],
        "access_scope": {
            "final_oos_opened": manifest.get("final_oos_opened") is True,
            "inference_only": manifest.get("inference_only") is True,
            "live_retrain": manifest.get("live_retrain") is True,
        },
        "processors": {
            "feature": ["RobustZScoreNorm(train-window,clip_outlier=True)", "Fillna"],
            "learn_label": ["DropnaLabel", "CSZScoreNorm(full-date-cross-section)"],
            "precision": "preserve-qlib-dtypes-and-common-feature-promotion",
        },
        "producer_identity": dict(producer_identity),
    }
