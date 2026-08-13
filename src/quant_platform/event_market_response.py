"""Build post-event market-response labels without exposing them as features.

The announcement NLP pipeline emits point-in-time fields at ``available_at``.
This module joins those fields to later daily bars and produces supervised
learning labels such as benchmark-adjusted returns and direction agreement.
The output contract is intentionally separate from factor artifacts:

* manifests declare ``role=training_label_only``;
* labels are not registered in ``factor_candidates``;
* every horizon carries an ``outcome_end`` and ``label_available_at`` timestamp;
* incomplete horizons and suspended/missing price paths remain null rather than
  being backfilled or fabricated.

The implementation is pure pandas plus atomic parquet/JSON writes so it can be
unit-tested with synthetic calendars before production orchestration is wired.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .announcement_nlp import (
    ANNOUNCEMENTS_DIR,
    LOGIC_FACTOR_NAME,
    NLP_SUBDIR,
    PROMPT_VERSION,
    _sha256_file,
)

LABEL_SCHEMA_VERSION = "event-market-response.v1"
DEFAULT_HORIZONS = (1, 3, 5, 20)
DEFAULT_BENCHMARK = "000300.SH"
LABEL_ROLE = "training_label_only"

_DIRECTION_SIGN = {"positive": 1.0, "negative": -1.0}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} misses required columns: {missing}")


def _normalise_bars(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    _require_columns(frame, {"ts_code", "trade_date", "close"}, label)
    bars = frame.copy()
    bars["ts_code"] = bars["ts_code"].astype(str).str.upper()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="coerce").dt.normalize()
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    if "pre_close" in bars:
        bars["pre_close"] = pd.to_numeric(bars["pre_close"], errors="coerce")
    if "amount" in bars:
        bars["amount"] = pd.to_numeric(bars["amount"], errors="coerce")
    if bars[["ts_code", "trade_date"]].isna().any(axis=None):
        raise ValueError(f"{label} contains invalid ts_code/trade_date values")
    key = ["ts_code", "trade_date"]
    conflicting = bars.groupby(key, sort=False)["close"].nunique(dropna=False)
    if (conflicting > 1).any():
        raise ValueError(f"{label} contains conflicting duplicate instrument/date closes")
    return bars.drop_duplicates(key, keep="last").sort_values(key, kind="stable")


def _base_close(bars: pd.DataFrame, position: int) -> float | None:
    if "pre_close" in bars.columns:
        value = bars.iloc[position]["pre_close"]
        if pd.notna(value) and float(value) > 0:
            return float(value)
    if position > 0:
        value = bars.iloc[position - 1]["close"]
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return None


def _safe_return(end_close: Any, base_close: float | None) -> float | None:
    if base_close is None or pd.isna(end_close) or float(end_close) <= 0:
        return None
    return float(end_close) / base_close - 1.0


def _amount_surprise(
    bars: pd.DataFrame,
    *,
    start_position: int,
    end_date: pd.Timestamp,
    trailing_sessions: int,
) -> float | None:
    if "amount" not in bars.columns:
        return None
    history = bars.iloc[max(0, start_position - trailing_sessions) : start_position]["amount"]
    event = bars[
        (bars["trade_date"] >= bars.iloc[start_position]["trade_date"])
        & (bars["trade_date"] <= end_date)
    ]["amount"]
    history = history[(history > 0) & history.notna()]
    event = event[(event > 0) & event.notna()]
    if len(history) < min(5, trailing_sessions) or event.empty:
        return None
    baseline = float(history.median())
    return None if baseline <= 0 else float(event.mean()) / baseline - 1.0


def build_event_market_response_labels(
    fields: pd.DataFrame,
    daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    benchmark_code: str = DEFAULT_BENCHMARK,
    trailing_sessions: int = 20,
) -> pd.DataFrame:
    """Return wide post-event labels with explicit observation timestamps.

    Horizon ``1`` measures the announcement's first available session from
    ``pre_close`` to that session's close.  Longer horizons use the benchmark
    trading calendar, not the security's row count, so suspensions do not
    silently turn a 5-session label into a later-date return.
    """

    _require_columns(
        fields,
        {"process_key", "ts_code", "available_at", "impact_direction"},
        "announcement fields",
    )
    horizon_values = tuple(sorted({int(value) for value in horizons}))
    if not horizon_values or any(value <= 0 for value in horizon_values):
        raise ValueError("horizons must contain positive integers")
    if trailing_sessions <= 0:
        raise ValueError("trailing_sessions must be positive")

    stocks = _normalise_bars(daily, label="daily bars")
    benchmarks = _normalise_bars(benchmark_daily, label="benchmark bars")
    benchmark = benchmarks[benchmarks["ts_code"] == benchmark_code.upper()].reset_index(
        drop=True
    )
    if benchmark.empty:
        raise ValueError(f"benchmark {benchmark_code} is absent from benchmark bars")
    benchmark_by_date = benchmark.set_index("trade_date", drop=False)
    calendar = benchmark["trade_date"].tolist()
    calendar_position = {date: index for index, date in enumerate(calendar)}

    events = fields.copy()
    events["ts_code"] = events["ts_code"].astype(str).str.upper()
    events["available_at"] = pd.to_datetime(
        events["available_at"], errors="coerce"
    ).dt.normalize()
    if events["available_at"].isna().any():
        raise ValueError("announcement fields contain invalid available_at values")

    stock_groups = {
        code: group.reset_index(drop=True)
        for code, group in stocks.groupby("ts_code", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for event in events.sort_values(
        ["available_at", "ts_code", "process_key"], kind="stable"
    ).itertuples(index=False):
        available_at = pd.Timestamp(event.available_at)
        result: dict[str, Any] = {
            "process_key": str(event.process_key),
            "ts_code": str(event.ts_code),
            "available_at": available_at,
            "impact_direction": str(event.impact_direction),
            "benchmark_code": benchmark_code.upper(),
            "label_role": LABEL_ROLE,
        }
        stock = stock_groups.get(str(event.ts_code))
        start_calendar_position = calendar_position.get(available_at)
        start_stock_position: int | None = None
        if stock is not None:
            matches = stock.index[stock["trade_date"] == available_at].tolist()
            if matches:
                start_stock_position = int(matches[0])
        direction_sign = _DIRECTION_SIGN.get(str(event.impact_direction))

        for horizon in horizon_values:
            prefix = f"{horizon}d"
            values: dict[str, Any] = {
                f"outcome_end_{prefix}": pd.NaT,
                f"label_available_at_{prefix}": pd.NaT,
                f"stock_return_{prefix}": np.nan,
                f"benchmark_return_{prefix}": np.nan,
                f"abnormal_return_{prefix}": np.nan,
                f"market_recognition_{prefix}": np.nan,
                f"amount_surprise_{prefix}": np.nan,
                f"complete_{prefix}": False,
            }
            if start_calendar_position is None or start_stock_position is None:
                result.update(values)
                continue
            end_position = start_calendar_position + horizon - 1
            if end_position >= len(calendar):
                result.update(values)
                continue
            end_date = pd.Timestamp(calendar[end_position])
            next_date = (
                pd.Timestamp(calendar[end_position + 1])
                if end_position + 1 < len(calendar)
                else pd.NaT
            )
            # Even if the last close is present in an immutable snapshot, the
            # outcome cannot be exposed to a training row until the next
            # market session.  Keeping it incomplete prevents consumers from
            # treating a null availability timestamp as "available now".
            if pd.isna(next_date):
                result.update(values)
                continue
            stock_end = stock[stock["trade_date"] == end_date]
            if stock_end.empty or available_at not in benchmark_by_date.index:
                result.update(values)
                continue
            benchmark_end = (
                benchmark_by_date.loc[end_date]
                if end_date in benchmark_by_date.index
                else None
            )
            if benchmark_end is None:
                result.update(values)
                continue
            stock_return = _safe_return(
                stock_end.iloc[0]["close"], _base_close(stock, start_stock_position)
            )
            benchmark_return = _safe_return(
                benchmark_end["close"], _base_close(benchmark, start_calendar_position)
            )
            if stock_return is None or benchmark_return is None:
                result.update(values)
                continue
            abnormal = stock_return - benchmark_return
            values.update(
                {
                    f"outcome_end_{prefix}": end_date,
                    f"label_available_at_{prefix}": next_date,
                    f"stock_return_{prefix}": stock_return,
                    f"benchmark_return_{prefix}": benchmark_return,
                    f"abnormal_return_{prefix}": abnormal,
                    f"market_recognition_{prefix}": (
                        direction_sign * abnormal if direction_sign is not None else np.nan
                    ),
                    f"amount_surprise_{prefix}": _amount_surprise(
                        stock,
                        start_position=start_stock_position,
                        end_date=end_date,
                        trailing_sessions=trailing_sessions,
                    ),
                    f"complete_{prefix}": True,
                }
            )
            result.update(values)
        rows.append(result)
    return pd.DataFrame(rows)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd", engine="pyarrow")
    os.replace(temporary, path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class MarketResponseSummary:
    labels_path: Path
    manifest_path: Path
    rows: int
    sha256: str
    complete_by_horizon: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "succeeded",
            "labels_path": str(self.labels_path),
            "manifest_path": str(self.manifest_path),
            "rows": self.rows,
            "sha256": self.sha256,
            "complete_by_horizon": dict(self.complete_by_horizon),
            "role": LABEL_ROLE,
        }


def write_event_market_response_labels(
    labels: pd.DataFrame,
    output_dir: Path,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    benchmark_code: str = DEFAULT_BENCHMARK,
    source: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> MarketResponseSummary:
    """Write an auditable labels artifact and a fail-closed role manifest."""

    horizon_values = tuple(sorted({int(value) for value in horizons}))
    labels_path = output_dir / "event_market_response_labels.parquet"
    manifest_path = output_dir / "event_market_response_labels.json"
    _write_parquet_atomic(labels, labels_path)
    sha256 = _sha256_file(labels_path)
    complete = {
        f"{horizon}d": int(labels.get(f"complete_{horizon}d", pd.Series(dtype=bool)).sum())
        for horizon in horizon_values
    }
    manifest = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "role": LABEL_ROLE,
        "artifact": labels_path.name,
        "sha256": sha256,
        "rows": int(len(labels)),
        "horizons": list(horizon_values),
        "benchmark_code": benchmark_code.upper(),
        "complete_by_horizon": complete,
        "availability_contract": (
            "each horizon is usable only at label_available_at_<horizon>; null means the "
            "outcome is not yet observable or the price path is incomplete"
        ),
        "forbidden_consumers": [
            "factor_candidates",
            "qlib_inference_features",
            "live_signal_generation",
        ],
        "source": dict(source or {}),
        "generated_at": (now or datetime.now(UTC)).isoformat(),
    }
    _write_json_atomic(manifest, manifest_path)
    return MarketResponseSummary(
        labels_path=labels_path,
        manifest_path=manifest_path,
        rows=len(labels),
        sha256=sha256,
        complete_by_horizon=complete,
    )


def _read_partition_years(dataset_dir: Path, years: set[int]) -> pd.DataFrame:
    paths = [
        path
        for year in sorted(years)
        for path in sorted((dataset_dir / f"partition_year={year}").glob("**/*.parquet"))
    ]
    if not paths:
        raise RuntimeError(
            f"no parquet partitions found in {dataset_dir} for years {sorted(years)}"
        )
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _validated_logic_source(
    manifest_path: Path, *, prompt_version: str
) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    """Return a checksum-verified logic manifest and its bound model set.

    Checkpoint reuse can legitimately combine rows produced by multiple
    governed models.  The factor manifest binds those rows with both a
    canonical ``mixed[...]`` label and the exact list in ``scope.models``.
    Validate that binding and return the concrete models for field selection.
    """

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("governed logic factor manifest is unreadable") from exc
    source = manifest.get("source")
    artifact_name = manifest.get("artifact")
    expected_sha256 = manifest.get("sha256")
    model = str(source.get("model") or "").strip() if isinstance(source, dict) else ""
    scope = source.get("scope") if isinstance(source, dict) else None
    if model.startswith("mixed[") and model.endswith("]"):
        raw_models = scope.get("models") if isinstance(scope, dict) else None
        models = tuple(
            sorted(
                {
                    str(value).strip()
                    for value in raw_models or []
                    if str(value).strip()
                }
            )
        )
        model_binding_valid = len(models) > 1 and model == f"mixed[{','.join(models)}]"
    else:
        models = (model,) if model else ()
        model_binding_valid = bool(models)
    if (
        manifest.get("factor") != LOGIC_FACTOR_NAME
        or not isinstance(source, dict)
        or source.get("dataset") != "announcement_nlp_fields"
        or source.get("prompt_version") != prompt_version
        or not model_binding_valid
        or not isinstance(artifact_name, str)
        or Path(artifact_name).name != artifact_name
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256.lower())
    ):
        raise RuntimeError("governed logic factor manifest has an incompatible source")
    artifact_path = manifest_path.parent / artifact_name
    if not artifact_path.is_file() or _sha256_file(artifact_path) != expected_sha256:
        raise RuntimeError("governed logic factor artifact failed checksum verification")
    return manifest, model, models


def process_event_market_response(
    data_root: Path,
    *,
    snapshot_name: str,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    benchmark_code: str = DEFAULT_BENCHMARK,
    prompt_version: str = PROMPT_VERSION,
) -> MarketResponseSummary:
    """Build labels from a verified immutable snapshot and current NLP fields."""

    snapshot = data_root / "snapshots" / snapshot_name
    verification_path = snapshot / "verification.json"
    manifest_path = snapshot / "manifest.json"
    if not verification_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"snapshot {snapshot_name} is missing manifest/verification evidence")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("ok") is not True or verification.get("errors"):
        raise RuntimeError(f"snapshot {snapshot_name} did not pass the blocking quality gate")

    fields_path = data_root / ANNOUNCEMENTS_DIR / NLP_SUBDIR / "fields.parquet"
    if not fields_path.is_file():
        raise RuntimeError(f"announcement NLP fields are unavailable at {fields_path}")
    logic_manifest_path = (
        data_root
        / ANNOUNCEMENTS_DIR
        / NLP_SUBDIR
        / "factors"
        / f"{LOGIC_FACTOR_NAME}.json"
    )
    if not logic_manifest_path.is_file():
        raise RuntimeError(
            f"governed logic factor manifest is unavailable at {logic_manifest_path}"
        )
    _, model, models = _validated_logic_source(
        logic_manifest_path, prompt_version=prompt_version
    )
    fields = pd.read_parquet(fields_path)
    _require_columns(
        fields, {"prompt_version", "model", "available_at"}, "announcement fields"
    )
    fields = fields[
        (fields["prompt_version"].astype(str) == prompt_version)
        & (fields["model"].astype(str).isin(models))
    ].copy()
    if fields.empty:
        raise RuntimeError(
            "no announcement fields exist for "
            f"prompt_version={prompt_version}, model={model}"
        )
    available = pd.to_datetime(fields["available_at"], errors="coerce")
    if available.isna().any():
        raise RuntimeError("announcement fields contain invalid available_at values")
    years = set(range(int(available.dt.year.min()) - 1, int(available.dt.year.max()) + 2))
    daily = _read_partition_years(snapshot / "parquet" / "daily", years)
    benchmark = _read_partition_years(snapshot / "parquet" / "index_daily", years)
    horizon_values = tuple(sorted({int(value) for value in horizons}))
    labels = build_event_market_response_labels(
        fields,
        daily,
        benchmark,
        horizons=horizon_values,
        benchmark_code=benchmark_code,
    )
    output_dir = data_root / ANNOUNCEMENTS_DIR / NLP_SUBDIR / "labels" / snapshot_name
    return write_event_market_response_labels(
        labels,
        output_dir,
        horizons=horizon_values,
        benchmark_code=benchmark_code,
        source={
            "snapshot_name": snapshot_name,
            "snapshot_manifest_sha256": _sha256_file(manifest_path),
            "snapshot_verification_sha256": _sha256_file(verification_path),
            "fields_sha256": _sha256_file(fields_path),
            "logic_factor_manifest_sha256": _sha256_file(logic_manifest_path),
            "prompt_version": prompt_version,
            "model": model,
            "models": list(models),
        },
    )
