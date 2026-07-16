#!/usr/bin/env python3
"""Extract one immutable 5-minute execution day for transactional simulation booking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_data.execution_contract import require_minute_execution_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    provider = Path(args.provider_uri)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    provenance = json.loads(
        (provider / "metadata" / "provenance.json").read_text(encoding="utf-8")
    )
    require_minute_execution_contract(provenance, frequency="5min")
    instruments = [str(value) for value in manifest.get("instruments") or []]
    if not instruments:
        raise ValueError("simulation batch has no target or held instruments")
    trade_date = str(manifest["trade_date"])
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(provider), region=REG_CN)
    fields = ["$close", "$vwap", "$volume", "$paused", "$up_limit", "$down_limit"]
    values = D.features(
        instruments,
        fields,
        start_time=f"{trade_date} 00:00:00",
        end_time=f"{trade_date} 23:59:59",
        freq="5min",
    ).reset_index()
    values.rename(
        columns={
            "$close": "close",
            "$vwap": "vwap",
            "$volume": "volume",
            "$paused": "paused",
            "$up_limit": "up_limit",
            "$down_limit": "down_limit",
        },
        inplace=True,
    )
    if "datetime" not in values or "instrument" not in values:
        raise ValueError("Qlib minute result has no datetime/instrument index")
    values["datetime"] = pd.to_datetime(values["datetime"], errors="coerce")
    for field in ("close", "vwap", "volume", "paused", "up_limit", "down_limit"):
        values[field] = pd.to_numeric(values[field], errors="coerce")
    values = values.dropna(
        subset=["datetime", "instrument", "close", "vwap", "volume"]
    )
    values = values[values["datetime"].dt.date == pd.Timestamp(trade_date).date()]
    if values.empty:
        raise ValueError("simulation execution day has no 5-minute bars")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bars_path = output.parent / "minute_bars.parquet"
    values.to_parquet(bars_path, index=False, compression="zstd")
    closing_prices = {
        str(instrument): {
            "price": float(group.sort_values("datetime").iloc[-1]["close"]),
            "market_date": trade_date,
        }
        for instrument, group in values.groupby("instrument")
        if group["close"].notna().any()
    }
    result = {
        "status": "ok",
        "batch_id": manifest["batch_id"],
        "dataset_identity_sha256": provenance["dataset_identity_sha256"],
        "dataset_lineage_id": provenance["dataset_lineage_id"],
        "execution_contract_version": provenance["execution_contract_version"],
        "minute_bars_file": bars_path.name,
        "closing_prices": closing_prices,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
