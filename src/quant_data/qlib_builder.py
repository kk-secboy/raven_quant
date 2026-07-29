from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Callable, Collection
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from quant_platform.eligibility import (
    ELIGIBILITY_CONTRACT_VERSION,
    EligibilityPolicy,
    build_point_in_time_eligibility,
)
from quant_platform.style_exposures import standardize_panel

from .availability import (
    AVAILABILITY_POLICY_VERSION,
    availability_contract_label,
    recoverability_level,
)
from .execution_contract import (
    DAILY_QLIB_FIELD_CONTRACT_VERSION,
    INDEX_VOLUME_POLICY,
    QLIB_DAILY_AMOUNT_UNIT,
    QLIB_DAILY_VOLUME_UNIT,
    TUSHARE_DAILY_AMOUNT_UNIT,
    TUSHARE_DAILY_VOLUME_UNIT,
    TUSHARE_HAND_SIZE,
)
from .path_utils import to_wsl_path as _to_wsl_path
from .regulatory_events import (
    REGULATORY_EVENTS_RULE_VERSION,
    derive_regulatory_events,
    open_days_from_trade_cal,
)
from .style_exposure_panel import build_adjusted_close, build_raw_style_panel

logger = logging.getLogger(__name__)

_BASE_QLIB_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "factor",
    "change",
    "amount",
    "paused",
    "up_limit",
    "down_limit",
)

_GOVERNED_BENCHMARK = "000300.SH"

_DAILY_RESEARCH_FIELDS = (
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "dv_ttm",
    "total_mv",
    "circ_mv",
)

# Fundamental research fields dumped into the Qlib binaries, grouped by the
# source statement table. Every table feeds the same point-in-time channel:
# an ASOF join keyed on the announcement date (trade_date > ann_date), never
# on the report period (end_date), so no field here can leak an unpublished
# report. Fundamentals are never price-normalized: absolute amounts stay in
# CNY yuan and per-share values in yuan per share.
#
# q_profit_yoy, inv_turn, ocf_to_or, ocf_to_profit and salescash_to_or are
# documented Tushare fina_indicator output columns (doc_id=79) but flagged
# non-default there, so the downloader (which requests no explicit field
# list) currently never receives them. They stay declared on purpose: the
# build-time coverage diagnostics in _research_feature_contract warn about
# the missing source columns instead of silently dropping the fields, and a
# future downloader change that requests them explicitly lights the fields
# up without another contract change.
_FUNDAMENTAL_RESEARCH_FIELDS = {
    "fina_indicator": {
        "roe": "fund_roe",
        "roa": "fund_roa",
        "grossprofit_margin": "fund_grossprofit_margin",
        "debt_to_assets": "fund_debt_to_assets",
        "current_ratio": "fund_current_ratio",
        "or_yoy": "fund_revenue_yoy",
        "netprofit_yoy": "fund_netprofit_yoy",
        "q_sales_yoy": "fund_quarter_revenue_yoy",
        "q_profit_yoy": "fund_quarter_profit_yoy",
        "eps": "fund_eps",
        "bps": "fund_bps",
        "ocfps": "fund_ocfps",
        "roe_waa": "fund_roe_weighted",
        "roe_dt": "fund_roe_diluted",
        "roic": "fund_roic",
        "netprofit_margin": "fund_netprofit_margin",
        "assets_turn": "fund_assets_turnover",
        "inv_turn": "fund_inventory_turnover",
        "ar_turn": "fund_receivables_turnover",
        "quick_ratio": "fund_quick_ratio",
        "debt_to_eqt": "fund_debt_to_equity",
        "saleexp_to_gr": "fund_sales_expense_ratio",
        "adminexp_of_gr": "fund_admin_expense_ratio",
        "finaexp_of_gr": "fund_finance_expense_ratio",
        "op_yoy": "fund_op_profit_yoy",
        "equity_yoy": "fund_equity_yoy",
        "ocf_to_or": "fund_ocf_to_revenue",
        "ocf_to_profit": "fund_ocf_to_profit",
        "salescash_to_or": "fund_sales_cash_to_revenue",
        "interestdebt": "fund_interest_debt",
    },
    "income": {
        "n_income_attr_p": "fund_net_profit",
        "rd_exp": "fund_rd_expense",
    },
    "balancesheet": {
        "total_assets": "fund_total_assets",
        "money_cap": "fund_money_cap",
        "goodwill": "fund_goodwill",
    },
    "cashflow": {
        "n_cashflow_act": "fund_ocf_net",
        "c_pay_acq_const_fiolta": "fund_capex",
    },
}

# Per-field unit declarations for the fundamental research fields, merged
# into the provenance field_units next to _DAILY_FIELD_UNITS. Ratios reported
# by Tushare in percent keep the percent scale (no 0-1 rescaling).
_FUNDAMENTAL_FIELD_UNITS = {
    "fund_roe": "percent",
    "fund_roa": "percent",
    "fund_grossprofit_margin": "percent",
    "fund_debt_to_assets": "percent",
    "fund_revenue_yoy": "percent",
    "fund_netprofit_yoy": "percent",
    "fund_quarter_revenue_yoy": "percent",
    "fund_quarter_profit_yoy": "percent",
    "fund_roe_weighted": "percent",
    "fund_roe_diluted": "percent",
    "fund_roic": "percent",
    "fund_netprofit_margin": "percent",
    "fund_sales_expense_ratio": "percent",
    "fund_admin_expense_ratio": "percent",
    "fund_finance_expense_ratio": "percent",
    "fund_op_profit_yoy": "percent",
    "fund_equity_yoy": "percent",
    "fund_debt_to_equity": "percent",
    "fund_current_ratio": "ratio_unitless",
    "fund_quick_ratio": "ratio_unitless",
    "fund_ocf_to_revenue": "ratio_unitless",
    "fund_ocf_to_profit": "ratio_unitless",
    "fund_sales_cash_to_revenue": "ratio_unitless",
    "fund_assets_turnover": "turnover_times",
    "fund_inventory_turnover": "turnover_times",
    "fund_receivables_turnover": "turnover_times",
    "fund_eps": "cny_yuan_per_share",
    "fund_bps": "cny_yuan_per_share",
    "fund_ocfps": "cny_yuan_per_share",
    "fund_interest_debt": "cny_yuan",
    "fund_net_profit": "cny_yuan",
    "fund_rd_expense": "cny_yuan",
    "fund_total_assets": "cny_yuan",
    "fund_money_cap": "cny_yuan",
    "fund_goodwill": "cny_yuan",
    "fund_ocf_net": "cny_yuan",
    "fund_capex": "cny_yuan",
}

# Explicit per-field unit declarations written into the dataset provenance.
# Prices are normalized to 1.0 at the snapshot anchor (first adjusted close),
# volume is value-consistent shares (price x volume = true CNY turnover), and
# amount is CNY yuan (converted from the Tushare thousand-CNY source unit).
_DAILY_FIELD_UNITS = {
    "open": "snapshot_anchor_normalized_price",
    "high": "snapshot_anchor_normalized_price",
    "low": "snapshot_anchor_normalized_price",
    "close": "snapshot_anchor_normalized_price",
    "vwap": "snapshot_anchor_normalized_price",
    "volume": "value_consistent_shares_price_times_volume_equals_cny_amount",
    "factor": "adj_factor_div_base_price",
    "change": "decimal_return",
    "amount": "cny_yuan",
    "paused": "flag_1_when_no_volume",
    "up_limit": "snapshot_anchor_normalized_price",
    "down_limit": "snapshot_anchor_normalized_price",
}


class QlibBuilder:
    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path.resolve()
        self.research_feature_contract = self._research_feature_contract()

    @property
    def qlib_fields(self) -> tuple[str, ...]:
        return (*_BASE_QLIB_FIELDS, *self.research_feature_contract["fields"])

    def build_staging(self, staging_path: Path) -> Path:
        daily_glob = self.snapshot_path / "parquet" / "daily" / "**" / "*.parquet"
        adj_glob = self.snapshot_path / "parquet" / "adj_factor" / "**" / "*.parquet"
        limit_glob = self.snapshot_path / "parquet" / "stk_limit" / "**" / "*.parquet"
        if not list((self.snapshot_path / "parquet" / "daily").rglob("*.parquet")):
            raise FileNotFoundError("snapshot does not contain daily Parquet data")
        if not list((self.snapshot_path / "parquet" / "adj_factor").rglob("*.parquet")):
            raise FileNotFoundError("snapshot does not contain adj_factor Parquet data")
        if not list((self.snapshot_path / "parquet" / "stk_limit").rglob("*.parquet")):
            raise FileNotFoundError("snapshot does not contain A-share price-limit Parquet data")
        self._validate_research_sources()

        staging_path = staging_path.resolve()
        temporary = staging_path.with_name(f".{staging_path.name}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        partitions = temporary / "partitions"
        by_symbol = temporary / "by_symbol"
        partitions.mkdir(parents=True, exist_ok=True)
        by_symbol.mkdir(parents=True, exist_ok=True)

        connection = duckdb.connect()
        try:
            query = self._normalized_query(daily_glob, adj_glob, limit_glob)
            missing = connection.execute(
                self._missing_market_controls_query(daily_glob, adj_glob, limit_glob)
            ).fetchone()[0]
            if missing:
                raise RuntimeError(
                    f"{missing} daily rows have no valid adjustment factor or price limits"
                )
            invalid_units = connection.execute(
                self._invalid_daily_units_query(daily_glob)
            ).fetchone()[0]
            if invalid_units:
                raise RuntimeError(
                    f"{invalid_units} daily rows violate the Tushare hand/amount price contract"
                )
            connection.execute(
                f"COPY ({query}) TO {_sql_string(str(partitions))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (symbol), "
                "ROW_GROUP_SIZE 100000)"
            )
        finally:
            connection.close()

        for partition in sorted(partitions.glob("symbol=*")):
            symbol = partition.name.split("=", 1)[1]
            files = sorted(partition.glob("*.parquet"))
            if not files:
                continue
            if len(files) == 1:
                frame = pd.read_parquet(files[0])
            else:
                frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
            frame.insert(1, "symbol", symbol)
            frame.sort_values("date", inplace=True)
            frame.to_parquet(
                by_symbol / f"{symbol}.parquet",
                index=False,
                compression="zstd",
            )
        self._write_index_staging(by_symbol)
        shutil.rmtree(partitions)
        if not any(by_symbol.glob("*.parquet")):
            raise RuntimeError("Qlib staging produced no per-symbol Parquet files")
        if staging_path.exists():
            shutil.rmtree(staging_path)
        os.replace(temporary, staging_path)
        return staging_path / "by_symbol"

    def dump_bin(
        self,
        *,
        staging_by_symbol: Path,
        qlib_dir: Path,
        qlib_repo: Path,
        qlib_python: str,
        wsl_distro: str,
        max_workers: int = 8,
    ) -> Path:
        script = qlib_repo.resolve() / "scripts" / "dump_bin.py"
        if not script.exists():
            raise FileNotFoundError(f"Qlib dump script not found: {script}")
        qlib_dir = qlib_dir.resolve()
        qlib_dir.parent.mkdir(parents=True, exist_ok=True)
        if qlib_dir.exists():
            raise FileExistsError(f"Qlib output already exists: {qlib_dir}")
        self._snapshot_manifest_digest()
        if os.name == "nt" and qlib_python.startswith("/"):
            command = [
                "wsl",
                "-d",
                wsl_distro,
                "--exec",
                qlib_python,
                _to_wsl_path(script),
                "dump_all",
                "--data_path",
                _to_wsl_path(staging_by_symbol),
                "--qlib_dir",
                _to_wsl_path(qlib_dir),
            ]
        else:
            command = [
                qlib_python,
                str(script),
                "dump_all",
                "--data_path",
                str(staging_by_symbol),
                "--qlib_dir",
                str(qlib_dir),
            ]
        command.extend(
            [
                "--freq",
                "day",
                "--file_suffix",
                ".parquet",
                "--date_field_name",
                "date",
                "--symbol_field_name",
                "symbol",
                "--include_fields",
                ",".join(self.qlib_fields),
                "--max_workers",
                str(max_workers),
            ]
        )
        try:
            subprocess.run(command, check=True)
            required = (
                qlib_dir / "calendars" / "day.txt",
                qlib_dir / "instruments" / "all.txt",
                qlib_dir / "features",
            )
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise RuntimeError(f"Qlib dump completed without required outputs: {missing}")
            if not any((qlib_dir / "features").rglob("*.day.bin")):
                raise RuntimeError("Qlib dump produced no daily feature binaries")
            self._write_stock_universe(qlib_dir)
            self._write_portfolio_metadata(qlib_dir)
            self._write_provenance(qlib_dir)
        except Exception:
            if qlib_dir.exists():
                shutil.rmtree(qlib_dir)
            raise
        return qlib_dir

    def _field_units(self) -> dict[str, str]:
        """Unit declarations for every dumped field, including fundamentals."""

        units = dict(_DAILY_FIELD_UNITS)
        for field in self.research_feature_contract["fields"]:
            unit = _FUNDAMENTAL_FIELD_UNITS.get(field)
            if unit is not None:
                units[field] = unit
        return units

    def _write_provenance(self, qlib_dir: Path) -> None:
        snapshot_digest = self._snapshot_manifest_digest()
        snapshot_manifest = json.loads(
            (self.snapshot_path / "manifest.json").read_text(encoding="utf-8")
        )
        builder_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        fields = list(self.qlib_fields)
        field_units = self._field_units()
        contract = {
            "version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
            "frequency": "day",
            "fields": fields,
            "source_volume_unit": TUSHARE_DAILY_VOLUME_UNIT,
            "qlib_volume_unit": QLIB_DAILY_VOLUME_UNIT,
            "source_amount_unit": TUSHARE_DAILY_AMOUNT_UNIT,
            "qlib_amount_unit": QLIB_DAILY_AMOUNT_UNIT,
            "source_hand_size": int(TUSHARE_HAND_SIZE),
            "index_volume_policy": INDEX_VOLUME_POLICY,
            "field_units": field_units,
            "research_features": self.research_feature_contract,
            "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
        }
        contract_sha256 = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        source_lineage_id = str(snapshot_manifest.get("lineage_id") or "")
        lineage_verified = len(source_lineage_id) == 64 and all(
            character in "0123456789abcdef" for character in source_lineage_id
        )
        dataset_lineage_id = (
            hashlib.sha256(
                json.dumps(
                    {
                        "source_lineage_id": source_lineage_id,
                        "dataset_contract_sha256": contract_sha256,
                        "qlib_builder_sha256": builder_digest,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if lineage_verified
            else None
        )
        identity = {
            "snapshot_name": self.snapshot_path.name,
            "snapshot_manifest_sha256": snapshot_digest,
            "qlib_builder_sha256": builder_digest,
            "field_contract_version": DAILY_QLIB_FIELD_CONTRACT_VERSION,
            "frequency": "day",
            "fields": fields,
            "source_volume_unit": TUSHARE_DAILY_VOLUME_UNIT,
            "qlib_volume_unit": QLIB_DAILY_VOLUME_UNIT,
            "source_amount_unit": TUSHARE_DAILY_AMOUNT_UNIT,
            "qlib_amount_unit": QLIB_DAILY_AMOUNT_UNIT,
            "source_hand_size": int(TUSHARE_HAND_SIZE),
            "index_volume_policy": INDEX_VOLUME_POLICY,
            "field_units": field_units,
            "research_features": self.research_feature_contract,
            "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        provenance = {
            **identity,
            "dataset_identity_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "dataset_contract_sha256": contract_sha256,
            "dataset_lineage_id": dataset_lineage_id,
            "source_lineage_id": source_lineage_id or None,
            "source_lineage_generation": snapshot_manifest.get("lineage_generation"),
            "source_parent_snapshot": snapshot_manifest.get("parent_snapshot"),
            "source_start_date": snapshot_manifest.get("start_date"),
            "source_end_date": snapshot_manifest.get("end_date"),
            "lineage_verified": lineage_verified,
            "created_at": datetime.now(UTC).isoformat(),
        }
        target = qlib_dir / "metadata" / "provenance.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
        (target.parent / "research_feature_contract.json").write_text(
            json.dumps(self.research_feature_contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _snapshot_manifest_digest(self) -> str:
        snapshot_manifest = self.snapshot_path / "manifest.json"
        if not snapshot_manifest.exists():
            raise FileNotFoundError("immutable source snapshot is missing manifest.json")
        manifest = json.loads(snapshot_manifest.read_text(encoding="utf-8"))
        datasets = manifest.get("datasets")
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError("snapshot manifest has no dataset content identities")
        for dataset, entry in datasets.items():
            if not isinstance(entry, dict) or not entry.get("source_sha256"):
                raise ValueError(f"snapshot dataset {dataset} has no source SHA-256")
            files = entry.get("files")
            if not isinstance(files, list):
                raise ValueError(f"snapshot dataset {dataset} has no file manifest")
            if int(entry.get("rows") or 0) > 0 and not files:
                raise ValueError(f"snapshot dataset {dataset} has rows but no content files")
            for item in files:
                relative = Path(str(item.get("path") or ""))
                target = (self.snapshot_path / relative).resolve()
                try:
                    target.relative_to(self.snapshot_path)
                except ValueError as exc:
                    raise ValueError("snapshot manifest contains an unsafe file path") from exc
                if not target.is_file() or target.stat().st_size != int(item.get("bytes") or -1):
                    raise ValueError(f"snapshot file size mismatch: {relative.as_posix()}")
                if _sha256_file(target) != item.get("sha256"):
                    raise ValueError(f"snapshot file digest mismatch: {relative.as_posix()}")
        return hashlib.sha256(snapshot_manifest.read_bytes()).hexdigest()

    def _write_index_staging(self, by_symbol: Path) -> None:
        index_root = self.snapshot_path / "parquet" / "index_daily"
        files = sorted(index_root.rglob("*.parquet")) if index_root.exists() else []
        if not files:
            return
        frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        required = {"ts_code", "trade_date", "open", "high", "low", "close"}
        if frame.empty or not required.issubset(frame.columns):
            return
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame.sort_values(["ts_code", "trade_date"], inplace=True)
        for ts_code, group in frame.groupby("ts_code", sort=True):
            group = group.copy()
            base_price = float(group["close"].dropna().iloc[0])
            if base_price <= 0:
                continue
            exchange, code = str(ts_code).split(".", 1)[1], str(ts_code).split(".", 1)[0]
            symbol = f"{exchange.upper()}{code}"
            amount = (
                pd.to_numeric(group.get("amount", 0.0), errors="coerce").fillna(0.0) * 1000.0
            )
            close = pd.to_numeric(group["close"], errors="coerce")
            normalized = pd.DataFrame(
                {
                    "date": group["trade_date"],
                    "symbol": symbol,
                    "open": pd.to_numeric(group["open"], errors="coerce") / base_price,
                    "high": pd.to_numeric(group["high"], errors="coerce") / base_price,
                    "low": pd.to_numeric(group["low"], errors="coerce") / base_price,
                    "close": close / base_price,
                    # The index amount/volume ratio is not an index-point VWAP and
                    # its volume is not executable stock capacity.
                    "vwap": close / base_price,
                    "volume": 0.0,
                    "factor": 1.0 / base_price,
                    "change": pd.to_numeric(group.get("pct_chg", 0.0), errors="coerce")
                    .fillna(0.0)
                    .div(100.0),
                    "amount": amount,
                    "paused": 0.0,
                }
            )
            normalized.to_parquet(by_symbol / f"{symbol}.parquet", index=False, compression="zstd")

    def _write_stock_universe(self, qlib_dir: Path) -> None:
        daily_glob = self.snapshot_path / "parquet" / "daily" / "**" / "*.parquet"
        connection = duckdb.connect()
        try:
            rows = connection.execute(
                f"""
                SELECT ts_code, min(trade_date), max(trade_date)
                FROM read_parquet({_sql_string(str(daily_glob.resolve()))}, hive_partitioning=true)
                WHERE ts_code IS NOT NULL AND trade_date IS NOT NULL
                GROUP BY ts_code ORDER BY ts_code
                """
            ).fetchall()
        finally:
            connection.close()
        lines = []
        for ts_code, start, end in rows:
            code, exchange = str(ts_code).split(".", 1)
            lines.append(f"{exchange.upper()}{code}\t{start}\t{end}")
        if not lines:
            raise RuntimeError("Qlib stock universe is empty")
        target = qlib_dir / "instruments" / "cn_all.txt"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_portfolio_metadata(self, qlib_dir: Path) -> None:
        target = qlib_dir / "metadata"
        wrote_metadata = False

        calendar_source = self.snapshot_path / "parquet" / "trade_cal"
        calendar_files = (
            sorted(calendar_source.rglob("*.parquet"))
            if calendar_source.exists()
            else []
        )
        if calendar_files:
            frame = pd.concat(
                [pd.read_parquet(path) for path in calendar_files],
                ignore_index=True,
            )
            if {"cal_date", "is_open"}.issubset(frame.columns):
                calendar = pd.DataFrame(
                    {
                        "date": pd.to_datetime(frame["cal_date"], errors="coerce"),
                        "is_open": pd.to_numeric(
                            frame["is_open"], errors="coerce"
                        ),
                    }
                )
                calendar = calendar[calendar["is_open"] == 1].dropna(
                    subset=["date"]
                )
                calendar = (
                    calendar[["date"]]
                    .drop_duplicates()
                    .sort_values("date")
                )
                if not calendar.empty:
                    target.mkdir(parents=True, exist_ok=True)
                    calendar.to_parquet(
                        target / "known_trading_calendar.parquet",
                        index=False,
                        compression="zstd",
                    )
                    wrote_metadata = True

        industry_source = self.snapshot_path / "parquet" / "index_member_all"
        industry_files = (
            sorted(industry_source.rglob("*.parquet")) if industry_source.exists() else []
        )
        if industry_files:
            frame = pd.concat(
                [pd.read_parquet(path) for path in industry_files], ignore_index=True
            )
            instrument_column = "ts_code" if "ts_code" in frame.columns else "con_code"
            industry_column = next(
                (
                    name
                    for name in ("l1_code", "index_code", "l2_code")
                    if name in frame.columns
                ),
                None,
            )
            if industry_column and instrument_column in frame.columns:
                metadata = pd.DataFrame(
                    {
                        "instrument": frame[instrument_column].map(_qlib_symbol),
                        "industry": frame[industry_column].astype("string"),
                        "in_date": pd.to_datetime(frame.get("in_date"), errors="coerce"),
                        "out_date": pd.to_datetime(frame.get("out_date"), errors="coerce"),
                    }
                ).dropna(subset=["instrument", "industry", "in_date"])
                metadata.drop_duplicates(
                    ["instrument", "industry", "in_date", "out_date"], inplace=True
                )
                metadata.sort_values(["instrument", "in_date", "industry"], inplace=True)
                if not metadata.empty:
                    target.mkdir(parents=True, exist_ok=True)
                    metadata.to_parquet(
                        target / "industry_memberships.parquet", index=False
                    )
                    wrote_metadata = True

        weight_source = self.snapshot_path / "parquet" / "index_weight"
        weight_files = sorted(weight_source.rglob("*.parquet")) if weight_source.exists() else []
        if weight_files:
            frame = pd.concat([pd.read_parquet(path) for path in weight_files], ignore_index=True)
            required = {"index_code", "con_code", "trade_date", "weight"}
            if required.issubset(frame.columns):
                weights = pd.DataFrame(
                    {
                        "benchmark": frame["index_code"].map(_qlib_symbol),
                        "instrument": frame["con_code"].map(_qlib_symbol),
                        "datetime": pd.to_datetime(frame["trade_date"], errors="coerce"),
                        "weight": pd.to_numeric(frame["weight"], errors="coerce"),
                    }
                ).dropna()
                if not weights.empty and float(weights["weight"].max()) > 1.0:
                    weights["weight"] = weights["weight"] / 100.0
                weights = weights[weights["weight"] > 0]
                weights.drop_duplicates(
                    ["benchmark", "instrument", "datetime"], keep="last", inplace=True
                )
                weights.sort_values(
                    ["benchmark", "datetime", "instrument"], inplace=True
                )
                if not weights.empty:
                    target.mkdir(parents=True, exist_ok=True)
                    weights.to_parquet(target / "benchmark_weights.parquet", index=False)
                    wrote_metadata = True

        style_source = self.snapshot_path / "parquet" / "daily_basic"
        style_files = sorted(style_source.rglob("*.parquet")) if style_source.exists() else []
        if style_files:
            frame = pd.concat([pd.read_parquet(path) for path in style_files], ignore_index=True)
            required = {"ts_code", "trade_date", "total_mv"}
            if required.issubset(frame.columns):
                styles = self._build_style_exposures(frame)
                if not styles.empty:
                    target.mkdir(parents=True, exist_ok=True)
                    float_cap_column = "circ_mv" if "circ_mv" in frame.columns else "total_mv"
                    float_cap = pd.DataFrame(
                        {
                            "instrument": frame["ts_code"].map(_qlib_symbol),
                            "datetime": pd.to_datetime(frame["trade_date"], errors="coerce"),
                            "float_market_cap": pd.to_numeric(
                                frame[float_cap_column], errors="coerce"
                            ),
                        }
                    ).dropna()
                    float_cap = float_cap[float_cap["float_market_cap"] > 0]
                    float_cap.drop_duplicates(
                        ["instrument", "datetime"], keep="last", inplace=True
                    )
                    totals = float_cap.groupby("datetime")["float_market_cap"].transform(
                        "sum"
                    )
                    float_cap["weight"] = float_cap["float_market_cap"] / totals
                    float_cap[["instrument", "datetime", "weight"]].sort_values(
                        ["datetime", "instrument"]
                    ).to_parquet(
                        target / "full_market_weights.parquet",
                        index=False,
                        compression="zstd",
                    )
                    styles.to_parquet(target / "style_exposures.parquet", index=False)
                    wrote_metadata = True

        if self._write_market_context_metadata(target):
            wrote_metadata = True

        if not self._write_eligibility_metadata(target):
            raise RuntimeError("Qlib point-in-time eligibility metadata is incomplete")
        wrote_metadata = True

        if not wrote_metadata and target.exists() and not any(target.iterdir()):
            target.rmdir()

    def _build_style_exposures(self, daily_basic: pd.DataFrame) -> pd.DataFrame:
        """Extended Barra-style exposure panel with a backward-compatible schema.

        The historical raw ``log_market_cap`` column is preserved; the
        standardized style columns of quant_platform.style_exposures are added
        alongside it. Rows keep daily_basic's same-trade-date-after-close
        semantics (an exposure dated ``t`` supports decisions after the close
        of ``t``); fundamental descriptors arrive through the
        announcement-date ASOF channel inside style_exposure_panel.
        """

        panel = build_raw_style_panel(
            daily_basic,
            adjusted_close=self._load_adjusted_close(),
            fina_indicator=self._load_fina_indicator(),
        )
        panel = panel.rename(columns={"ts_code": "instrument", "trade_date": "datetime"})
        panel["instrument"] = panel["instrument"].map(_qlib_symbol)
        return standardize_panel(panel)

    def _load_adjusted_close(self) -> pd.DataFrame | None:
        daily_root = self.snapshot_path / "parquet" / "daily"
        daily_files = sorted(daily_root.rglob("*.parquet")) if daily_root.exists() else []
        daily = _read_parquet_columns(daily_files, {"ts_code", "trade_date", "close"})
        if daily is None:
            return None
        adj_root = self.snapshot_path / "parquet" / "adj_factor"
        adj_files = sorted(adj_root.rglob("*.parquet")) if adj_root.exists() else []
        factors = _read_parquet_columns(adj_files, {"ts_code", "trade_date", "adj_factor"})
        return build_adjusted_close(daily, factors)

    def _load_fina_indicator(self) -> pd.DataFrame | None:
        root = self.snapshot_path / "parquet" / "fina_indicator"
        files = sorted(root.rglob("*.parquet")) if root.exists() else []
        return _read_parquet_columns(
            files,
            {"ts_code", "ann_date", "roe", "or_yoy", "netprofit_yoy", "debt_to_assets"},
            required={"ts_code", "ann_date"},
        )

    def _write_market_context_metadata(self, target: Path) -> bool:
        """Store non-equity daily context once, without copying it into every stock."""

        definitions = {
            "index_global": (
                ("trade_date", "date"),
                ("close", "pct_chg", "vol", "amount"),
            ),
            "fx_daily": (
                ("trade_date", "date"),
                ("bid_open", "bid_close", "ask_open", "ask_close", "tick_qty"),
            ),
            "fut_daily": (
                ("trade_date", "date"),
                ("close", "settle", "vol", "amount", "oi"),
            ),
            "shibor": (
                ("date",),
                ("on", "1w", "2w", "1m", "3m", "6m", "9m", "1y"),
            ),
            "shibor_lpr": (("date",), ("1y", "5y")),
            "us_tycr": (
                ("date",),
                ("m1", "m2", "m3", "m6", "y1", "y2", "y3", "y5", "y7", "y10", "y20", "y30"),
            ),
        }
        chunks: list[pd.DataFrame] = []
        sources: dict[str, dict[str, object]] = {}
        for dataset, (date_candidates, value_candidates) in definitions.items():
            root = self.snapshot_path / "parquet" / dataset
            files = sorted(root.rglob("*.parquet")) if root.exists() else []
            if not files:
                continue
            frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
            date_field = next((name for name in date_candidates if name in frame.columns), None)
            value_fields = [name for name in value_candidates if name in frame.columns]
            if date_field is None or not value_fields:
                continue
            instrument = (
                frame["ts_code"].astype("string")
                if "ts_code" in frame.columns
                else pd.Series(dataset, index=frame.index, dtype="string")
            )
            normalized = frame[value_fields].apply(pd.to_numeric, errors="coerce")
            normalized.insert(0, "instrument", instrument)
            normalized.insert(0, "datetime", pd.to_datetime(frame[date_field], errors="coerce"))
            long = normalized.melt(
                id_vars=["datetime", "instrument"],
                var_name="feature",
                value_name="value",
            ).dropna(subset=["datetime", "instrument", "value"])
            if long.empty:
                continue
            long.insert(1, "source", dataset)
            chunks.append(long)
            sources[dataset] = {
                "date_field": date_field,
                "features": value_fields,
                "availability": "same_timestamp_after_close",
            }
        if not chunks:
            return False
        context = pd.concat(chunks, ignore_index=True)
        context.drop_duplicates(
            ["datetime", "source", "instrument", "feature"], keep="last", inplace=True
        )
        context.sort_values(["datetime", "source", "instrument", "feature"], inplace=True)
        target.mkdir(parents=True, exist_ok=True)
        context.to_parquet(target / "market_context.parquet", index=False, compression="zstd")
        (target / "market_context_contract.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "join_policy": "asof_backward_or_exact_for_next_period_signals",
                    "sources": sources,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return True

    def _write_eligibility_metadata(self, target: Path) -> bool:
        def read(dataset: str) -> pd.DataFrame:
            root = self.snapshot_path / "parquet" / dataset
            files = sorted(root.rglob("*.parquet")) if root.exists() else []
            return (
                pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
                if files
                else pd.DataFrame()
            )

        daily = read("daily")
        stock_basic = read("stock_basic")
        balancesheet = read("balancesheet")
        audit = read("fina_audit")
        if any(frame.empty for frame in (daily, stock_basic, balancesheet, audit)):
            return False
        market = pd.DataFrame(
            {
                "datetime": pd.to_datetime(daily["trade_date"], errors="coerce"),
                "instrument": daily["ts_code"].map(_qlib_symbol),
                "amount": pd.to_numeric(daily["amount"], errors="coerce") * 1000.0,
                "paused": pd.to_numeric(daily["vol"], errors="coerce").fillna(0).le(0),
            }
        )
        listings = pd.DataFrame(
            {
                "instrument": stock_basic["ts_code"].map(_qlib_symbol),
                "list_date": pd.to_datetime(stock_basic["list_date"], errors="coerce"),
                "delist_date": pd.to_datetime(
                    stock_basic.get("delist_date"), errors="coerce"
                ),
            }
        )
        namechange = read("namechange")
        if not namechange.empty and {"ts_code", "name", "start_date"}.issubset(
            namechange.columns
        ):
            st_source = namechange[
                namechange["name"].astype(str).str.contains(r"(?:\*?ST|退)", regex=True)
            ]
            st_intervals = pd.DataFrame(
                {
                    "instrument": st_source["ts_code"].map(_qlib_symbol),
                    "start_date": st_source["start_date"],
                    "end_date": st_source.get("end_date"),
                    "is_st": True,
                }
            )
        else:
            st_intervals = pd.DataFrame(
                columns=["instrument", "start_date", "end_date", "is_st"]
            )
        suspend = read("suspend_d")
        suspension_rows: list[dict[str, Any]] = []
        if not suspend.empty and "ts_code" in suspend:
            date_column = next(
                (name for name in ("suspend_date", "trade_date") if name in suspend), None
            )
            if date_column:
                suspension_rows = [
                    {
                        "instrument": _qlib_symbol(row.ts_code),
                        "datetime": getattr(row, date_column),
                        "suspended": True,
                    }
                    for row in suspend.itertuples(index=False)
                ]
        suspensions = pd.DataFrame(
            suspension_rows,
            columns=["instrument", "datetime", "suspended"],
        )
        equity_column = next(
            (
                name
                for name in (
                    "total_hldr_eqy_exc_min_int",
                    "total_hldr_eqy_inc_min_int",
                    "total_hldr_eqy",
                )
                if name in balancesheet
            ),
            None,
        )
        if equity_column is None or "ann_date" not in balancesheet:
            raise ValueError("balancesheet has no announced shareholder equity")
        financials = pd.DataFrame(
            {
                "instrument": balancesheet["ts_code"].map(_qlib_symbol),
                "announcement_date": balancesheet["ann_date"],
                "equity": pd.to_numeric(balancesheet[equity_column], errors="coerce"),
            }
        )
        opinion_column = next(
            (name for name in ("audit_result", "audit_opinion") if name in audit), None
        )
        if opinion_column is None or "ann_date" not in audit:
            raise ValueError("fina_audit has no announced audit opinion")
        audits = pd.DataFrame(
            {
                "instrument": audit["ts_code"].map(_qlib_symbol),
                "announcement_date": audit["ann_date"],
                "audit_opinion": audit[opinion_column].astype(str),
            }
        )
        regulatory_source = read("regulatory_events")
        regulatory = None
        regulatory_origin: str | None = None
        if not regulatory_source.empty:
            required = {"ts_code", "event_date", "known_date", "major"}
            if not required.issubset(regulatory_source.columns):
                raise ValueError("regulatory event source violates its data contract")
            regulatory = regulatory_source.rename(columns={"ts_code": "instrument"}).copy()
            regulatory["instrument"] = regulatory["instrument"].map(_qlib_symbol)
            regulatory_origin = "materialized_dataset"
        else:
            # Fail-soft fallback: derive major-violation events from the anns_d
            # announcement titles persisted in the same immutable snapshot.
            regulatory = self._derive_regulatory_events(read)
            if regulatory is not None:
                regulatory_origin = f"anns_d_title_rules({REGULATORY_EVENTS_RULE_VERSION})"
        matrix = build_point_in_time_eligibility(
            market=market,
            listings=listings,
            st_intervals=st_intervals,
            suspensions=suspensions,
            financials=financials,
            audits=audits,
            regulatory_events=regulatory,
            policy=EligibilityPolicy(),
        )
        target.mkdir(parents=True, exist_ok=True)
        matrix.to_parquet(target / "eligibility_matrix.parquet", index=False, compression="zstd")
        (target / "eligibility_contract.json").write_text(
            json.dumps(
                {
                    "version": ELIGIBILITY_CONTRACT_VERSION,
                    "regulatory_data_available": regulatory is not None,
                    "regulatory_origin": regulatory_origin,
                    "financial_availability": "strictly_after_announcement_date",
                    "delisting_availability": "effective_date_only_no_backfill",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return True

    def _derive_regulatory_events(
        self, read: Callable[[str], pd.DataFrame]
    ) -> pd.DataFrame | None:
        """Derive major-violation events from snapshot anns_d titles.

        Returns None (current fail-soft behavior) when the snapshot carries no
        anns_d parquet; otherwise applies the versioned conservative title
        rules with the snapshot trade_cal as the known_date calendar. The
        derivation is deterministic over immutable snapshot inputs, so the
        snapshot lineage already covers the result.
        """

        anns_root = self.snapshot_path / "parquet" / "anns_d"
        if not anns_root.is_dir() or not any(anns_root.rglob("*.parquet")):
            return None
        open_days = open_days_from_trade_cal(read("trade_cal"))
        events = derive_regulatory_events(read("anns_d"), open_days)
        regulatory = events.rename(columns={"ts_code": "instrument"}).copy()
        regulatory["instrument"] = regulatory["instrument"].map(_qlib_symbol)
        return regulatory

    def _normalized_query(self, daily_glob: Path, adj_glob: Path, limit_glob: Path) -> str:
        daily = _sql_string(str(daily_glob.resolve()))
        adj = _sql_string(str(adj_glob.resolve()))
        limits = _sql_string(str(limit_glob.resolve()))
        daily_features = self.research_feature_contract["daily_fields"]
        fundamental_features = self.research_feature_contract["fundamental_fields"]
        daily_basic_root = self.snapshot_path / "parquet" / "daily_basic"

        joined_daily_select = ""
        daily_join = ""
        if daily_features:
            daily_basic = _sql_string(
                str((daily_basic_root / "**" / "*.parquet").resolve())
            )
            joined_daily_select = "".join(
                f"\n                    , try_cast(db.{field} AS DOUBLE) AS {field}"
                for field in daily_features
            )
            daily_join = f"""
                LEFT JOIN read_parquet({daily_basic}, hive_partitioning=true) db
                  ON d.ts_code = db.ts_code
                 AND try_cast(d.trade_date AS DATE) = try_cast(db.trade_date AS DATE)
            """

        joined_fundamental_select = ""
        fundamental_join = ""
        for dataset, features in fundamental_features.items():
            if not features:
                continue
            statement = _sql_string(
                str((self.snapshot_path / "parquet" / dataset / "**" / "*.parquet").resolve())
            )
            joined_fundamental_select += "".join(
                f"\n                    , try_cast({dataset}.{source} AS DOUBLE) AS {target}"
                for source, target in features.items()
            )
            projected_columns = ["ts_code", "ann_date", "end_date", *features]
            projected = ", ".join(projected_columns)
            revision_order = _fundamental_revision_order(
                projected_columns, self._parquet_columns(dataset)
            )
            fundamental_join += f"""
                ASOF LEFT JOIN (
                    SELECT {projected}
                    FROM read_parquet({statement}, hive_partitioning=true)
                    WHERE ts_code IS NOT NULL AND try_cast(ann_date AS DATE) IS NOT NULL
                    QUALIFY row_number() OVER (
                        PARTITION BY ts_code, try_cast(ann_date AS DATE)
                        ORDER BY {revision_order}
                    ) = 1
                ) {dataset}
                  ON d.ts_code = {dataset}.ts_code
                 AND try_cast(d.trade_date AS DATE) > try_cast({dataset}.ann_date AS DATE)
            """
        return f"""
            WITH joined AS (
                SELECT
                    d.ts_code,
                    d.trade_date,
                    d.open,
                    d.high,
                    d.low,
                    d.close,
                    d.vol,
                    d.amount,
                    d.pct_chg,
                    a.adj_factor,
                    l.up_limit,
                    l.down_limit
                    {joined_daily_select}
                    {joined_fundamental_select}
                    , first_value(d.close * a.adj_factor) OVER (
                        PARTITION BY d.ts_code ORDER BY d.trade_date
                    ) AS base_price
                FROM read_parquet({daily}, hive_partitioning=true) d
                LEFT JOIN read_parquet({adj}, hive_partitioning=true) a
                  ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
                LEFT JOIN read_parquet({limits}, hive_partitioning=true) l
                  ON d.ts_code = l.ts_code AND d.trade_date = l.trade_date
                {daily_join}
                {fundamental_join}
                WHERE d.ts_code IS NOT NULL AND d.close IS NOT NULL
            )
            SELECT
                trade_date AS date,
                upper(split_part(ts_code, '.', 2) || split_part(ts_code, '.', 1)) AS symbol,
                open * adj_factor / base_price AS open,
                high * adj_factor / base_price AS high,
                low * adj_factor / base_price AS low,
                close * adj_factor / base_price AS close,
                CASE
                    WHEN vol IS NOT NULL AND vol > 0 AND amount IS NOT NULL
                    THEN amount * 10.0 / vol * adj_factor / base_price
                    ELSE close * adj_factor / base_price
                END AS vwap,
                vol * {float(TUSHARE_HAND_SIZE)} * base_price / adj_factor AS volume,
                adj_factor / base_price AS factor,
                pct_chg / 100.0 AS change,
                -- Tushare amount is thousand-CNY; the Qlib field contract is CNY yuan
                amount * 1000.0 AS amount,
                CASE WHEN vol IS NULL OR vol <= 0 THEN 1.0 ELSE 0.0 END AS paused
                , up_limit * adj_factor / base_price AS up_limit
                , down_limit * adj_factor / base_price AS down_limit
                {''.join(f', {field}' for field in daily_features)}
                {''.join(
                    f', {target}'
                    for features in fundamental_features.values()
                    for target in features.values()
                )}
            FROM joined
            WHERE adj_factor IS NOT NULL AND adj_factor > 0 AND base_price > 0
        """

    @staticmethod
    def _invalid_daily_units_query(daily_glob: Path) -> str:
        """Reject daily rows whose Tushare amount/hand units imply an impossible VWAP."""

        daily = _sql_string(str(daily_glob.resolve()))
        return f"""
            SELECT count(*)
            FROM read_parquet({daily}, hive_partitioning=true)
            WHERE try_cast(vol AS DOUBLE) > 0
              AND (
                try_cast(amount AS DOUBLE) IS NULL
                OR try_cast(amount AS DOUBLE) <= 0
                OR try_cast(low AS DOUBLE) <= 0
                OR try_cast(high AS DOUBLE) < try_cast(low AS DOUBLE)
                OR try_cast(amount AS DOUBLE) * 10.0 / try_cast(vol AS DOUBLE)
                    < try_cast(low AS DOUBLE) * 0.95
                OR try_cast(amount AS DOUBLE) * 10.0 / try_cast(vol AS DOUBLE)
                    > try_cast(high AS DOUBLE) * 1.05
              )
        """

    def _research_feature_contract(self) -> dict[str, object]:
        daily_columns = self._parquet_columns("daily_basic")
        fundamental_columns = {
            dataset: self._parquet_columns(dataset)
            for dataset in _FUNDAMENTAL_RESEARCH_FIELDS
        }
        daily_fields = [field for field in _DAILY_RESEARCH_FIELDS if field in daily_columns]
        missing_daily_fields = [
            field for field in _DAILY_RESEARCH_FIELDS if field not in daily_columns
        ]
        fundamental_fields = {
            dataset: {
                source: target
                for source, target in mapping.items()
                if source in fundamental_columns[dataset]
            }
            for dataset, mapping in _FUNDAMENTAL_RESEARCH_FIELDS.items()
        }
        missing_fundamental_fields = {
            dataset: {
                source: target
                for source, target in mapping.items()
                if source not in fundamental_columns[dataset]
            }
            for dataset, mapping in _FUNDAMENTAL_RESEARCH_FIELDS.items()
        }
        fundamental_fields = {
            dataset: fields
            for dataset, fields in fundamental_fields.items()
            if fields
        }
        missing_fundamental_fields = {
            dataset: fields
            for dataset, fields in missing_fundamental_fields.items()
            if fields
        }
        # Distinguish "source column does not exist" from "source column
        # exists but holds no non-null value": both keep a field out of the
        # dumped binaries (an all-null channel carries no signal), but only
        # the latter proves the pipeline received the column.
        all_null_daily_fields = sorted(self._all_null_columns("daily_basic", daily_fields))
        all_null_fundamental_fields = {
            dataset: {
                source: target
                for source, target in fields.items()
                if source in self._all_null_columns(dataset, set(fields))
            }
            for dataset, fields in fundamental_fields.items()
        }
        all_null_fundamental_fields = {
            dataset: fields
            for dataset, fields in all_null_fundamental_fields.items()
            if fields
        }
        if missing_daily_fields or missing_fundamental_fields:
            logger.warning(
                "research field contract drift: declared source columns absent "
                "from snapshot parquets (fields skipped, not injected): "
                "daily_basic=%s fundamentals=%s",
                missing_daily_fields,
                missing_fundamental_fields,
            )
        if all_null_daily_fields or all_null_fundamental_fields:
            logger.warning(
                "research field sources contain only null values (fields "
                "injected as all-NaN channels): daily_basic=%s fundamentals=%s",
                all_null_daily_fields,
                all_null_fundamental_fields,
            )
        # Version 2: availability policies and recoverability levels come from
        # the shared registry in quant_data.availability and now also cover the
        # index/industry metadata consumed next to the feature fields.
        # Version 3: fundamental fields are grouped by source statement table
        # (fina_indicator plus the income/balancesheet/cashflow line items).
        # Version 4: declared-vs-available coverage diagnostics distinguish
        # source columns missing from the snapshot parquets from columns that
        # exist but are entirely null.
        availability_datasets = (
            "daily_basic",
            "fina_indicator",
            "income",
            "balancesheet",
            "cashflow",
            "index_weight",
            "index_member_all",
        )
        return {
            "version": 4,
            "daily_fields": daily_fields,
            "fundamental_fields": fundamental_fields,
            "fields": [
                *daily_fields,
                *(target for fields in fundamental_fields.values() for target in fields.values()),
            ],
            "missing_daily_fields": missing_daily_fields,
            "missing_fundamental_fields": missing_fundamental_fields,
            "all_null_daily_fields": all_null_daily_fields,
            "all_null_fundamental_fields": all_null_fundamental_fields,
            "availability_policy_version": AVAILABILITY_POLICY_VERSION,
            "availability_policy": {
                dataset: availability_contract_label(dataset)
                for dataset in availability_datasets
            },
            "recoverability": {
                dataset: recoverability_level(dataset)
                for dataset in availability_datasets
            },
        }

    def _all_null_columns(self, dataset: str, columns: Collection[str]) -> set[str]:
        """Source columns present in the dataset schema with zero non-null rows."""

        if not columns:
            return set()
        root = self.snapshot_path / "parquet" / dataset
        if not root.exists() or not any(root.rglob("*.parquet")):
            return set()
        glob = _sql_string(str((root / "**" / "*.parquet").resolve()))
        ordered = sorted(columns)
        projection = ", ".join(
            f'count("{column.replace(chr(34), chr(34) * 2)}") AS "c{index}"'
            for index, column in enumerate(ordered)
        )
        connection = duckdb.connect()
        try:
            row = connection.execute(
                f"SELECT {projection} FROM read_parquet({glob}, "
                "hive_partitioning=true, union_by_name=true)"
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return set()
        return {
            column
            for index, column in enumerate(ordered)
            if int(row[index] or 0) == 0
        }

    def _validate_research_sources(self) -> None:
        """Reject a price-only snapshot before it can become a research dataset."""

        issues: list[str] = []
        required_columns = {
            "daily_basic": {"ts_code", "trade_date", "total_mv"},
            "fina_indicator": {"ts_code", "ann_date", "end_date"},
            "index_weight": {"index_code", "con_code", "trade_date", "weight"},
            "stock_basic": {"ts_code", "list_date"},
            "balancesheet": {"ts_code", "ann_date"},
            "fina_audit": {"ts_code", "ann_date", "audit_result"},
            "namechange": {"ts_code", "name", "start_date"},
        }
        columns_by_dataset = {
            dataset: self._parquet_columns(dataset)
            for dataset in (*required_columns, "index_member_all")
        }
        for dataset, required in required_columns.items():
            columns = columns_by_dataset[dataset]
            if not columns:
                issues.append(f"missing {dataset}")
                continue
            missing = sorted(required - columns)
            if missing:
                issues.append(f"{dataset} missing columns: {', '.join(missing)}")

        financial_columns = columns_by_dataset["fina_indicator"]
        if financial_columns and not set(
            _FUNDAMENTAL_RESEARCH_FIELDS["fina_indicator"]
        ).intersection(financial_columns):
            issues.append("fina_indicator has no supported financial factor columns")

        industry_columns = columns_by_dataset["index_member_all"]
        if not industry_columns:
            issues.append("missing index_member_all")
        else:
            if not {"ts_code", "con_code"}.intersection(industry_columns):
                issues.append("index_member_all has no stock-code column")
            if not {"l1_code", "index_code", "l2_code"}.intersection(industry_columns):
                issues.append("index_member_all has no industry-code column")
            if "in_date" not in industry_columns:
                issues.append("index_member_all missing columns: in_date")

        if not issues:
            usable_predicates = {
                "daily_basic": (
                    f"ts_code IS NOT NULL AND {_as_date_sql('trade_date')} IS NOT NULL "
                    "AND try_cast(total_mv AS DOUBLE) > 0"
                ),
                "fina_indicator": (
                    f"ts_code IS NOT NULL AND {_as_date_sql('ann_date')} IS NOT NULL "
                    f"AND {_as_date_sql('end_date')} IS NOT NULL AND ("
                    + " OR ".join(
                        f"try_cast({field} AS DOUBLE) IS NOT NULL"
                        for field in _FUNDAMENTAL_RESEARCH_FIELDS["fina_indicator"]
                        if field in financial_columns
                    )
                    + ")"
                ),
                "index_member_all": self._industry_usable_predicate(industry_columns),
                "index_weight": (
                    "index_code IS NOT NULL AND con_code IS NOT NULL "
                    f"AND {_as_date_sql('trade_date')} IS NOT NULL "
                    "AND try_cast(weight AS DOUBLE) > 0"
                ),
                "stock_basic": (
                    f"ts_code IS NOT NULL AND {_as_date_sql('list_date')} IS NOT NULL"
                ),
                "balancesheet": (
                    f"ts_code IS NOT NULL AND {_as_date_sql('ann_date')} IS NOT NULL"
                ),
                "fina_audit": (
                    f"ts_code IS NOT NULL AND {_as_date_sql('ann_date')} IS NOT NULL "
                    "AND audit_result IS NOT NULL"
                ),
                "namechange": (
                    f"ts_code IS NOT NULL AND {_as_date_sql('start_date')} IS NOT NULL "
                    "AND name IS NOT NULL"
                ),
            }
            for dataset, predicate in usable_predicates.items():
                if not self._has_usable_row(dataset, predicate):
                    issues.append(f"{dataset} has no usable rows")

        if not issues:
            coverage_issue = self._benchmark_industry_coverage_issue(
                industry_columns
            )
            if coverage_issue:
                issues.append(coverage_issue)

        if issues:
            raise RuntimeError("Qlib research inputs are incomplete: " + "; ".join(issues))

    def _benchmark_industry_coverage_issue(
        self, industry_columns: set[str]
    ) -> str | None:
        """Require point-in-time industry coverage for the governed benchmark.

        The benchmark-relative optimizer rejects constituents without an
        industry. Catch an incomplete/capped ``index_member_all`` snapshot here
        instead of allowing Qlib generation to succeed and failing much later
        during the formal backtest.
        """

        instrument_column = next(
            name for name in ("ts_code", "con_code") if name in industry_columns
        )
        industry_column = next(
            name
            for name in ("l1_code", "index_code", "l2_code")
            if name in industry_columns
        )
        out_date = (
            _as_date_sql("out_date")
            if "out_date" in industry_columns
            else "NULL::DATE"
        )
        weight_root = self.snapshot_path / "parquet" / "index_weight"
        industry_root = self.snapshot_path / "parquet" / "index_member_all"
        weight_glob = _sql_string(
            str((weight_root / "**" / "*.parquet").resolve())
        )
        industry_glob = _sql_string(
            str((industry_root / "**" / "*.parquet").resolve())
        )
        benchmark = _sql_string(_GOVERNED_BENCHMARK)
        query = f"""
            WITH weight_rows AS (
                SELECT
                    upper(trim(CAST(index_code AS VARCHAR))) AS benchmark,
                    upper(trim(CAST(con_code AS VARCHAR))) AS instrument,
                    {_as_date_sql("trade_date")} AS weight_date,
                    try_cast(weight AS DOUBLE) AS weight
                FROM read_parquet(
                    {weight_glob}, hive_partitioning=true, union_by_name=true
                )
            ),
            constituents AS (
                SELECT DISTINCT w.weight_date, w.instrument
                FROM weight_rows w
                WHERE w.benchmark = {benchmark}
                  AND w.instrument IS NOT NULL
                  AND w.weight_date IS NOT NULL
                  AND w.weight > 0
            ),
            industry_rows AS (
                SELECT
                    upper(trim(CAST("{instrument_column}" AS VARCHAR))) AS instrument,
                    {_as_date_sql("in_date")} AS in_date,
                    {out_date} AS out_date
                FROM read_parquet(
                    {industry_glob}, hive_partitioning=true, union_by_name=true
                )
                WHERE "{instrument_column}" IS NOT NULL
                  AND "{industry_column}" IS NOT NULL
                  AND {_as_date_sql("in_date")} IS NOT NULL
            ),
            coverage AS (
                SELECT
                    c.weight_date,
                    c.instrument,
                    count(i.instrument) > 0 AS covered
                FROM constituents c
                LEFT JOIN industry_rows i
                  ON i.instrument = c.instrument
                 AND i.in_date <= c.weight_date
                 AND (i.out_date IS NULL OR i.out_date >= c.weight_date)
                GROUP BY c.weight_date, c.instrument
            )
            SELECT
                count(*) AS total_rows,
                count(DISTINCT weight_date) AS benchmark_dates,
                count(*) FILTER (WHERE NOT covered) AS missing_rows,
                min(weight_date) FILTER (WHERE NOT covered) AS first_missing_date,
                (
                    SELECT string_agg(instrument, ', ')
                    FROM (
                        SELECT instrument
                        FROM coverage
                        WHERE NOT covered
                        ORDER BY weight_date, instrument
                        LIMIT 10
                    ) examples
                ) AS examples
            FROM coverage
        """
        connection = duckdb.connect()
        try:
            row = connection.execute(query).fetchone()
        finally:
            connection.close()
        total_rows = int(row[0] or 0) if row is not None else 0
        if total_rows == 0:
            return (
                "index_weight has no positive constituents for governed "
                f"benchmark {_GOVERNED_BENCHMARK}"
            )
        missing_rows = int(row[2] or 0)
        if missing_rows == 0:
            return None
        benchmark_dates = int(row[1] or 0)
        first_missing_date = row[3]
        examples = str(row[4] or "")
        return (
            f"index_member_all has no active point-in-time industry for "
            f"{missing_rows}/{total_rows} {_GOVERNED_BENCHMARK} constituent-date "
            f"rows across {benchmark_dates} benchmark dates; first affected date "
            f"{first_missing_date}: {examples}"
        )

    @staticmethod
    def _industry_usable_predicate(columns: set[str]) -> str:
        instrument = next(
            name for name in ("ts_code", "con_code") if name in columns
        )
        industry = next(
            name for name in ("l1_code", "index_code", "l2_code") if name in columns
        )
        return (
            f"{instrument} IS NOT NULL AND {industry} IS NOT NULL "
            f"AND {_as_date_sql('in_date')} IS NOT NULL"
        )

    def _has_usable_row(self, dataset: str, predicate: str) -> bool:
        root = self.snapshot_path / "parquet" / dataset
        glob = _sql_string(str((root / "**" / "*.parquet").resolve()))
        connection = duckdb.connect()
        try:
            row = connection.execute(
                f"SELECT 1 FROM read_parquet({glob}, hive_partitioning=true, "
                f"union_by_name=true) WHERE {predicate} LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
        return row is not None

    def _parquet_columns(self, dataset: str) -> set[str]:
        root = self.snapshot_path / "parquet" / dataset
        if not root.exists() or not any(root.rglob("*.parquet")):
            return set()
        glob = _sql_string(str((root / "**" / "*.parquet").resolve()))
        connection = duckdb.connect()
        try:
            rows = connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet({glob}, hive_partitioning=true, "
                "union_by_name=true)"
            ).fetchall()
        finally:
            connection.close()
        return {str(row[0]) for row in rows}

    @staticmethod
    def _missing_market_controls_query(daily_glob: Path, adj_glob: Path, limit_glob: Path) -> str:
        daily = _sql_string(str(daily_glob.resolve()))
        adj = _sql_string(str(adj_glob.resolve()))
        limits = _sql_string(str(limit_glob.resolve()))
        return f"""
            SELECT count(*)
            FROM read_parquet({daily}, hive_partitioning=true) d
            LEFT JOIN read_parquet({adj}, hive_partitioning=true) a
              ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
            LEFT JOIN read_parquet({limits}, hive_partitioning=true) l
              ON d.ts_code = l.ts_code AND d.trade_date = l.trade_date
            WHERE d.ts_code IS NOT NULL AND d.close IS NOT NULL
              AND (
                a.adj_factor IS NULL OR a.adj_factor <= 0
                OR l.up_limit IS NULL OR l.down_limit IS NULL
                OR (
                  (l.up_limit <= 0 OR l.down_limit <= 0)
                  AND NOT (l.up_limit >= 99999.0 AND l.down_limit = 0)
                )
              )
        """


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _read_parquet_columns(
    files: list[Path], wanted: set[str], *, required: set[str] | None = None
) -> pd.DataFrame | None:
    """Concatenate parquet files projected to the wanted columns they have."""

    needed = set(wanted) if required is None else set(required)
    frames = []
    for path in files:
        available = wanted.intersection(pq.read_schema(path).names)
        if not needed.issubset(available):
            continue
        frames.append(pd.read_parquet(path, columns=sorted(available)))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def _as_date_sql(column: str) -> str:
    identifier = '"' + column.replace('"', '""') + '"'
    return (
        f"coalesce(try_cast({identifier} AS DATE), "
        f"try_strptime(CAST({identifier} AS VARCHAR), '%Y%m%d')::DATE)"
    )


def _fundamental_revision_order(
    projected_columns: list[str], source_columns: set[str]
) -> str:
    """Deterministic total order for conflicting financial revision rows.

    Rows sharing (ts_code, ann_date, end_date) conflict when a report is
    re-announced or silently revised and both versions survive in the snapshot.
    Resolve them deterministically: newest f_ann_date / update_flag when the
    source provides them, then the newest row-level ingested_at, and finally a
    content hash over the projected columns so the chosen row never depends on
    parquet file or row order.
    """

    ordering = ["try_cast(end_date AS DATE) DESC NULLS LAST"]
    if "f_ann_date" in source_columns:
        ordering.append("try_cast(f_ann_date AS DATE) DESC NULLS LAST")
    if "update_flag" in source_columns:
        ordering.append("try_cast(update_flag AS DOUBLE) DESC NULLS LAST")
    if "ingested_at" in source_columns:
        ordering.append("ingested_at DESC NULLS LAST")
    hashed = ", ".join(
        f'coalesce(CAST("{column}" AS VARCHAR), \'\')' for column in projected_columns
    )
    ordering.append(f"md5(concat_ws('|', {hashed})) ASC")
    return ", ".join(ordering)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qlib_symbol(value: object) -> str | None:
    text = str(value or "")
    if "." not in text:
        return None
    code, exchange = text.split(".", 1)
    return f"{exchange.upper()}{code}"
