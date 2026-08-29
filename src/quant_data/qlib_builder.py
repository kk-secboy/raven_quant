from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
from collections.abc import Callable, Collection
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_platform.eligibility import (
    ELIGIBILITY_CONTRACT_VERSION,
    EligibilityPolicy,
    build_point_in_time_eligibility,
)
from quant_platform.style_exposures import STYLE_COLUMNS, standardize_panel

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
    QLIB_OUTPUT_MANIFEST_VERSION,
    TUSHARE_DAILY_AMOUNT_UNIT,
    TUSHARE_DAILY_VOLUME_UNIT,
    TUSHARE_HAND_SIZE,
)
from .history_bounds import PRIMARY_MARKET_HISTORY_START
from .path_utils import to_wsl_path as _to_wsl_path
from .regulatory_events import (
    REGULATORY_EVENTS_RULE_VERSION,
    REGULATORY_TERMINAL_DEFERRAL_POLICY,
    derive_regulatory_events_for_horizon,
    open_days_from_trade_cal,
)
from .snapshot_lineage import verify_snapshot_lineage
from .style_exposure_panel import build_adjusted_close, build_raw_style_panel

logger = logging.getLogger(__name__)

_QLIB_PROVENANCE_PATH = "metadata/provenance.json"

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
_UNKNOWN_INDUSTRY = "__UNKNOWN__"
# Cap unresolved benchmark industry exposure at 2%.  This is conservative in
# the risk dimension: the observed 2.2169% pre-remediation gap remains
# fail-closed, while the source-backed 1.49% residual may pass only as explicit
# ``__UNKNOWN__`` and is never future-filled.
_MAX_UNKNOWN_BENCHMARK_WEIGHT_RATIO = 0.02
_UNRESTRICTED_UP_LIMIT = 99999.99
_MAX_EXCLUDED_DAILY_UNIT_RATIO = 0.00001

# A full-market daily build contains wide ASOF joins, ordered windows and
# cross-sectional metadata.  DuckDB otherwise inherits host-wide defaults
# (roughly 80% of RAM and every visible core), which can make one durable
# data_qlib job starve PostgreSQL, SSH and the API.  Spill belongs beside the
# staging attempt on the governed data volume, never on the container root.
DAILY_QLIB_DUCKDB_MEMORY_LIMIT = "8GB"
DAILY_QLIB_DUCKDB_THREADS = 8
DAILY_QLIB_DUMP_WORKERS = 4
DAILY_QLIB_STYLE_SYMBOL_BATCH = 128
DAILY_QLIB_ELIGIBILITY_SYMBOL_BATCH = 128

_ADJUSTMENT_BOUNDARY_POLICY_VERSION = "baostock-primary-adj-boundary-v1"
_ADJUSTMENT_BOUNDARY_MAX_PRICE_ABS_ERROR = 0.051
_ADJUSTMENT_BOUNDARY_MAX_PRICE_RELATIVE_ERROR = 0.005
_ADJUSTMENT_BOUNDARY_MAX_MASKED_RATIO = 0.01

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

_DAILY_RESEARCH_FIELD_UNITS = {
    "turnover_rate": "percent",
    "turnover_rate_f": "percent",
    "volume_ratio": "ratio_unitless",
    "pe_ttm": "ratio_unitless",
    "pb": "ratio_unitless",
    "ps_ttm": "ratio_unitless",
    "dv_ttm": "percent",
    "total_mv": "cny_ten_thousand",
    "circ_mv": "cny_ten_thousand",
}

# The official ``moneyflow`` endpoint is a per-stock, per-session order-flow
# decomposition.  Source amounts are ten-thousand CNY and are known only after
# that session closes.  Keep the Qlib surface compact: one amount channel plus
# two scale-free descriptors are enough for RD-Agent/Qlib to research capital
# flow without exposing nine highly collinear raw order-size columns.
_CAPITAL_FLOW_FEATURE_REQUIREMENTS = {
    "mf_net_inflow_amount": frozenset({"net_mf_amount"}),
    "mf_net_inflow_ratio": frozenset({"net_mf_amount"}),
    "mf_large_order_imbalance": frozenset(
        {
            "buy_lg_amount",
            "sell_lg_amount",
            "buy_elg_amount",
            "sell_elg_amount",
        }
    ),
}

_CAPITAL_FLOW_FIELD_UNITS = {
    "mf_net_inflow_amount": "cny_yuan",
    "mf_net_inflow_ratio": "ratio_unitless",
    "mf_large_order_imbalance": "ratio_unitless",
}

# Fundamental research fields dumped into the Qlib binaries, grouped by the
# source statement table. Every table feeds the same point-in-time channel:
# an ASOF join keyed on the announcement date (trade_date > ann_date), never
# on the report period (end_date), so no field here can leak an unpublished
# report. Fundamentals are never price-normalized: absolute amounts stay in
# CNY yuan and per-share values in yuan per share.
#
# q_profit_yoy, inv_turn, ocf_to_or, ocf_to_profit and salescash_to_or are
# documented Tushare fina_indicator output columns (doc_id=79) but flagged
# non-default there. The downloader therefore stores a narrow, versioned
# companion dataset and joins it through the same announcement-date ASOF
# channel; diagnostics still fail closed if the relay omits a requested field.
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
        "eps": "fund_eps",
        "bps": "fund_bps",
        "ocfps": "fund_ocfps",
        "roe_waa": "fund_roe_weighted",
        "roe_dt": "fund_roe_diluted",
        "roic": "fund_roic",
        "netprofit_margin": "fund_netprofit_margin",
        "assets_turn": "fund_assets_turnover",
        "ar_turn": "fund_receivables_turnover",
        "quick_ratio": "fund_quick_ratio",
        "debt_to_eqt": "fund_debt_to_equity",
        "saleexp_to_gr": "fund_sales_expense_ratio",
        "adminexp_of_gr": "fund_admin_expense_ratio",
        "finaexp_of_gr": "fund_finance_expense_ratio",
        "op_yoy": "fund_op_profit_yoy",
        "equity_yoy": "fund_equity_yoy",
        "interestdebt": "fund_interest_debt",
    },
    "fina_indicator_nondefault": {
        "q_profit_yoy": "fund_quarter_profit_yoy",
        "inv_turn": "fund_inventory_turnover",
        "ocf_to_or": "fund_ocf_to_revenue",
        "ocf_to_profit": "fund_ocf_to_profit",
        "salescash_to_or": "fund_sales_cash_to_revenue",
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


def qlib_research_field_catalog() -> dict[str, dict[str, str]]:
    """Return the governed field surface exposed to factor research.

    This is a schema contract, not proof that every optional field exists in a
    particular dataset snapshot. Consumers must still compare it with the
    sealed Qlib provenance before accepting a factor definition.
    """

    catalog: dict[str, dict[str, str]] = {
        name: {
            "unit": unit,
            "availability": "after_same_session_close",
            "source": "daily_market",
        }
        for name, unit in _DAILY_FIELD_UNITS.items()
    }
    catalog.update(
        {
            name: {
                "unit": unit,
                "availability": "after_same_session_close",
                "source": "daily_basic",
            }
            for name, unit in _DAILY_RESEARCH_FIELD_UNITS.items()
        }
    )
    catalog.update(
        {
            name: {
                "unit": unit,
                "availability": "next_session_after_announcement",
                "source": "fundamental_pit",
            }
            for name, unit in _FUNDAMENTAL_FIELD_UNITS.items()
        }
    )
    catalog.update(
        {
            name: {
                "unit": unit,
                "availability": "after_same_session_close",
                "source": "moneyflow",
            }
            for name, unit in _CAPITAL_FLOW_FIELD_UNITS.items()
        }
    )
    return dict(sorted(catalog.items()))


class QlibBuilder:
    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path.resolve()
        self._parquet_columns_cache: dict[str, set[str]] = {}
        self.research_feature_contract = self._research_feature_contract()
        self._daily_unit_quality_cache: dict[str, Any] | None = None
        self._adjustment_boundary_cache: dict[str, Any] | None = None

    @property
    def qlib_fields(self) -> tuple[str, ...]:
        return (*_BASE_QLIB_FIELDS, *self.research_feature_contract["fields"])

    @staticmethod
    def builder_sha256() -> str:
        module_root = Path(__file__).resolve().parent
        project_root = module_root.parent
        files = {
            "qlib_builder": Path(__file__).resolve(),
            "availability": module_root / "availability.py",
            "execution_contract": module_root / "execution_contract.py",
            "eligibility": project_root / "quant_platform" / "eligibility.py",
            "regulatory_events": module_root / "regulatory_events.py",
            "style_exposure_panel": module_root / "style_exposure_panel.py",
            "style_exposures": project_root / "quant_platform" / "style_exposures.py",
        }
        contract = {
            name: _sha256_file(path) for name, path in sorted(files.items())
        }
        return hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _duckdb_connection(
        *, spill_dir: Path | None = None
    ) -> duckdb.DuckDBPyConnection:
        """Open one host-safe DuckDB connection for daily publication work."""

        connection = duckdb.connect()
        try:
            connection.execute(
                f"SET memory_limit='{DAILY_QLIB_DUCKDB_MEMORY_LIMIT}'"
            )
            connection.execute(f"SET threads={DAILY_QLIB_DUCKDB_THREADS}")
            connection.execute("SET preserve_insertion_order=false")
            if spill_dir is not None:
                spill_dir.mkdir(parents=True, exist_ok=True)
                connection.execute(
                    f"SET temp_directory={_sql_string(str(spill_dir.resolve()))}"
                )
        except Exception:
            connection.close()
            raise
        return connection

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
        spill_dir = temporary / "duckdb_spill"
        partitions.mkdir(parents=True, exist_ok=True)
        by_symbol.mkdir(parents=True, exist_ok=True)

        connection = self._duckdb_connection(spill_dir=spill_dir)
        try:
            query = self._normalized_query(daily_glob, adj_glob, limit_glob)
            invalid = connection.execute(
                self._missing_market_controls_query(daily_glob, adj_glob, limit_glob)
            ).fetchone()[0]
            if invalid:
                raise RuntimeError(
                    f"{invalid} daily rows have an invalid adjustment factor or "
                    "partial/malformed price limits"
                )
            daily_unit_quality = self._daily_unit_quality_coverage()
            invalid_units = int(daily_unit_quality["excluded_rows"])
            if invalid_units and float(daily_unit_quality["excluded_ratio"]) > float(
                daily_unit_quality["max_excluded_ratio"]
            ):
                raise RuntimeError(
                    f"{invalid_units} daily rows violate the Tushare hand/amount price contract"
                )
            if invalid_units:
                logger.warning(
                    "excluding %s/%s daily rows with internally inconsistent price/volume/amount "
                    "units from research history",
                    invalid_units,
                    daily_unit_quality["total_rows"],
                )
            connection.execute(
                f"COPY ({query}) TO {_sql_string(str(partitions))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (symbol), "
                "ROW_GROUP_SIZE 100000)"
            )
        finally:
            connection.close()
            shutil.rmtree(spill_dir, ignore_errors=True)

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
        max_workers: int = DAILY_QLIB_DUMP_WORKERS,
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

        units = {**_DAILY_FIELD_UNITS, **_DAILY_RESEARCH_FIELD_UNITS}
        for field in self.research_feature_contract["fields"]:
            unit = _FUNDAMENTAL_FIELD_UNITS.get(field) or _CAPITAL_FLOW_FIELD_UNITS.get(
                field
            )
            if unit is not None:
                units[field] = unit
        return units

    def _write_provenance(self, qlib_dir: Path) -> None:
        snapshot_digest = self._snapshot_manifest_digest()
        snapshot_manifest = json.loads(
            (self.snapshot_path / "manifest.json").read_text(encoding="utf-8")
        )
        builder_digest = self.builder_sha256()
        fields = list(self.qlib_fields)
        field_units = self._field_units()
        execution_controls = self._execution_control_coverage()
        daily_unit_quality = self._daily_unit_quality_coverage()
        adjustment_boundary = self._require_adjustment_boundary_evidence()
        adjustment_boundary_contract = {
            key: adjustment_boundary.get(key)
            for key in (
                "version",
                "status",
                "cutoff_date",
                "policy",
                "max_price_abs_error",
                "max_price_relative_error",
                "max_masked_ratio",
                "cross_source_symbols",
                "rebased_symbol_count",
                "masked_symbol_count",
                "masked_ratio",
                "evidence_sha256",
            )
        }
        adjustment_boundary_contract["artifact_path"] = (
            "metadata/adjustment_boundary.json"
        )
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
            "industry_missing_value_policy": {
                "label": _UNKNOWN_INDUSTRY,
                "max_benchmark_weight_ratio": _MAX_UNKNOWN_BENCHMARK_WEIGHT_RATIO,
                "method": "bounded_source_gap_intervals_without_future_fill",
            },
            "execution_controls": execution_controls,
            "daily_unit_quality": daily_unit_quality,
            "adjustment_boundary": adjustment_boundary_contract,
        }
        contract_sha256 = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        source_lineage_id = str(snapshot_manifest.get("lineage_id") or "")
        lineage_verified = False
        if source_lineage_id:
            verified_manifest = verify_snapshot_lineage(self.snapshot_path)
            if verified_manifest != snapshot_manifest:
                raise ValueError("snapshot lineage verification returned different manifest data")
            lineage_verified = True
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
            "industry_missing_value_policy": {
                "label": _UNKNOWN_INDUSTRY,
                "max_benchmark_weight_ratio": _MAX_UNKNOWN_BENCHMARK_WEIGHT_RATIO,
                "method": "bounded_source_gap_intervals_without_future_fill",
            },
            "execution_controls": execution_controls,
            "daily_unit_quality": daily_unit_quality,
            "adjustment_boundary": adjustment_boundary_contract,
        }
        canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        target = qlib_dir / _QLIB_PROVENANCE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        (target.parent / "adjustment_boundary.json").write_text(
            json.dumps(adjustment_boundary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (target.parent / "research_feature_contract.json").write_text(
            json.dumps(self.research_feature_contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
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
            "output_manifest": build_qlib_output_manifest(qlib_dir),
            "created_at": datetime.now(UTC).isoformat(),
        }
        target.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

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

    def _execution_control_coverage(self) -> dict[str, Any]:
        daily_files = list((self.snapshot_path / "parquet" / "daily").rglob("*.parquet"))
        limit_files = list((self.snapshot_path / "parquet" / "stk_limit").rglob("*.parquet"))
        if not daily_files or not limit_files:
            return {
                "source": "native_stk_limit",
                "missing_rows": 0,
                "total_rows": 0,
                "first_missing_date": None,
                "last_missing_date": None,
                "native_complete_from": None,
                "missing_row_policy": "research_only_unrestricted_sentinel",
                "formal_execution_requires_native_controls": True,
            }
        daily = _sql_string(
            str((self.snapshot_path / "parquet" / "daily" / "**" / "*.parquet").resolve())
        )
        limits = _sql_string(
            str(
                (self.snapshot_path / "parquet" / "stk_limit" / "**" / "*.parquet").resolve()
            )
        )
        connection = self._duckdb_connection()
        try:
            row = connection.execute(
                f"""
                WITH coverage AS (
                    SELECT
                        try_cast(d.trade_date AS DATE) AS trade_date,
                        l.up_limit,
                        l.down_limit
                    FROM read_parquet(
                        {daily}, hive_partitioning=true, union_by_name=true
                    ) d
                    LEFT JOIN read_parquet(
                        {limits}, hive_partitioning=true, union_by_name=true
                    ) l
                      ON d.ts_code = l.ts_code AND d.trade_date = l.trade_date
                    WHERE d.ts_code IS NOT NULL AND d.close IS NOT NULL
                )
                SELECT
                    count(*) FILTER (
                        WHERE up_limit IS NULL AND down_limit IS NULL
                    ) AS missing_rows,
                    min(trade_date) FILTER (
                        WHERE up_limit IS NULL AND down_limit IS NULL
                    ) AS first_missing_date,
                    max(trade_date) FILTER (
                        WHERE up_limit IS NULL AND down_limit IS NULL
                    ) AS last_missing_date,
                    count(*) AS total_rows,
                    min(trade_date) AS first_trade_date
                FROM coverage
                """
            ).fetchone()
        finally:
            connection.close()
        missing_rows = int(row[0] or 0) if row is not None else 0
        first_missing = row[1] if row is not None else None
        last_missing = row[2] if row is not None else None
        total_rows = int(row[3] or 0) if row is not None else 0
        first_trade = row[4] if row is not None else None
        native_complete_from = (
            (last_missing + timedelta(days=1)).isoformat()
            if last_missing is not None
            else (str(first_trade) if first_trade is not None else None)
        )
        return {
            "source": "native_stk_limit",
            "missing_rows": missing_rows,
            "total_rows": total_rows,
            "first_missing_date": str(first_missing) if first_missing is not None else None,
            "last_missing_date": str(last_missing) if last_missing is not None else None,
            "native_complete_from": native_complete_from,
            "missing_row_policy": "research_only_unrestricted_sentinel",
            "formal_execution_requires_native_controls": True,
        }

    def _adjustment_boundary_evidence(self) -> dict[str, Any]:
        """Prove and freeze the BaoStock-to-primary adjustment-factor bridge.

        Adjustment factors are only defined up to a positive per-instrument
        constant.  BaoStock and the primary source can therefore agree on the
        entire relative path while using different absolute levels.  A direct
        concatenation would create a fake adjusted-price jump at 2016-01-01.

        For instruments with usable daily/factor rows on both sides, the last
        legacy factor is scaled to the first primary factor.  This is admitted
        only when the legacy close and the primary row's ``pre_close`` prove
        price continuity.  Failed bridges are explicitly masked from Qlib;
        immutable snapshot values are never overwritten.
        """

        if self._adjustment_boundary_cache is not None:
            return json.loads(json.dumps(self._adjustment_boundary_cache))
        daily_root = self.snapshot_path / "parquet" / "daily"
        adj_root = self.snapshot_path / "parquet" / "adj_factor"
        if not any(daily_root.rglob("*.parquet")) or not any(
            adj_root.rglob("*.parquet")
        ):
            result = {
                "version": _ADJUSTMENT_BOUNDARY_POLICY_VERSION,
                "status": "not_applicable",
                "cutoff_date": PRIMARY_MARKET_HISTORY_START.isoformat(),
                "policy": "rebase_legacy_factor_or_mask_cross_source_instrument",
                "max_price_abs_error": _ADJUSTMENT_BOUNDARY_MAX_PRICE_ABS_ERROR,
                "max_price_relative_error": (
                    _ADJUSTMENT_BOUNDARY_MAX_PRICE_RELATIVE_ERROR
                ),
                "max_masked_ratio": _ADJUSTMENT_BOUNDARY_MAX_MASKED_RATIO,
                "cross_source_symbols": 0,
                "rebased_symbol_count": 0,
                "masked_symbol_count": 0,
                "masked_ratio": 0.0,
                "rebased_symbols": [],
                "masked_symbols": [],
            }
            result["evidence_sha256"] = _canonical_sha256(result)
            self._adjustment_boundary_cache = result
            return json.loads(json.dumps(result))

        daily = _sql_string(str((daily_root / "**" / "*.parquet").resolve()))
        adj = _sql_string(str((adj_root / "**" / "*.parquet").resolve()))
        cutoff = _sql_string(PRIMARY_MARKET_HISTORY_START.isoformat())
        pre_close_expression = (
            "try_cast(d.pre_close AS DOUBLE)"
            if "pre_close" in self._parquet_columns("daily")
            else "NULL::DOUBLE"
        )
        connection = self._duckdb_connection()
        try:
            rows = connection.execute(
                f"""
                WITH daily_rows AS (
                    SELECT
                        d.ts_code,
                        coalesce(
                            try_cast(d.trade_date AS DATE),
                            try_strptime(
                                CAST(d.trade_date AS VARCHAR), '%Y%m%d'
                            )::DATE
                        ) AS trade_date,
                        try_cast(d.close AS DOUBLE) AS close,
                        {pre_close_expression} AS pre_close
                    FROM read_parquet(
                        {daily}, hive_partitioning=true, union_by_name=true
                    ) d
                    WHERE d.ts_code IS NOT NULL
                      AND d.close IS NOT NULL
                ),
                factor_rows AS (
                    SELECT
                        a.ts_code,
                        coalesce(
                            try_cast(a.trade_date AS DATE),
                            try_strptime(
                                CAST(a.trade_date AS VARCHAR), '%Y%m%d'
                            )::DATE
                        ) AS trade_date,
                        CASE
                            WHEN count(*) = 1
                            THEN max(try_cast(a.adj_factor AS DOUBLE))
                            ELSE NULL::DOUBLE
                        END AS adj_factor
                    FROM read_parquet(
                        {adj}, hive_partitioning=true, union_by_name=true
                    ) a
                    WHERE a.ts_code IS NOT NULL
                    GROUP BY a.ts_code, trade_date
                ),
                legacy_anchor AS (
                    SELECT ts_code, trade_date, close
                    FROM daily_rows
                    WHERE trade_date < DATE {cutoff}
                    QUALIFY row_number() OVER (
                        PARTITION BY ts_code ORDER BY trade_date DESC
                    ) = 1
                ),
                primary_anchor AS (
                    SELECT ts_code, trade_date, pre_close
                    FROM daily_rows
                    WHERE trade_date >= DATE {cutoff}
                    QUALIFY row_number() OVER (
                        PARTITION BY ts_code ORDER BY trade_date ASC
                    ) = 1
                )
                SELECT
                    legacy.ts_code,
                    legacy.trade_date,
                    current_anchor.trade_date,
                    legacy.close,
                    current_anchor.pre_close,
                    legacy_factor.adj_factor,
                    current_factor.adj_factor
                FROM legacy_anchor legacy
                INNER JOIN primary_anchor current_anchor USING (ts_code)
                LEFT JOIN factor_rows legacy_factor
                  ON legacy.ts_code = legacy_factor.ts_code
                 AND legacy.trade_date = legacy_factor.trade_date
                LEFT JOIN factor_rows current_factor
                  ON current_anchor.ts_code = current_factor.ts_code
                 AND current_anchor.trade_date = current_factor.trade_date
                ORDER BY legacy.ts_code
                """
            ).fetchall()
        finally:
            connection.close()

        rebased: list[dict[str, Any]] = []
        masked: list[dict[str, Any]] = []
        for (
            ts_code,
            legacy_date,
            primary_date,
            legacy_close,
            primary_pre_close,
            legacy_factor,
            primary_factor,
        ) in rows:
            legacy_price = _finite_float(legacy_close)
            previous_price = _finite_float(primary_pre_close)
            legacy_level = _finite_float(legacy_factor)
            primary_level = _finite_float(primary_factor)
            valid_factor_levels = (
                legacy_level is not None
                and primary_level is not None
                and math.isfinite(legacy_level)
                and math.isfinite(primary_level)
                and legacy_level > 0
                and primary_level > 0
            )
            scale = (
                primary_level / legacy_level
                if valid_factor_levels
                else None
            )
            abs_error = (
                abs(legacy_price - previous_price)
                if legacy_price is not None and previous_price is not None
                else None
            )
            relative_error = (
                abs_error / max(abs(legacy_price), abs(previous_price), 1e-9)
                if abs_error is not None
                and legacy_price is not None
                and previous_price is not None
                else None
            )
            item = {
                "ts_code": str(ts_code),
                "legacy_date": str(legacy_date),
                "primary_date": str(primary_date),
                "legacy_close": legacy_price,
                "primary_pre_close": previous_price,
                "legacy_adj_factor": legacy_level,
                "primary_adj_factor": primary_level,
                "legacy_scale": scale,
                "price_abs_error": abs_error,
                "price_relative_error": relative_error,
            }
            if scale is None or not math.isfinite(scale) or scale <= 0:
                masked.append(
                    {**item, "reason": "missing_or_invalid_boundary_factor"}
                )
            elif (
                previous_price is None
                or legacy_price is None
                or not math.isfinite(previous_price)
                or not math.isfinite(legacy_price)
                or previous_price <= 0
                or legacy_price <= 0
            ):
                masked.append({**item, "reason": "missing_or_invalid_boundary_price"})
            elif (
                abs_error is None
                or relative_error is None
                or not math.isfinite(abs_error)
                or not math.isfinite(relative_error)
                or abs_error > _ADJUSTMENT_BOUNDARY_MAX_PRICE_ABS_ERROR
                or relative_error
                > _ADJUSTMENT_BOUNDARY_MAX_PRICE_RELATIVE_ERROR
            ):
                masked.append({**item, "reason": "boundary_price_mismatch"})
            else:
                rebased.append(item)

        masked_ratio = len(masked) / len(rows) if rows else 0.0
        result = {
            "version": _ADJUSTMENT_BOUNDARY_POLICY_VERSION,
            "status": (
                "failed"
                if masked_ratio > _ADJUSTMENT_BOUNDARY_MAX_MASKED_RATIO
                else ("pass_with_masks" if masked else "pass")
            ),
            "cutoff_date": PRIMARY_MARKET_HISTORY_START.isoformat(),
            "policy": "rebase_legacy_factor_or_mask_cross_source_instrument",
            "max_price_abs_error": _ADJUSTMENT_BOUNDARY_MAX_PRICE_ABS_ERROR,
            "max_price_relative_error": (
                _ADJUSTMENT_BOUNDARY_MAX_PRICE_RELATIVE_ERROR
            ),
            "max_masked_ratio": _ADJUSTMENT_BOUNDARY_MAX_MASKED_RATIO,
            "cross_source_symbols": len(rows),
            "rebased_symbol_count": len(rebased),
            "masked_symbol_count": len(masked),
            "masked_ratio": masked_ratio,
            "rebased_symbols": rebased,
            "masked_symbols": masked,
        }
        result["evidence_sha256"] = _canonical_sha256(result)
        self._adjustment_boundary_cache = result
        return json.loads(json.dumps(result))

    def _require_adjustment_boundary_evidence(self) -> dict[str, Any]:
        evidence = self._adjustment_boundary_evidence()
        if evidence.get("status") == "failed":
            raise RuntimeError(
                "cross-source adjustment boundary rejected: "
                f"{evidence.get('masked_symbol_count', 0)}/"
                f"{evidence.get('cross_source_symbols', 0)} instruments "
                "failed continuity"
            )
        return evidence

    def _daily_unit_quality_coverage(self) -> dict[str, Any]:
        if self._daily_unit_quality_cache is not None:
            return dict(self._daily_unit_quality_cache)
        daily_files = list((self.snapshot_path / "parquet" / "daily").rglob("*.parquet"))
        if not daily_files:
            result = {
                "policy": "exclude_internally_inconsistent_rows_from_research_history",
                "excluded_rows": 0,
                "total_rows": 0,
                "excluded_ratio": 0.0,
                "max_excluded_ratio": _MAX_EXCLUDED_DAILY_UNIT_RATIO,
            }
            self._daily_unit_quality_cache = result
            return dict(result)
        daily = _sql_string(
            str((self.snapshot_path / "parquet" / "daily" / "**" / "*.parquet").resolve())
        )
        predicate = self._invalid_daily_units_predicate("")
        connection = self._duckdb_connection()
        try:
            row = connection.execute(
                f"""
                SELECT count(*) FILTER (WHERE {predicate}), count(*)
                FROM read_parquet({daily}, hive_partitioning=true, union_by_name=true)
                """
            ).fetchone()
        finally:
            connection.close()
        excluded_rows = int(row[0] or 0) if row is not None else 0
        total_rows = int(row[1] or 0) if row is not None else 0
        result = {
            "policy": "exclude_internally_inconsistent_rows_from_research_history",
            "excluded_rows": excluded_rows,
            "total_rows": total_rows,
            "excluded_ratio": excluded_rows / total_rows if total_rows else 0.0,
            "max_excluded_ratio": _MAX_EXCLUDED_DAILY_UNIT_RATIO,
        }
        self._daily_unit_quality_cache = result
        return dict(result)

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
        masked_symbols = [
            str(item["ts_code"])
            for item in self._adjustment_boundary_evidence().get("masked_symbols")
            or []
        ]
        mask_predicate = ""
        if masked_symbols:
            mask_predicate = (
                "AND ts_code NOT IN ("
                + ", ".join(_sql_string(symbol) for symbol in masked_symbols)
                + ")"
            )
        connection = self._duckdb_connection()
        try:
            rows = connection.execute(
                f"""
                SELECT ts_code, min(trade_date), max(trade_date)
                FROM read_parquet(
                    {_sql_string(str(daily_glob.resolve()))},
                    hive_partitioning=true,
                    union_by_name=true
                )
                WHERE ts_code IS NOT NULL AND trade_date IS NOT NULL
                  {mask_predicate}
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
                instruments = (
                    frame[instrument_column]
                    .astype("string")
                    .fillna("")
                    .str.replace(r"^\s+|\s+$", "", regex=True)
                )
                industries = (
                    frame[industry_column]
                    .astype("string")
                    .str.replace(r"^\s+|\s+$", "", regex=True)
                )
                metadata = pd.DataFrame(
                    {
                        "instrument": instruments.map(_qlib_symbol),
                        "industry": industries.mask(industries.eq("")),
                        "in_date": pd.to_datetime(frame.get("in_date"), errors="coerce"),
                        "out_date": pd.to_datetime(frame.get("out_date"), errors="coerce"),
                    }
                ).dropna(subset=["instrument", "industry", "in_date"])
                metadata.drop_duplicates(
                    ["instrument", "industry", "in_date", "out_date"], inplace=True
                )
                unknown = self._unknown_benchmark_industry_memberships(metadata)
                if not unknown.empty:
                    metadata = pd.concat([metadata, unknown], ignore_index=True)
                    logger.warning(
                        "assigned %s source-gap intervals to explicit industry %s",
                        len(unknown),
                        _UNKNOWN_INDUSTRY,
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

        if self._write_style_metadata_bounded(target):
            wrote_metadata = True

        if self._write_market_context_metadata(target):
            wrote_metadata = True

        if not self._write_eligibility_metadata(target):
            raise RuntimeError("Qlib point-in-time eligibility metadata is incomplete")
        wrote_metadata = True

        if not wrote_metadata and target.exists() and not any(target.iterdir()):
            target.rmdir()

    def _unknown_benchmark_industry_memberships(
        self, metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """Represent small provider gaps without future-filling classifications.

        Tushare occasionally leaves an interval between two historical Shenwan
        classifications.  Forward/back-filling an adjacent industry would
        introduce look-ahead or stale-classification bias.  For constituents
        affected on governed-benchmark weight dates, emit an explicit unknown
        interval bounded by the surrounding source intervals instead.
        """

        weight_source = self.snapshot_path / "parquet" / "index_weight"
        weight_files = (
            sorted(weight_source.rglob("*.parquet"))
            if weight_source.exists()
            else []
        )
        if not weight_files or metadata.empty:
            return pd.DataFrame(columns=metadata.columns)
        weights = pd.concat(
            [pd.read_parquet(path) for path in weight_files], ignore_index=True
        )
        required = {"index_code", "con_code", "trade_date", "weight"}
        if not required.issubset(weights.columns):
            return pd.DataFrame(columns=metadata.columns)
        benchmarks = (
            weights["index_code"]
            .astype("string")
            .str.replace(r"^\s+|\s+$", "", regex=True)
            .str.upper()
        )
        instruments = (
            weights["con_code"]
            .astype("string")
            .fillna("")
            .str.replace(r"^\s+|\s+$", "", regex=True)
        )
        weights = pd.DataFrame(
            {
                "benchmark": benchmarks,
                "instrument": instruments.map(_qlib_symbol),
                "datetime": pd.to_datetime(weights["trade_date"], errors="coerce"),
                "weight": pd.to_numeric(weights["weight"], errors="coerce"),
            }
        ).dropna()
        weights = weights[
            (weights["benchmark"] == _GOVERNED_BENCHMARK)
            & (weights["weight"] > 0)
        ]
        weights.drop_duplicates(["instrument", "datetime"], keep="last", inplace=True)
        if weights.empty:
            return pd.DataFrame(columns=metadata.columns)

        normalized = metadata.copy()
        normalized["in_date"] = pd.to_datetime(normalized["in_date"], errors="coerce")
        normalized["out_date"] = pd.to_datetime(normalized["out_date"], errors="coerce")
        rows: list[dict[str, Any]] = []
        for instrument, observations in weights.groupby("instrument", sort=True):
            known = normalized[normalized["instrument"] == instrument].sort_values(
                "in_date"
            )
            dates = sorted(pd.Timestamp(value).normalize() for value in observations["datetime"])
            missing = [
                value
                for value in dates
                if known[
                    (known["in_date"] <= value)
                    & (known["out_date"].isna() | (known["out_date"] >= value))
                ].empty
            ]
            if not missing:
                continue
            first_missing = min(missing)
            intervals: set[tuple[pd.Timestamp, pd.Timestamp | None]] = set()
            for value in missing:
                previous = known.loc[known["out_date"].notna() & (known["out_date"] < value)]
                following = known.loc[known["in_date"] > value]
                start = (
                    pd.Timestamp(previous["out_date"].max()).normalize()
                    + pd.Timedelta(days=1)
                    if not previous.empty
                    else first_missing
                )
                end = (
                    pd.Timestamp(following["in_date"].min()).normalize()
                    - pd.Timedelta(days=1)
                    if not following.empty
                    else None
                )
                intervals.add((start, end))
            rows.extend(
                {
                    "instrument": instrument,
                    "industry": _UNKNOWN_INDUSTRY,
                    "in_date": start,
                    "out_date": end,
                }
                for start, end in sorted(
                    intervals,
                    key=lambda item: (item[0], item[1] or pd.Timestamp.max),
                )
            )
        result = pd.DataFrame(rows, columns=metadata.columns)
        if not result.empty:
            result["in_date"] = pd.to_datetime(result["in_date"])
            result["out_date"] = pd.to_datetime(result["out_date"])
        return result

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

    def _read_dataset_for_symbols(
        self,
        dataset: str,
        columns: Collection[str],
        symbols: Collection[str],
        *,
        required: Collection[str] = (),
    ) -> pd.DataFrame | None:
        """Read a bounded full-history symbol batch from one snapshot dataset."""

        selected_symbols = sorted({str(item) for item in symbols if str(item)})
        available = self._parquet_columns(dataset)
        selected_columns = sorted(set(columns).intersection(available))
        if not selected_symbols or not set(required).issubset(selected_columns):
            return None
        root = self.snapshot_path / "parquet" / dataset
        glob = _sql_string(str((root / "**" / "*.parquet").resolve()))
        projection = ", ".join(_sql_identifier(column) for column in selected_columns)
        symbol_values = ", ".join(_sql_string(symbol) for symbol in selected_symbols)
        connection = self._duckdb_connection()
        try:
            return connection.execute(
                f"SELECT {projection} FROM read_parquet({glob}, "
                "hive_partitioning=true, union_by_name=true) "
                f"WHERE ts_code IN ({symbol_values})"
            ).fetch_df()
        finally:
            connection.close()

    def _read_dataset_columns(
        self,
        dataset: str,
        columns: Collection[str],
        *,
        required: Collection[str] = (),
    ) -> pd.DataFrame | None:
        """Read only a projected snapshot surface into pandas.

        This is reserved for bounded reference/event tables. Full-history
        market panels must use ``_read_dataset_for_symbols`` instead.
        """

        available = self._parquet_columns(dataset)
        selected_columns = sorted(set(columns).intersection(available))
        if not set(required).issubset(selected_columns):
            return None
        root = self.snapshot_path / "parquet" / dataset
        if not root.exists() or not any(root.rglob("*.parquet")):
            return None
        glob = _sql_string(str((root / "**" / "*.parquet").resolve()))
        projection = ", ".join(_sql_identifier(column) for column in selected_columns)
        connection = self._duckdb_connection()
        try:
            return connection.execute(
                f"SELECT {projection} FROM read_parquet({glob}, "
                "hive_partitioning=true, union_by_name=true)"
            ).fetch_df()
        finally:
            connection.close()

    def _style_symbols(self) -> list[str]:
        root = self.snapshot_path / "parquet" / "daily_basic"
        if not root.exists() or not any(root.rglob("*.parquet")):
            return []
        glob = _sql_string(str((root / "**" / "*.parquet").resolve()))
        connection = self._duckdb_connection()
        try:
            return [
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT trim(CAST(ts_code AS VARCHAR)) AS ts_code "
                    f"FROM read_parquet({glob}, hive_partitioning=true, "
                    "union_by_name=true) WHERE ts_code IS NOT NULL "
                    "ORDER BY ts_code"
                ).fetchall()
                if str(row[0] or "")
            ]
        finally:
            connection.close()

    def _write_style_metadata_bounded(self, target: Path) -> bool:
        """Build full-history descriptors by symbol and standardize by year.

        Rolling descriptors must see each instrument's complete history, while
        standardization must see the complete cross-section for each date.  A
        symbol-batch raw pass followed by a year-batch cross-sectional pass
        preserves both mathematical boundaries without retaining the entire
        market panel in one pandas object.
        """

        required = {"ts_code", "trade_date", "total_mv"}
        if not required.issubset(self._parquet_columns("daily_basic")):
            return False
        symbols = self._style_symbols()
        if not symbols:
            return False
        target.mkdir(parents=True, exist_ok=True)
        work = target / ".style_metadata_attempt"
        if work.exists():
            shutil.rmtree(work)
        raw_dir = work / "raw"
        spill_dir = work / "duckdb_spill"
        raw_dir.mkdir(parents=True, exist_ok=True)
        style_tmp = target / ".style_exposures.parquet.tmp"
        weights_tmp = target / ".full_market_weights.parquet.tmp"
        style_tmp.unlink(missing_ok=True)
        weights_tmp.unlink(missing_ok=True)
        writer: pq.ParquetWriter | None = None
        try:
            daily_basic_columns = {
                "ts_code",
                "trade_date",
                "total_mv",
                "circ_mv",
                "pb",
                "pe_ttm",
                "turnover_rate",
            }
            for offset in range(0, len(symbols), DAILY_QLIB_STYLE_SYMBOL_BATCH):
                batch = symbols[offset : offset + DAILY_QLIB_STYLE_SYMBOL_BATCH]
                daily_basic = self._read_dataset_for_symbols(
                    "daily_basic",
                    daily_basic_columns,
                    batch,
                    required=required,
                )
                if daily_basic is None or daily_basic.empty:
                    continue
                adjusted_close = self._load_adjusted_close(batch)
                fina_indicator = self._load_fina_indicator(batch)
                raw = build_raw_style_panel(
                    daily_basic,
                    adjusted_close=adjusted_close,
                    fina_indicator=fina_indicator,
                )
                raw = raw.rename(
                    columns={"ts_code": "instrument", "trade_date": "datetime"}
                )
                raw["instrument"] = raw["instrument"].map(_qlib_symbol)
                if not raw.empty:
                    raw.to_parquet(
                        raw_dir / f"batch-{offset // DAILY_QLIB_STYLE_SYMBOL_BATCH:05d}.parquet",
                        index=False,
                        compression="zstd",
                    )
            raw_files = sorted(raw_dir.glob("*.parquet"))
            if not raw_files:
                return False

            raw_glob = _sql_string(str((raw_dir / "*.parquet").resolve()))
            connection = self._duckdb_connection(spill_dir=spill_dir)
            try:
                years = [
                    int(row[0])
                    for row in connection.execute(
                        "SELECT DISTINCT year(try_cast(datetime AS TIMESTAMP)) AS y "
                        f"FROM read_parquet({raw_glob}, union_by_name=true) "
                        "WHERE try_cast(datetime AS TIMESTAMP) IS NOT NULL ORDER BY y"
                    ).fetchall()
                ]
                for year in years:
                    raw_year = connection.execute(
                        f"SELECT * FROM read_parquet({raw_glob}, union_by_name=true) "
                        f"WHERE year(try_cast(datetime AS TIMESTAMP)) = {year} "
                        "ORDER BY datetime, instrument"
                    ).fetch_df()
                    standardized = standardize_panel(raw_year)
                    if standardized.empty:
                        continue
                    ordered_columns = [
                        "instrument",
                        "datetime",
                        "log_market_cap",
                        *STYLE_COLUMNS,
                    ]
                    standardized = standardized.loc[:, ordered_columns]
                    standardized["instrument"] = standardized["instrument"].astype(str)
                    standardized["datetime"] = pd.to_datetime(
                        standardized["datetime"], errors="raise"
                    )
                    for column in ordered_columns[2:]:
                        standardized[column] = pd.to_numeric(
                            standardized[column], errors="coerce"
                        ).astype("float64")
                    table = pa.Table.from_pandas(standardized, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(style_tmp, table.schema, compression="zstd")
                    writer.write_table(table)

                connection.execute(
                    f"COPY ("
                    "WITH clean AS ("
                    "SELECT instrument, try_cast(datetime AS TIMESTAMP) AS datetime, "
                    "try_cast(float_market_cap AS DOUBLE) AS float_market_cap "
                    f"FROM read_parquet({raw_glob}, union_by_name=true) "
                    "WHERE try_cast(datetime AS TIMESTAMP) IS NOT NULL "
                    "AND try_cast(float_market_cap AS DOUBLE) > 0"
                    "), weighted AS ("
                    "SELECT instrument, datetime, float_market_cap / "
                    "sum(float_market_cap) OVER (PARTITION BY datetime) AS weight "
                    "FROM clean"
                    ") SELECT instrument, datetime, weight FROM weighted "
                    "ORDER BY datetime, instrument"
                    f") TO {_sql_string(str(weights_tmp.resolve()))} "
                    "(FORMAT PARQUET, COMPRESSION ZSTD)"
                )
            finally:
                connection.close()
            if writer is None:
                return False
            writer.close()
            writer = None
            os.replace(style_tmp, target / "style_exposures.parquet")
            os.replace(weights_tmp, target / "full_market_weights.parquet")
            return True
        finally:
            if writer is not None:
                writer.close()
            style_tmp.unlink(missing_ok=True)
            weights_tmp.unlink(missing_ok=True)
            shutil.rmtree(work, ignore_errors=True)

    def _load_adjusted_close(
        self, symbols: Collection[str] | None = None
    ) -> pd.DataFrame | None:
        daily_root = self.snapshot_path / "parquet" / "daily"
        daily_files = sorted(daily_root.rglob("*.parquet")) if daily_root.exists() else []
        daily = (
            self._read_dataset_for_symbols(
                "daily",
                {"ts_code", "trade_date", "close"},
                symbols,
                required={"ts_code", "trade_date", "close"},
            )
            if symbols is not None
            else _read_parquet_columns(
                daily_files, {"ts_code", "trade_date", "close"}
            )
        )
        if daily is None:
            return None
        adj_root = self.snapshot_path / "parquet" / "adj_factor"
        adj_files = sorted(adj_root.rglob("*.parquet")) if adj_root.exists() else []
        factors = (
            self._read_dataset_for_symbols(
                "adj_factor",
                {"ts_code", "trade_date", "adj_factor"},
                symbols,
                required={"ts_code", "trade_date", "adj_factor"},
            )
            if symbols is not None
            else _read_parquet_columns(
                adj_files, {"ts_code", "trade_date", "adj_factor"}
            )
        )
        evidence = self._require_adjustment_boundary_evidence()
        masked_symbols = {
            str(item["ts_code"])
            for item in evidence.get("masked_symbols") or []
        }
        if masked_symbols:
            daily = daily.loc[~daily["ts_code"].astype(str).isin(masked_symbols)].copy()
            if factors is not None:
                factors = factors.loc[
                    ~factors["ts_code"].astype(str).isin(masked_symbols)
                ].copy()
        if factors is not None and not factors.empty:
            scale_by_symbol = {
                str(item["ts_code"]): float(item["legacy_scale"])
                for item in evidence.get("rebased_symbols") or []
            }
            if scale_by_symbol:
                factor_dates = pd.to_datetime(
                    factors["trade_date"].astype(str),
                    format="mixed",
                    errors="coerce",
                )
                legacy_rows = factor_dates.lt(pd.Timestamp(PRIMARY_MARKET_HISTORY_START))
                scales = factors["ts_code"].astype(str).map(scale_by_symbol)
                rebased_rows = legacy_rows & scales.notna()
                factors.loc[rebased_rows, "adj_factor"] = (
                    pd.to_numeric(
                        factors.loc[rebased_rows, "adj_factor"], errors="coerce"
                    )
                    * scales.loc[rebased_rows]
                )
        return build_adjusted_close(daily, factors)

    def _load_fina_indicator(
        self, symbols: Collection[str] | None = None
    ) -> pd.DataFrame | None:
        root = self.snapshot_path / "parquet" / "fina_indicator"
        files = sorted(root.rglob("*.parquet")) if root.exists() else []
        if symbols is not None:
            return self._read_dataset_for_symbols(
                "fina_indicator",
                {"ts_code", "ann_date", "roe", "or_yoy", "netprofit_yoy", "debt_to_assets"},
                symbols,
                required={"ts_code", "ann_date"},
            )
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
        """Write PIT eligibility without materializing the full daily panel."""

        required_daily = {"ts_code", "trade_date", "amount", "vol"}
        if not required_daily.issubset(self._parquet_columns("daily")):
            return False
        daily_glob = _sql_string(
            str((self.snapshot_path / "parquet" / "daily" / "**" / "*.parquet").resolve())
        )
        connection = self._duckdb_connection()
        try:
            calendar_rows = connection.execute(
                f"SELECT DISTINCT {_as_date_sql('trade_date')} AS trade_date "
                f"FROM read_parquet({daily_glob}, hive_partitioning=true, "
                "union_by_name=true) WHERE ts_code IS NOT NULL "
                f"AND {_as_date_sql('trade_date')} IS NOT NULL ORDER BY trade_date"
            ).fetchall()
            symbols = [
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT trim(CAST(ts_code AS VARCHAR)) AS ts_code "
                    f"FROM read_parquet({daily_glob}, hive_partitioning=true, "
                    "union_by_name=true) WHERE ts_code IS NOT NULL ORDER BY ts_code"
                ).fetchall()
                if str(row[0] or "")
            ]
        finally:
            connection.close()
        trading_calendar = pd.DatetimeIndex([row[0] for row in calendar_rows])
        if trading_calendar.empty or not symbols:
            raise ValueError("daily has no valid publication-horizon market rows")
        normalized_symbols = [_qlib_symbol(symbol) for symbol in symbols]
        if any(symbol is None for symbol in normalized_symbols):
            raise ValueError("daily contains a symbol that cannot be normalized for Qlib")
        if len(set(normalized_symbols)) != len(normalized_symbols):
            raise ValueError("daily symbols collide after Qlib normalization")
        regulatory_horizon = trading_calendar.max().date()

        stock_basic = self._read_dataset_columns(
            "stock_basic",
            {"ts_code", "list_date", "delist_date"},
            required={"ts_code", "list_date"},
        )
        if stock_basic is None or stock_basic.empty:
            return False
        listings = pd.DataFrame(
            {
                "instrument": stock_basic["ts_code"].map(_qlib_symbol),
                "list_date": pd.to_datetime(stock_basic["list_date"], errors="coerce"),
                "delist_date": pd.to_datetime(
                    stock_basic.get(
                        "delist_date", pd.Series(pd.NaT, index=stock_basic.index)
                    ),
                    errors="coerce",
                ),
            }
        )
        namechange = self._read_dataset_columns(
            "namechange",
            {"ts_code", "name", "start_date", "end_date"},
            required={"ts_code", "name", "start_date"},
        )
        if namechange is not None and not namechange.empty:
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

        balance_columns = self._parquet_columns("balancesheet")
        equity_column = next(
            (
                name
                for name in (
                    "total_hldr_eqy_exc_min_int",
                    "total_hldr_eqy_inc_min_int",
                    "total_hldr_eqy",
                )
                if name in balance_columns
            ),
            None,
        )
        if equity_column is None or "ann_date" not in balance_columns:
            raise ValueError("balancesheet has no announced shareholder equity")
        audit_columns = self._parquet_columns("fina_audit")
        opinion_column = next(
            (
                name
                for name in ("audit_result", "audit_opinion")
                if name in audit_columns
            ),
            None,
        )
        if opinion_column is None or "ann_date" not in audit_columns:
            raise ValueError("fina_audit has no announced audit opinion")

        regulatory_columns = self._parquet_columns("regulatory_events")
        has_regulatory_source = bool(regulatory_columns) and self._has_usable_row(
            "regulatory_events", "TRUE"
        )
        if has_regulatory_source and not {
            "ts_code",
            "event_date",
            "known_date",
            "major",
        }.issubset(regulatory_columns):
            raise ValueError("regulatory event source violates its data contract")
        anns_root = self.snapshot_path / "parquet" / "anns_d"
        has_anns_source = anns_root.is_dir() and any(anns_root.rglob("*.parquet"))
        anns_columns = self._parquet_columns("anns_d")
        has_anns_fallback = not has_regulatory_source and has_anns_source
        open_days: list[date] = []
        if has_anns_fallback:
            if not {"ts_code", "ann_date", "title"}.issubset(anns_columns):
                raise ValueError(
                    "anns_d source violates its regulatory fallback contract"
                )
            trade_cal = self._read_dataset_columns(
                "trade_cal",
                {"cal_date", "is_open"},
                required={"cal_date", "is_open"},
            )
            if trade_cal is None or trade_cal.empty:
                raise ValueError(
                    "anns_d regulatory fallback requires a valid trading calendar"
                )
            open_days = open_days_from_trade_cal(trade_cal)
        regulatory_origin = (
            "materialized_dataset"
            if has_regulatory_source
            else (
                f"anns_d_title_rules({REGULATORY_EVENTS_RULE_VERSION})"
                if has_anns_fallback
                else None
            )
        )
        target.mkdir(parents=True, exist_ok=True)
        work = target / ".eligibility_attempt"
        if work.exists():
            shutil.rmtree(work)
        batches_dir = work / "batches"
        spill_dir = work / "duckdb_spill"
        batches_dir.mkdir(parents=True)
        final_tmp = target / ".eligibility_matrix.parquet.tmp"
        final_tmp.unlink(missing_ok=True)
        deferred_regulatory_events: list[dict[str, str]] = []
        try:
            for offset in range(0, len(symbols), DAILY_QLIB_ELIGIBILITY_SYMBOL_BATCH):
                batch = symbols[offset : offset + DAILY_QLIB_ELIGIBILITY_SYMBOL_BATCH]
                matrix, deferred = self._eligibility_symbol_batch(
                    batch=batch,
                    required_daily=required_daily,
                    listings=listings,
                    st_intervals=st_intervals,
                    equity_column=equity_column,
                    opinion_column=opinion_column,
                    has_regulatory_source=has_regulatory_source,
                    has_anns_fallback=has_anns_fallback,
                    open_days=open_days,
                    regulatory_horizon=regulatory_horizon,
                    trading_calendar=trading_calendar,
                )
                deferred_regulatory_events.extend(deferred)
                matrix.to_parquet(
                    batches_dir
                    / f"batch-{offset // DAILY_QLIB_ELIGIBILITY_SYMBOL_BATCH:05d}.parquet",
                    index=False,
                    compression="zstd",
                )
            batch_glob = _sql_string(str((batches_dir / "*.parquet").resolve()))
            connection = self._duckdb_connection(spill_dir=spill_dir)
            try:
                duplicate = connection.execute(
                    "SELECT datetime, instrument, COUNT(*) AS row_count "
                    f"FROM read_parquet({batch_glob}, union_by_name=true) "
                    "GROUP BY datetime, instrument HAVING COUNT(*) > 1 LIMIT 1"
                ).fetchone()
                if duplicate is not None:
                    raise ValueError(
                        "eligibility batches contain duplicate datetime/instrument keys"
                    )
                connection.execute(
                    "COPY (SELECT * FROM read_parquet("
                    f"{batch_glob}, union_by_name=true) ORDER BY datetime, instrument) "
                    f"TO {_sql_string(str(final_tmp.resolve()))} "
                    "(FORMAT PARQUET, COMPRESSION ZSTD)"
                )
            finally:
                connection.close()
            os.replace(final_tmp, target / "eligibility_matrix.parquet")
        finally:
            final_tmp.unlink(missing_ok=True)
            shutil.rmtree(work, ignore_errors=True)

        regulatory_terminal_audit = {
            "publication_horizon": regulatory_horizon.isoformat(),
            "policy": REGULATORY_TERMINAL_DEFERRAL_POLICY,
            "deferred_event_count": len(deferred_regulatory_events),
            "deferred_events_sha256": _canonical_sha256(deferred_regulatory_events),
        }
        (target / "eligibility_contract.json").write_text(
            json.dumps(
                {
                    "version": ELIGIBILITY_CONTRACT_VERSION,
                    "regulatory_data_available": (
                        has_regulatory_source or has_anns_fallback
                    ),
                    "regulatory_origin": regulatory_origin,
                    "regulatory_publication_horizon": regulatory_horizon.isoformat(),
                    "regulatory_terminal_policy": REGULATORY_TERMINAL_DEFERRAL_POLICY,
                    "regulatory_deferred_event_count": len(deferred_regulatory_events),
                    "regulatory_deferred_events_sha256": regulatory_terminal_audit[
                        "deferred_events_sha256"
                    ],
                    "regulatory_terminal_audit_sha256": _canonical_sha256(
                        regulatory_terminal_audit
                    ),
                    "financial_availability": "strictly_after_announcement_date",
                    "delisting_availability": "effective_date_only_no_backfill",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return True

    def _eligibility_symbol_batch(
        self,
        *,
        batch: list[str],
        required_daily: set[str],
        listings: pd.DataFrame,
        st_intervals: pd.DataFrame,
        equity_column: str,
        opinion_column: str,
        has_regulatory_source: bool,
        has_anns_fallback: bool,
        open_days: list[date],
        regulatory_horizon: date,
        trading_calendar: pd.DatetimeIndex,
    ) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        daily = self._read_dataset_for_symbols(
            "daily", required_daily, batch, required=required_daily
        )
        balancesheet = self._read_dataset_for_symbols(
            "balancesheet",
            {"ts_code", "ann_date", equity_column},
            batch,
            required={"ts_code", "ann_date", equity_column},
        )
        audit = self._read_dataset_for_symbols(
            "fina_audit",
            {"ts_code", "ann_date", opinion_column},
            batch,
            required={"ts_code", "ann_date", opinion_column},
        )
        if daily is None or daily.empty:
            raise ValueError("eligibility symbol batch lacks daily market evidence")
        if balancesheet is None or audit is None:
            raise ValueError("eligibility financial source schema changed during publication")
        market = pd.DataFrame(
            {
                "datetime": pd.to_datetime(daily["trade_date"], errors="coerce"),
                "instrument": daily["ts_code"].map(_qlib_symbol),
                "amount": pd.to_numeric(daily["amount"], errors="coerce") * 1000.0,
                "paused": pd.to_numeric(daily["vol"], errors="coerce").fillna(0).le(0),
            }
        )
        instruments = set(market["instrument"].dropna().astype(str))
        suspension_columns = self._parquet_columns("suspend_d")
        suspension_date = next(
            (
                name
                for name in ("suspend_date", "trade_date")
                if name in suspension_columns
            ),
            None,
        )
        suspend = (
            self._read_dataset_for_symbols(
                "suspend_d",
                {"ts_code", suspension_date},
                batch,
                required={"ts_code", suspension_date},
            )
            if suspension_date is not None
            else None
        )
        suspensions = pd.DataFrame(
            {
                "instrument": (
                    suspend["ts_code"].map(_qlib_symbol)
                    if suspend is not None
                    else pd.Series(dtype="object")
                ),
                "datetime": (
                    suspend[suspension_date]
                    if suspend is not None
                    else pd.Series(dtype="datetime64[ns]")
                ),
                "suspended": True,
            }
        )
        financials = pd.DataFrame(
            {
                "instrument": balancesheet["ts_code"].map(_qlib_symbol),
                "announcement_date": balancesheet["ann_date"],
                "equity": pd.to_numeric(balancesheet[equity_column], errors="coerce"),
            }
        )
        audits = pd.DataFrame(
            {
                "instrument": audit["ts_code"].map(_qlib_symbol),
                "announcement_date": audit["ann_date"],
                "audit_opinion": audit[opinion_column].astype(str),
            }
        )
        regulatory: pd.DataFrame | None = None
        deferred: list[dict[str, str]] = []
        if has_regulatory_source:
            source = self._read_dataset_for_symbols(
                "regulatory_events",
                {"ts_code", "event_date", "known_date", "major"},
                batch,
                required={"ts_code", "event_date", "known_date", "major"},
            )
            regulatory = (
                source.rename(columns={"ts_code": "instrument"}).copy()
                if source is not None
                else pd.DataFrame(
                    columns=["instrument", "event_date", "known_date", "major"]
                )
            )
            regulatory["instrument"] = regulatory["instrument"].map(_qlib_symbol)
        elif has_anns_fallback:
            anns = self._read_dataset_for_symbols(
                "anns_d",
                {"ts_code", "ann_date", "title", "url"},
                batch,
                required={"ts_code", "ann_date", "title"},
            )
            if anns is None:
                raise ValueError(
                    "anns_d source schema changed during eligibility publication"
                )
            events, deferred = derive_regulatory_events_for_horizon(
                anns, open_days, publication_horizon=regulatory_horizon
            )
            regulatory = events.rename(columns={"ts_code": "instrument"}).copy()
            regulatory["instrument"] = regulatory["instrument"].map(_qlib_symbol)
        return (
            build_point_in_time_eligibility(
                market=market,
                listings=listings[listings["instrument"].isin(instruments)].copy(),
                st_intervals=st_intervals[
                    st_intervals["instrument"].isin(instruments)
                ].copy(),
                suspensions=suspensions,
                financials=financials,
                audits=audits,
                regulatory_events=regulatory,
                policy=EligibilityPolicy(),
                trading_calendar=trading_calendar,
            ),
            deferred,
        )

    def _derive_regulatory_events(
        self,
        read: Callable[[str], pd.DataFrame],
        *,
        publication_horizon: date,
    ) -> tuple[pd.DataFrame | None, list[dict[str, str]]]:
        """Derive major-violation events from snapshot anns_d titles.

        Returns None (current fail-soft behavior) when the snapshot carries no
        anns_d parquet. Otherwise the publication-boundary wrapper excludes
        only recognizable major events announced on the final market day: the
        next trading day is outside this immutable Qlib artifact, so those rows
        are deferred with deterministic audit evidence. Historical calendar
        gaps and source rows beyond the horizon still fail closed.
        """

        anns_root = self.snapshot_path / "parquet" / "anns_d"
        if not anns_root.is_dir() or not any(anns_root.rglob("*.parquet")):
            return None, []
        open_days = open_days_from_trade_cal(read("trade_cal"))
        events, deferred = derive_regulatory_events_for_horizon(
            read("anns_d"),
            open_days,
            publication_horizon=publication_horizon,
        )
        regulatory = events.rename(columns={"ts_code": "instrument"}).copy()
        regulatory["instrument"] = regulatory["instrument"].map(_qlib_symbol)
        return regulatory, deferred

    def _normalized_query(self, daily_glob: Path, adj_glob: Path, limit_glob: Path) -> str:
        daily = _sql_string(str(daily_glob.resolve()))
        adj = _sql_string(str(adj_glob.resolve()))
        limits = _sql_string(str(limit_glob.resolve()))
        adjustment_boundary = self._require_adjustment_boundary_evidence()
        scale_rows = [
            (
                str(item["ts_code"]),
                float(item["legacy_scale"]),
            )
            for item in adjustment_boundary.get("rebased_symbols") or []
        ]
        masked_symbols = [
            str(item["ts_code"])
            for item in adjustment_boundary.get("masked_symbols") or []
        ]
        if scale_rows:
            scale_values = ", ".join(
                f"({_sql_string(symbol)}, {format(scale, '.17g')})"
                for symbol, scale in scale_rows
            )
            scale_relation = (
                "boundary_scales(ts_code, legacy_scale) AS "
                f"(VALUES {scale_values})"
            )
        else:
            scale_relation = (
                "boundary_scales AS (SELECT NULL::VARCHAR AS ts_code, "
                "NULL::DOUBLE AS legacy_scale WHERE FALSE)"
            )
        if masked_symbols:
            mask_values = ", ".join(
                f"({_sql_string(symbol)})" for symbol in masked_symbols
            )
            mask_relation = (
                "boundary_masks(ts_code) AS "
                f"(VALUES {mask_values})"
            )
        else:
            mask_relation = (
                "boundary_masks AS (SELECT NULL::VARCHAR AS ts_code WHERE FALSE)"
            )
        cutoff = _sql_string(PRIMARY_MARKET_HISTORY_START.isoformat())
        source_adjustment_factor = (
            "CASE WHEN coalesce(try_cast(d.trade_date AS DATE), "
            "try_strptime(CAST(d.trade_date AS VARCHAR), '%Y%m%d')::DATE) "
            f"< DATE {cutoff} AND boundary_scales.legacy_scale IS NOT NULL "
            "THEN a.adj_factor * boundary_scales.legacy_scale "
            "ELSE a.adj_factor END"
        )
        daily_features = self.research_feature_contract["daily_fields"]
        fundamental_features = self.research_feature_contract["fundamental_fields"]
        capital_flow_features = self.research_feature_contract["capital_flow_fields"]
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
                LEFT JOIN read_parquet(
                    {daily_basic}, hive_partitioning=true, union_by_name=true
                ) db
                  ON d.ts_code = db.ts_code
                 AND try_cast(d.trade_date AS DATE) = try_cast(db.trade_date AS DATE)
            """

        capital_flow_select = ""
        capital_flow_join = ""
        if capital_flow_features:
            moneyflow = _sql_string(
                str(
                    (
                        self.snapshot_path
                        / "parquet"
                        / "moneyflow"
                        / "**"
                        / "*.parquet"
                    ).resolve()
                )
            )
            large_buy = (
                "try_cast(mf.buy_lg_amount AS DOUBLE) + "
                "try_cast(mf.buy_elg_amount AS DOUBLE)"
            )
            large_sell = (
                "try_cast(mf.sell_lg_amount AS DOUBLE) + "
                "try_cast(mf.sell_elg_amount AS DOUBLE)"
            )
            capital_flow_expressions = {
                # Tushare moneyflow amounts are ten-thousand CNY.
                "mf_net_inflow_amount": (
                    "try_cast(mf.net_mf_amount AS DOUBLE) * 10000.0"
                ),
                # Daily amount is thousand CNY, hence net_mf_amount * 10 / amount.
                "mf_net_inflow_ratio": (
                    "CASE WHEN try_cast(d.amount AS DOUBLE) > 0 "
                    "THEN try_cast(mf.net_mf_amount AS DOUBLE) * 10.0 "
                    "/ try_cast(d.amount AS DOUBLE) END"
                ),
                "mf_large_order_imbalance": (
                    f"CASE WHEN ({large_buy}) + ({large_sell}) > 0 "
                    f"THEN (({large_buy}) - ({large_sell})) "
                    f"/ (({large_buy}) + ({large_sell})) END"
                ),
            }
            capital_flow_select = "".join(
                f"\n                    , {capital_flow_expressions[field]} AS {field}"
                for field in capital_flow_features
            )
            capital_flow_join = f"""
                LEFT JOIN read_parquet(
                    {moneyflow}, hive_partitioning=true, union_by_name=true
                ) mf
                  ON d.ts_code = mf.ts_code
                 AND try_cast(d.trade_date AS DATE) = try_cast(mf.trade_date AS DATE)
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
                    FROM read_parquet(
                        {statement}, hive_partitioning=true, union_by_name=true
                    )
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
            WITH {scale_relation},
            {mask_relation},
            joined AS (
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
                    {source_adjustment_factor} AS adj_factor,
                    coalesce(l.up_limit, {_UNRESTRICTED_UP_LIMIT}) AS up_limit,
                    coalesce(l.down_limit, 0.0) AS down_limit
                    {joined_daily_select}
                    {joined_fundamental_select}
                    {capital_flow_select}
                    , first_value(d.close * ({source_adjustment_factor})) OVER (
                        PARTITION BY d.ts_code ORDER BY d.trade_date
                    ) AS base_price
                FROM read_parquet({daily}, hive_partitioning=true, union_by_name=true) d
                LEFT JOIN read_parquet({adj}, hive_partitioning=true, union_by_name=true) a
                  ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
                LEFT JOIN boundary_scales
                  ON d.ts_code = boundary_scales.ts_code
                LEFT JOIN boundary_masks
                  ON d.ts_code = boundary_masks.ts_code
                LEFT JOIN read_parquet({limits}, hive_partitioning=true, union_by_name=true) l
                  ON d.ts_code = l.ts_code AND d.trade_date = l.trade_date
                {daily_join}
                {fundamental_join}
                {capital_flow_join}
                WHERE d.ts_code IS NOT NULL AND d.close IS NOT NULL
                  AND boundary_masks.ts_code IS NULL
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
                {''.join(f', {field}' for field in capital_flow_features)}
            FROM joined
            WHERE adj_factor IS NOT NULL AND adj_factor > 0 AND base_price > 0
              AND NOT ({self._invalid_daily_units_predicate("")})
        """

    @staticmethod
    def _invalid_daily_units_predicate(alias: str) -> str:
        """Identify rows whose amount/hand units imply an impossible traded price."""

        prefix = f"{alias}." if alias else ""
        vol = f"try_cast({prefix}vol AS DOUBLE)"
        amount = f"try_cast({prefix}amount AS DOUBLE)"
        low = f"try_cast({prefix}low AS DOUBLE)"
        high = f"try_cast({prefix}high AS DOUBLE)"
        return f"""
            {vol} > 0 AND (
                {amount} IS NULL
                OR {amount} <= 0
                OR {low} <= 0
                OR {high} < {low}
                OR {amount} * 10.0 / {vol} < {low} * 0.95
                OR {amount} * 10.0 / {vol} > {high} * 1.05
            )
        """

    def _research_feature_contract(self) -> dict[str, object]:
        daily_columns = self._parquet_columns("daily_basic")
        capital_flow_columns = self._parquet_columns("moneyflow")
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
        capital_flow_fields = [
            target
            for target, required in _CAPITAL_FLOW_FEATURE_REQUIREMENTS.items()
            if required.issubset(capital_flow_columns)
        ]
        missing_capital_flow_fields = {
            target: sorted(required - capital_flow_columns)
            for target, required in _CAPITAL_FLOW_FEATURE_REQUIREMENTS.items()
            if not required.issubset(capital_flow_columns)
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
        admitted_capital_sources = set().union(
            *(
                _CAPITAL_FLOW_FEATURE_REQUIREMENTS[target]
                for target in capital_flow_fields
            )
        ) if capital_flow_fields else set()
        all_null_capital_sources = self._all_null_columns(
            "moneyflow", admitted_capital_sources
        )
        all_null_capital_flow_fields = {
            target: sorted(
                _CAPITAL_FLOW_FEATURE_REQUIREMENTS[target]
                & all_null_capital_sources
            )
            for target in capital_flow_fields
            if _CAPITAL_FLOW_FEATURE_REQUIREMENTS[target]
            & all_null_capital_sources
        }
        if (
            missing_daily_fields
            or missing_fundamental_fields
            or missing_capital_flow_fields
        ):
            logger.warning(
                "research field contract drift: declared source columns absent "
                "from snapshot parquets (fields skipped, not injected): "
                "daily_basic=%s fundamentals=%s moneyflow=%s",
                missing_daily_fields,
                missing_fundamental_fields,
                missing_capital_flow_fields,
            )
        if (
            all_null_daily_fields
            or all_null_fundamental_fields
            or all_null_capital_flow_fields
        ):
            logger.warning(
                "research field sources contain only null values (fields "
                "injected as all-NaN channels): daily_basic=%s fundamentals=%s "
                "moneyflow=%s",
                all_null_daily_fields,
                all_null_fundamental_fields,
                all_null_capital_flow_fields,
            )
        # Version 2: availability policies and recoverability levels come from
        # the shared registry in quant_data.availability and now also cover the
        # index/industry metadata consumed next to the feature fields.
        # Version 3: fundamental fields are grouped by source statement table
        # (fina_indicator plus the income/balancesheet/cashflow line items).
        # Version 4: declared-vs-available coverage diagnostics distinguish
        # source columns missing from the snapshot parquets from columns that
        # exist but are entirely null.
        # Version 5: evidence-grade moneyflow rows add a compact capital-flow
        # channel (CNY net amount, turnover-scaled net ratio and large-order
        # imbalance) with the same after-close PIT semantics as daily_basic.
        # Version 6: non-default fina_indicator columns are sourced from a
        # narrow companion dataset because the relay rejects an all-field
        # cross-section request; both use the same announcement-date policy.
        availability_datasets = (
            "daily_basic",
            "moneyflow",
            "fina_indicator",
            "fina_indicator_nondefault",
            "income",
            "balancesheet",
            "cashflow",
            "index_weight",
            "index_member_all",
        )
        return {
            "version": 6,
            "daily_fields": daily_fields,
            "fundamental_fields": fundamental_fields,
            "capital_flow_fields": capital_flow_fields,
            "fields": [
                *daily_fields,
                *(target for fields in fundamental_fields.values() for target in fields.values()),
                *capital_flow_fields,
            ],
            "missing_daily_fields": missing_daily_fields,
            "missing_fundamental_fields": missing_fundamental_fields,
            "missing_capital_flow_fields": missing_capital_flow_fields,
            "all_null_daily_fields": all_null_daily_fields,
            "all_null_fundamental_fields": all_null_fundamental_fields,
            "all_null_capital_flow_fields": all_null_capital_flow_fields,
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
        connection = self._duckdb_connection()
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
            "moneyflow": {
                "ts_code",
                "trade_date",
                "net_mf_amount",
                "buy_lg_amount",
                "sell_lg_amount",
                "buy_elg_amount",
                "sell_elg_amount",
            },
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
                "moneyflow": (
                    f"ts_code IS NOT NULL AND {_as_date_sql('trade_date')} IS NOT NULL "
                    "AND try_cast(net_mf_amount AS DOUBLE) IS NOT NULL"
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
            conflict_issue = self._industry_membership_conflict_issue(
                industry_columns
            )
            if conflict_issue:
                issues.append(conflict_issue)

        if not issues:
            coverage_issue = self._benchmark_industry_coverage_issue(
                industry_columns
            )
            if coverage_issue:
                issues.append(coverage_issue)

        if issues:
            raise RuntimeError("Qlib research inputs are incomplete: " + "; ".join(issues))

    def _industry_membership_conflict_issue(
        self, industry_columns: set[str]
    ) -> str | None:
        """Reject overlapping effective intervals with different L1 industries."""

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
        industry_root = self.snapshot_path / "parquet" / "index_member_all"
        industry_glob = _sql_string(
            str((industry_root / "**" / "*.parquet").resolve())
        )
        query = f"""
            WITH industry_rows AS (
                SELECT DISTINCT
                    upper({_nonblank_text_sql(instrument_column)}) AS instrument,
                    upper({_nonblank_text_sql(industry_column)}) AS industry,
                    {_as_date_sql("in_date")} AS in_date,
                    {out_date} AS out_date
                FROM read_parquet(
                    {industry_glob}, hive_partitioning=true, union_by_name=true
                )
                WHERE {_nonblank_text_sql(instrument_column)} IS NOT NULL
                  AND {_nonblank_text_sql(industry_column)} IS NOT NULL
                  AND {_as_date_sql("in_date")} IS NOT NULL
            ),
            conflicts AS (
                SELECT
                    lhs.instrument,
                    greatest(lhs.in_date, rhs.in_date) AS first_conflict_date,
                    lhs.industry AS first_industry,
                    rhs.industry AS second_industry
                FROM industry_rows lhs
                INNER JOIN industry_rows rhs
                  ON lhs.instrument = rhs.instrument
                 AND lhs.industry < rhs.industry
                 AND lhs.in_date <= coalesce(rhs.out_date, DATE '9999-12-31')
                 AND rhs.in_date <= coalesce(lhs.out_date, DATE '9999-12-31')
            )
            SELECT
                count(*) AS conflict_pairs,
                min(first_conflict_date) AS first_conflict_date,
                (
                    SELECT string_agg(
                        instrument || ':' || first_industry || '/' || second_industry,
                        ', '
                    )
                    FROM (
                        SELECT *
                        FROM conflicts
                        ORDER BY first_conflict_date, instrument,
                                 first_industry, second_industry
                        LIMIT 10
                    ) examples
                ) AS examples
            FROM conflicts
        """
        connection = self._duckdb_connection()
        try:
            row = connection.execute(query).fetchone()
        finally:
            connection.close()
        conflict_pairs = int(row[0] or 0) if row is not None else 0
        if conflict_pairs == 0:
            return None
        first_conflict_date = row[1]
        examples = str(row[2] or "")
        return (
            "index_member_all has overlapping distinct point-in-time L1 "
            f"industry intervals in {conflict_pairs} row pairs; first affected "
            f"date {first_conflict_date}: {examples}"
        )

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
                    upper({_nonblank_text_sql("index_code")}) AS benchmark,
                    upper({_nonblank_text_sql("con_code")}) AS instrument,
                    {_as_date_sql("trade_date")} AS weight_date,
                    try_cast(weight AS DOUBLE) AS weight
                FROM read_parquet(
                    {weight_glob}, hive_partitioning=true, union_by_name=true
                )
            ),
            constituents AS (
                SELECT w.weight_date, w.instrument, max(w.weight) AS weight
                FROM weight_rows w
                WHERE w.benchmark = {benchmark}
                  AND w.instrument IS NOT NULL
                  AND w.weight_date IS NOT NULL
                  AND w.weight > 0
                GROUP BY w.weight_date, w.instrument
            ),
            industry_rows AS (
                SELECT
                    upper({_nonblank_text_sql(instrument_column)}) AS instrument,
                    upper({_nonblank_text_sql(industry_column)}) AS industry,
                    {_as_date_sql("in_date")} AS in_date,
                    {out_date} AS out_date
                FROM read_parquet(
                    {industry_glob}, hive_partitioning=true, union_by_name=true
                )
                WHERE {_nonblank_text_sql(instrument_column)} IS NOT NULL
                  AND {_nonblank_text_sql(industry_column)} IS NOT NULL
                  AND {_as_date_sql("in_date")} IS NOT NULL
            ),
            coverage AS (
                SELECT
                    c.weight_date,
                    c.instrument,
                    c.weight,
                    count(i.instrument) > 0 AS covered
                FROM constituents c
                LEFT JOIN industry_rows i
                  ON i.instrument = c.instrument
                 AND i.in_date <= c.weight_date
                 AND (i.out_date IS NULL OR i.out_date >= c.weight_date)
                GROUP BY c.weight_date, c.instrument, c.weight
            ),
            date_coverage AS (
                SELECT
                    weight_date,
                    sum(weight) AS total_weight,
                    coalesce(sum(weight) FILTER (WHERE NOT covered), 0) AS missing_weight
                FROM coverage
                GROUP BY weight_date
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
                ) AS examples,
                (
                    SELECT max(missing_weight / nullif(total_weight, 0))
                    FROM date_coverage
                ) AS max_missing_weight_ratio
            FROM coverage
        """
        connection = self._duckdb_connection()
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
        max_missing_weight_ratio = float(row[5] or 0.0)
        if max_missing_weight_ratio <= _MAX_UNKNOWN_BENCHMARK_WEIGHT_RATIO:
            logger.warning(
                "index_member_all has %s/%s uncovered %s constituent-date rows; "
                "maximum missing benchmark weight %.4f%% is within the %.2f%% "
                "explicit-unknown limit",
                missing_rows,
                total_rows,
                _GOVERNED_BENCHMARK,
                max_missing_weight_ratio * 100.0,
                _MAX_UNKNOWN_BENCHMARK_WEIGHT_RATIO * 100.0,
            )
            return None
        return (
            f"index_member_all has no active point-in-time industry for "
            f"{missing_rows}/{total_rows} {_GOVERNED_BENCHMARK} constituent-date "
            f"rows across {benchmark_dates} benchmark dates; first affected date "
            f"{first_missing_date}; maximum missing benchmark weight "
            f"{max_missing_weight_ratio:.4%} exceeds "
            f"{_MAX_UNKNOWN_BENCHMARK_WEIGHT_RATIO:.2%}: {examples}"
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
            f"{_nonblank_text_sql(instrument)} IS NOT NULL "
            f"AND {_nonblank_text_sql(industry)} IS NOT NULL "
            f"AND {_as_date_sql('in_date')} IS NOT NULL"
        )

    def _has_usable_row(self, dataset: str, predicate: str) -> bool:
        root = self.snapshot_path / "parquet" / dataset
        glob = _sql_string(str((root / "**" / "*.parquet").resolve()))
        connection = self._duckdb_connection()
        try:
            row = connection.execute(
                f"SELECT 1 FROM read_parquet({glob}, hive_partitioning=true, "
                f"union_by_name=true) WHERE {predicate} LIMIT 1"
            ).fetchone()
        finally:
            connection.close()
        return row is not None

    def _parquet_columns(self, dataset: str) -> set[str]:
        cached = self._parquet_columns_cache.get(dataset)
        if cached is not None:
            return set(cached)
        root = self.snapshot_path / "parquet" / dataset
        if not root.exists() or not any(root.rglob("*.parquet")):
            self._parquet_columns_cache[dataset] = set()
            return set()
        glob = _sql_string(str((root / "**" / "*.parquet").resolve()))
        connection = self._duckdb_connection()
        try:
            rows = connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet({glob}, hive_partitioning=true, "
                "union_by_name=true)"
            ).fetchall()
        finally:
            connection.close()
        result = {str(row[0]) for row in rows}
        self._parquet_columns_cache[dataset] = result
        return set(result)

    @staticmethod
    def _missing_market_controls_query(daily_glob: Path, adj_glob: Path, limit_glob: Path) -> str:
        daily = _sql_string(str(daily_glob.resolve()))
        adj = _sql_string(str(adj_glob.resolve()))
        limits = _sql_string(str(limit_glob.resolve()))
        return f"""
            SELECT count(*)
            FROM read_parquet({daily}, hive_partitioning=true, union_by_name=true) d
            LEFT JOIN read_parquet({adj}, hive_partitioning=true, union_by_name=true) a
              ON d.ts_code = a.ts_code AND d.trade_date = a.trade_date
            LEFT JOIN read_parquet({limits}, hive_partitioning=true, union_by_name=true) l
              ON d.ts_code = l.ts_code AND d.trade_date = l.trade_date
            WHERE d.ts_code IS NOT NULL AND d.close IS NOT NULL
              AND (
                a.adj_factor IS NULL OR a.adj_factor <= 0
                OR ((l.up_limit IS NULL) <> (l.down_limit IS NULL))
                OR (
                  l.up_limit IS NOT NULL AND l.down_limit IS NOT NULL
                  AND
                  (l.up_limit <= 0 OR l.down_limit <= 0)
                  AND NOT (l.up_limit >= 99999.0 AND l.down_limit = 0)
                )
              )
        """


def _sql_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


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


def _nonblank_text_sql(column: str) -> str:
    identifier = '"' + column.replace('"', '""') + '"'
    return (
        f"nullif(regexp_replace(CAST({identifier} AS VARCHAR), "
        "'^[[:space:]]+|[[:space:]]+$', '', 'g'), '')"
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


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_qlib_output_manifest(qlib_dir: Path) -> dict[str, Any]:
    """Seal every published Qlib file except the self-referential provenance."""

    root = qlib_dir.resolve()
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == _QLIB_PROVENANCE_PATH:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {"version": QLIB_OUTPUT_MANIFEST_VERSION, "files": files}


def verify_qlib_output_manifest(qlib_dir: Path, provenance: dict[str, Any]) -> None:
    """Fail closed if any sealed Qlib output was added, removed, or changed."""

    recorded = provenance.get("output_manifest")
    if not isinstance(recorded, dict) or recorded.get("version") != QLIB_OUTPUT_MANIFEST_VERSION:
        raise ValueError("Qlib provenance has no supported output file manifest")
    files = recorded.get("files")
    if not isinstance(files, list):
        raise ValueError("Qlib output file manifest is invalid")
    expected: dict[str, tuple[int, str]] = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Qlib output file manifest entry is invalid")
        relative_text = str(item.get("path") or "")
        relative = Path(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or relative_text == _QLIB_PROVENANCE_PATH
            or relative.as_posix() != relative_text
            or ".." in relative.parts
            or relative_text in expected
        ):
            raise ValueError("Qlib output file manifest path is invalid")
        size = item.get("bytes")
        sha256 = str(item.get("sha256") or "").lower()
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError("Qlib output file manifest identity is invalid")
        expected[relative_text] = (size, sha256)

    actual_manifest = build_qlib_output_manifest(qlib_dir)
    actual = {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in actual_manifest["files"]
    }
    if actual != expected:
        raise ValueError("Qlib output files do not match the sealed manifest")


def _qlib_symbol(value: object) -> str | None:
    text = str(value or "")
    if "." not in text:
        return None
    code, exchange = text.split(".", 1)
    return f"{exchange.upper()}{code}"
