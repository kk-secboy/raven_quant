"""Drive the pinned Qlib Exchange with its real source, without a qlib install.

The formal backtest chain runs inside the pinned qlib checkout (QLIB_REPO,
commit pinned by ``quant_platform.upstream_versions.QLIB_COMMIT``) which is not
importable in the Windows test venv.  This harness loads the *real* pinned
modules (``qlib.backtest.exchange`` and its leaf dependencies) from that
checkout through a minimal package shim, replacing only the data provider
(``qlib.data.data.D``) and the quote container — the exchange decision logic
under differential test is the genuine pinned code.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.upstream_versions import QLIB_COMMIT

# Raw columns the harness quote frame must carry (without the leading "$").
RAW_QUOTE_COLUMNS = (
    "open",
    "close",
    "vwap",
    "volume",
    "paused",
    "up_limit",
    "down_limit",
    "factor",
    "change",
)


class PinnedQlibUnavailable(RuntimeError):
    pass


def pinned_qlib_root() -> Path | None:
    """Locate the pinned qlib checkout and verify it sits on the pinned commit."""

    root = Path(os.environ.get("QLIB_REPO", "E:/projects/qlib"))
    if not (root / "qlib" / "backtest" / "exchange.py").is_file():
        return None
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except OSError:
        head = ""
    if head != QLIB_COMMIT:
        raise PinnedQlibUnavailable(
            f"qlib checkout at {root} is {head or 'unknown'}, expected pinned {QLIB_COMMIT}"
        )
    return root


# ---------------------------------------------------------------------------
# Minimal expression evaluation for the adapter-provided limit/volume strings
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def _boolify(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return value.astype(bool)
    return bool(value)


_EXPRESSION_FUNCTIONS = {
    "Or": lambda a, b: _boolify(a) | _boolify(b),
    "And": lambda a, b: _boolify(a) & _boolify(b),
    "Not": lambda a: ~_boolify(a),
    "Ge": lambda a, b: a >= b,
    "Gt": lambda a, b: a > b,
    "Le": lambda a, b: a <= b,
    "Lt": lambda a, b: a < b,
    "Eq": lambda a, b: a == b,
    "Mul": lambda a, b: a * b,
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Div": lambda a, b: a / b,
    "Abs": lambda a: a.abs() if isinstance(a, pd.Series) else abs(a),
}


def _evaluate_expression(expression: str, frame: pd.DataFrame) -> Any:
    """Evaluate adapter expressions (Or/Ge/Le/arithmetic) over raw ``$`` columns.

    Only strings hard-coded in ``qlib_backtest``/``qlib_exchange`` reach this
    evaluator (limit thresholds and the participation volume clip).
    """

    translated = _TOKEN.sub(r"__raw__('\1')", expression)
    env = {**_EXPRESSION_FUNCTIONS, "__raw__": frame.__getitem__}
    return eval(translated, {"__builtins__": {}}, env)  # noqa: S307


class HarnessProvider:
    """Replacement for ``qlib.data.data.D`` backed by an in-memory raw frame."""

    def __init__(self) -> None:
        self.raw: pd.DataFrame | None = None

    def instruments(self, *_args: Any, **_kwargs: Any) -> list[str]:
        raise NotImplementedError("the harness provider requires explicit codes")

    def features(
        self,
        codes: list[str],
        fields: list[str],
        _start_time: Any,
        _end_time: Any,
        freq: str = "day",
        disk_cache: bool = False,
    ) -> pd.DataFrame:
        del freq, disk_cache
        if self.raw is None:
            raise RuntimeError("harness raw frame is not installed")
        mask = self.raw.index.get_level_values("instrument").isin(list(codes))
        selected = self.raw.loc[mask]
        result = pd.DataFrame(index=selected.index)
        for field in fields:
            if field.startswith("$"):
                result[field] = selected[field[1:]]
            else:
                result[field] = _evaluate_expression(field, selected)
        return result


class _UnsupportedCalendar:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError("qlib Cal is outside the differential harness scope")


_PROVIDER = HarnessProvider()


class HarnessQuote:
    """Replacement quote container implementing the BaseQuote access surface.

    Only the slicing/aggregation the pinned Exchange performs is supported;
    golden cases use single-bar windows so ``sum``/``all``/``ts_data_last``
    collapse to that bar.
    """

    def __init__(self, quote_df: pd.DataFrame, freq: str) -> None:
        del freq
        self._frames = {
            str(instrument): frame.droplevel("instrument").sort_index()
            for instrument, frame in quote_df.groupby(level="instrument")
        }

    def get_all_stock(self) -> Any:
        return self._frames.keys()

    def get_data(
        self,
        stock_id: str,
        start_time: Any,
        end_time: Any,
        field: str,
        method: str | None = None,
    ) -> Any:
        frame = self._frames.get(str(stock_id))
        if frame is None or field not in frame.columns:
            return None
        series = frame[field]
        start = pd.Timestamp(start_time)
        end = pd.Timestamp(end_time)
        sliced = series.loc[(series.index >= start) & (series.index < end)]
        if sliced.empty:
            return None
        if method == "sum":
            return float(sliced.sum())
        if method == "all":
            return bool(sliced.astype(bool).all())
        if method == "ts_data_last":
            return sliced.iloc[-1]
        if len(sliced) == 1:
            return sliced.iloc[0]
        index_data = sys.modules.get("qlib.utils.index_data")
        if index_data is not None:
            return index_data.SingleData(sliced)
        return sliced


def _shell_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_real_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PinnedQlibUnavailable(f"cannot load pinned module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_pinned_qlib(root: Path) -> None:
    """Load the real pinned qlib backtest modules through a minimal shim."""

    qlib_dir = root / "qlib"
    _shell_package("qlib", qlib_dir)
    _shell_package("qlib.utils", qlib_dir / "utils")
    _shell_package("qlib.data", qlib_dir / "data")
    _shell_package("qlib.backtest", qlib_dir / "backtest")

    data_module = types.ModuleType("qlib.data.data")
    data_module.D = _PROVIDER
    data_module.Cal = _UnsupportedCalendar
    sys.modules["qlib.data.data"] = data_module

    quote_module = types.ModuleType("qlib.backtest.high_performance_ds")
    quote_module.BaseQuote = HarnessQuote
    quote_module.NumpyQuote = HarnessQuote
    sys.modules["qlib.backtest.high_performance_ds"] = quote_module

    # The pinned qlib config imports pydantic_settings, which the test venv
    # does not carry; the settings surface qlib.config uses is a plain
    # defaults container, so a minimal stand-in keeps the real config code.
    if "pydantic_settings" not in sys.modules:
        settings_module = types.ModuleType("pydantic_settings")

        class _BaseSettings:
            def __init__(self, **kwargs: Any) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        settings_module.BaseSettings = _BaseSettings
        settings_module.SettingsConfigDict = dict
        sys.modules["pydantic_settings"] = settings_module

    for name, relative in (
        ("qlib.constant", "constant.py"),
        ("qlib.config", "config.py"),
        ("qlib.log", "log.py"),
        ("qlib.utils.index_data", "utils/index_data.py"),
        ("qlib.utils.time", "utils/time.py"),
        ("qlib.backtest.utils", "backtest/utils.py"),
        ("qlib.backtest.decision", "backtest/decision.py"),
        ("qlib.backtest.position", "backtest/position.py"),
        ("qlib.backtest.exchange", "backtest/exchange.py"),
    ):
        _load_real_module(name, qlib_dir / relative)

    # qlib.init normally merges the region config (trade_unit, deal_price);
    # the harness never initializes qlib, so apply the CN region explicitly.
    config_module = sys.modules["qlib.config"]
    constant_module = sys.modules["qlib.constant"]
    config_module.C.set_region(constant_module.REG_CN)


def raw_quote_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the (instrument, datetime)-indexed raw frame from row dicts."""

    frame = pd.DataFrame(rows)
    missing = set(RAW_QUOTE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"raw quote rows are missing columns: {sorted(missing)}")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    return frame.set_index(["instrument", "datetime"]).sort_index()


def make_position(*, cash: float, holdings: dict[str, float] | None = None) -> Any:
    """Create a real pinned Position for cash/position-sensitive order paths."""

    position_module = sys.modules["qlib.backtest.position"]
    position_dict = {
        instrument: {"amount": float(amount)} for instrument, amount in (holdings or {}).items()
    }
    return position_module.Position(cash=cash, position_dict=position_dict)


def run_pinned_exchange(
    *,
    raw: pd.DataFrame,
    orders: list[dict[str, Any]],
    freq: str,
    deal_price: str,
    limit_price: str,
    cost_schedule: Any,
    start_time: Any,
    end_time: Any,
) -> list[dict[str, Any]]:
    """Run order specs through the real pinned SquareRootImpactExchange.

    ``limit_price`` names the raw price field the limit expressions judge on
    ("open" for the daily formal chain, "vwap" for the minute formal chain).
    """

    from quant_platform.qlib_exchange import SquareRootImpactExchange

    _PROVIDER.raw = raw
    codes = sorted(str(code) for code in raw.index.get_level_values("instrument").unique())
    exchange = SquareRootImpactExchange(
        cost_schedule=cost_schedule,
        freq=freq,
        start_time=pd.Timestamp(start_time),
        end_time=pd.Timestamp(end_time),
        codes=codes,
        deal_price=deal_price,
        limit_threshold=(
            f"Or($paused, Ge(${limit_price}, $up_limit))",
            f"Or($paused, Le(${limit_price}, $down_limit))",
        ),
        quote_cls=HarnessQuote,
    )
    decision = sys.modules["qlib.backtest.decision"]
    results: list[dict[str, Any]] = []
    for spec in orders:
        order = decision.Order(
            stock_id=str(spec["instrument"]).upper(),
            amount=float(spec["quantity"]),
            direction=decision.Order.BUY if spec["side"] == "buy" else decision.Order.SELL,
            start_time=pd.Timestamp(spec["start_time"]),
            end_time=pd.Timestamp(spec["end_time"]),
        )
        trade_value, trade_cost, trade_price = exchange.deal_order(
            order,
            position=spec.get("position"),
            dealt_order_amount=defaultdict(float),
        )
        results.append(
            {
                "instrument": order.stock_id,
                "side": spec["side"],
                "requested_quantity": float(spec["quantity"]),
                "filled_quantity": float(order.deal_amount),
                "trade_price": float(trade_price) if trade_price == trade_price else None,
                "trade_value": float(trade_value),
                "cost": float(trade_cost),
                "factor": order.factor,
            }
        )
    return results
