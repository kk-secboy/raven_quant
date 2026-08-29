from __future__ import annotations

import os
import re
from collections.abc import Mapping
from datetime import date
from typing import Any

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.scenarios.qlib.developer.factor_runner import (
    QlibFactorRunner as UpstreamQlibFactorRunner,
)
from rdagent.scenarios.qlib.developer.model_runner import (
    QlibModelRunner as UpstreamQlibModelRunner,
)
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace

_UPSTREAM_MARKET = "market: &market csi300"
_GOVERNED_MARKET = "market: &market cn_all"
_BENCHMARK_ANCHOR = "benchmark: &benchmark SH000300"
_INSTRUMENT_ANCHOR = "instruments: *market"
_CSI300_PATTERN = re.compile(r"\bcsi300\b", flags=re.IGNORECASE)
_PERIOD_NAMES = (
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "test_start",
    "test_end",
)

_FACTOR_CONFIGS = frozenset(
    {
        "conf_baseline.yaml",
        "conf_combined_factors.yaml",
        "conf_combined_factors_sota_model.yaml",
    }
)
_MODEL_CONFIGS = frozenset(
    {
        "conf_baseline_factors_model.yaml",
        "conf_sota_factors_model.yaml",
    }
)


def _validate_template_period_contract(name: str, source: str) -> None:
    missing = [
        period
        for period in _PERIOD_NAMES
        if re.search(r"{{\s*" + re.escape(period) + r"\b", source) is None
    ]
    if missing:
        raise RuntimeError(
            f"RD-Agent Qlib template {name} omits governed periods: {missing}"
        )


def _expected_period_environment(prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in _PERIOD_NAMES:
        value = str(os.getenv(f"{prefix}_{name.upper()}") or "").strip()
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise RuntimeError(
                f"governed RD-Agent period {prefix}_{name.upper()} is missing or invalid"
            ) from exc
        result[name] = value
    ordered = [result[name] for name in _PERIOD_NAMES]
    if ordered != sorted(ordered):
        raise RuntimeError(f"governed RD-Agent periods for {prefix} are not ordered")
    if prefix != "QLIB_QUANT":
        quant = {
            name: str(os.getenv(f"QLIB_QUANT_{name.upper()}") or "").strip()
            for name in _PERIOD_NAMES
        }
        if result != quant:
            raise RuntimeError(
                f"governed RD-Agent periods for {prefix} disagree with QLIB_QUANT"
            )
    return result


def _validate_qlib_run_environment(run_env: Any, *, prefix: str) -> None:
    if not isinstance(run_env, Mapping):
        raise RuntimeError("RD-Agent Qlib run environment must be an object")
    expected = _expected_period_environment(prefix)
    actual = {name: str(run_env.get(name) or "") for name in _PERIOD_NAMES}
    if actual != expected:
        raise RuntimeError(
            f"RD-Agent Qlib effective dates disagree with governed {prefix} dates: "
            f"expected={expected}, got={actual}"
        )


class _GovernedQlibFBWorkspace(QlibFBWorkspace):
    """Pickle-safe execute guard for official RD-Agent Qlib workspaces."""

    def execute(
        self,
        qlib_config_name: str = "conf.yaml",
        run_env: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        prefix = str(getattr(self, "_quantlab_execute_guard", ""))
        expected_configs = getattr(self, "_quantlab_expected_configs", frozenset())
        if not prefix or not isinstance(expected_configs, frozenset):
            raise RuntimeError("governed RD-Agent Qlib execute guard is incomplete")
        if qlib_config_name not in expected_configs:
            raise RuntimeError(
                f"RD-Agent requested an unreviewed Qlib template: {qlib_config_name}"
            )
        source = self.file_dict.get(qlib_config_name)
        if not isinstance(source, str):
            raise RuntimeError(f"RD-Agent Qlib template {qlib_config_name} is unavailable")
        _validate_template_period_contract(qlib_config_name, source)
        _validate_qlib_run_environment(run_env, prefix=prefix)
        return super().execute(qlib_config_name, run_env, *args, **kwargs)


def _install_execute_guard(
    workspace: QlibFBWorkspace,
    *,
    prefix: str,
    expected_configs: frozenset[str],
) -> None:
    guarded = getattr(workspace, "_quantlab_execute_guard", None)
    if guarded == prefix:
        return
    if guarded is not None:
        raise RuntimeError("RD-Agent Qlib workspace was reused across governed runner types")
    if type(workspace) is not QlibFBWorkspace:
        raise RuntimeError("RD-Agent Qlib workspace class contract drifted")
    workspace.__class__ = _GovernedQlibFBWorkspace
    workspace._quantlab_execute_guard = prefix  # type: ignore[attr-defined]
    workspace._quantlab_expected_configs = expected_configs  # type: ignore[attr-defined]


def _govern_experiment_market(
    experiment: QlibFactorExperiment | QlibModelExperiment,
    expected_configs: frozenset[str],
) -> None:
    """Keep RD-Agent's internal experiment universe aligned with QuantLab.

    The pinned upstream templates use CSI300, while every independently
    evaluated QuantLab candidate uses the governed ``cn_all`` universe.  Patch
    only the reviewed template anchors and fail closed on any upstream drift.
    The CSI300 price series remains the benchmark; it is not the stock pool.
    """

    if RD_AGENT_SETTINGS.cache_with_pickle:
        raise RuntimeError("governed RD-Agent runners require pickle cache disabled")
    workspace = experiment.experiment_workspace
    if not isinstance(workspace, QlibFBWorkspace):
        raise RuntimeError("RD-Agent experiment workspace contract drifted")
    actual_configs = frozenset(
        name
        for name in workspace.file_dict
        if name.lower().endswith((".yaml", ".yml"))
    )
    if actual_configs != expected_configs:
        raise RuntimeError(
            "RD-Agent Qlib template set drifted: "
            f"expected {sorted(expected_configs)}, got {sorted(actual_configs)}"
        )

    updates: dict[str, str] = {}
    for name in sorted(expected_configs):
        source = workspace.file_dict.get(name)
        if not isinstance(source, str):
            raise RuntimeError(f"RD-Agent Qlib template {name} is unavailable")
        upstream_count = source.count(_UPSTREAM_MARKET)
        governed_count = source.count(_GOVERNED_MARKET)
        if (upstream_count, governed_count) == (1, 0):
            governed = source.replace(_UPSTREAM_MARKET, _GOVERNED_MARKET)
        elif (upstream_count, governed_count) == (0, 1):
            governed = source
        else:
            raise RuntimeError(f"RD-Agent Qlib template {name} market anchor drifted")
        if _CSI300_PATTERN.search(governed):
            raise RuntimeError(f"RD-Agent Qlib template {name} still selects CSI300")
        if governed.count(_GOVERNED_MARKET) != 1:
            raise RuntimeError(f"RD-Agent Qlib template {name} governed market is ambiguous")
        if governed.count(_BENCHMARK_ANCHOR) != 1:
            raise RuntimeError(f"RD-Agent Qlib template {name} benchmark anchor drifted")
        if governed.count(_INSTRUMENT_ANCHOR) != 1:
            raise RuntimeError(f"RD-Agent Qlib template {name} instrument anchor drifted")
        _validate_template_period_contract(name, governed)
        updates[name] = governed

    # QlibFBWorkspace.execute reads the physical workspace, so update both its
    # reproducible file_dict and the files that qrun will consume.
    workspace.inject_files(**updates)


class QuantLabFactorRunner(UpstreamQlibFactorRunner):
    def develop(self, exp: QlibFactorExperiment) -> QlibFactorExperiment:
        _govern_experiment_market(exp, _FACTOR_CONFIGS)
        _install_execute_guard(
            exp.experiment_workspace,
            prefix="QLIB_FACTOR",
            expected_configs=_FACTOR_CONFIGS,
        )
        return super().develop(exp)


class QuantLabModelRunner(UpstreamQlibModelRunner):
    def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
        _govern_experiment_market(exp, _MODEL_CONFIGS)
        _install_execute_guard(
            exp.experiment_workspace,
            prefix="QLIB_MODEL",
            expected_configs=_MODEL_CONFIGS,
        )
        return super().develop(exp)
