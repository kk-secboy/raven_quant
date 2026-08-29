from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from quant_data.qlib_builder import qlib_research_field_catalog

from .upstream_versions import QLIB_COMMIT

FACTOR_LIBRARY_CONTRACT_VERSION = "factor-library-v1"
QLIB_EXPRESSION_CONTRACT_VERSION = "qlib-expression-allowlist-v2"
FACTOR_FAMILY_POLICY_VERSION = "economic-family-v1"

ECONOMIC_FAMILIES = frozenset(
    {
        "market_state",
        "price_action",
        "trend",
        "mean_reversion",
        "volatility_risk",
        "liquidity",
        "capital_flow",
        "value",
        "quality",
        "growth",
        "event_sentiment",
        "mixed",
    }
)

_ALLOWED_FUNCTIONS = frozenset(
    {
        "Abs",
        "Corr",
        "EMA",
        "Greater",
        "If",
        "IdxMax",
        "IdxMin",
        "Less",
        "Log",
        "Max",
        "Mean",
        "Min",
        "Quantile",
        "Rank",
        "Ref",
        "Resi",
        "Rsquare",
        "Slope",
        "Std",
        "Sum",
    }
)
_FIELD_RE = re.compile(r"\$([A-Za-z][A-Za-z0-9_]*)")
_FUNCTION_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*\(")
_SAFE_EXPRESSION_RE = re.compile(r"^[A-Za-z0-9_$.,+\-*/<>=()\s]+$")


def canonical_expression(expression: str) -> str:
    return re.sub(r"\s+", "", str(expression).strip())


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _split_expression_arguments(value: str) -> list[str]:
    arguments: list[str] = []
    depth = 0
    start = 0
    for index, character in enumerate(value):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(value[start:index].strip())
            start = index + 1
    arguments.append(value[start:].strip())
    return arguments


def _literal_nonnegative_integer(value: str) -> int | None:
    candidate = value.strip()
    return int(candidate) if re.fullmatch(r"\+?\d+", candidate) else None


def _expression_lookback(value: str) -> int:
    expression = value.strip()
    calls: list[tuple[str, str]] = []
    index = 0
    while index < len(expression):
        match = re.match(r"([A-Za-z][A-Za-z0-9_]*)\s*\(", expression[index:])
        if match is None:
            index += 1
            continue
        open_index = index + match.end() - 1
        depth = 1
        cursor = open_index + 1
        while cursor < len(expression) and depth:
            if expression[cursor] == "(":
                depth += 1
            elif expression[cursor] == ")":
                depth -= 1
            cursor += 1
        if depth:
            raise ValueError("Qlib factor expression has unbalanced parentheses")
        calls.append((match.group(1), expression[open_index + 1 : cursor - 1]))
        index = cursor
    lookbacks = [0]
    rolling_last_argument = {
        "Corr",
        "EMA",
        "IdxMax",
        "IdxMin",
        "Max",
        "Mean",
        "Min",
        "Rank",
        "Resi",
        "Rsquare",
        "Slope",
        "Std",
        "Sum",
    }
    for function, contents in calls:
        arguments = _split_expression_arguments(contents)
        argument_lookback = max(
            (_expression_lookback(argument) for argument in arguments), default=0
        )
        if function == "Ref":
            offset_value = arguments[1].strip() if len(arguments) > 1 else ""
            if re.fullmatch(r"[+-]?\d+", offset_value) and int(offset_value) < 0:
                raise ValueError("Qlib factor expressions cannot reference future observations")
            offset = _literal_nonnegative_integer(offset_value)
            if offset is None:
                raise ValueError("Qlib Ref operators require a literal non-negative offset")
            lookbacks.append(_expression_lookback(arguments[0]) + offset)
        elif function in rolling_last_argument:
            window = (
                _literal_nonnegative_integer(arguments[-1]) if len(arguments) > 1 else None
            )
            if window is None or window < 1:
                raise ValueError(f"Qlib {function} requires a positive literal window")
            lookbacks.append(
                max(
                    (_expression_lookback(argument) for argument in arguments[:-1]),
                    default=0,
                )
                + window
                - 1
            )
        elif function == "Quantile":
            window = (
                _literal_nonnegative_integer(arguments[1]) if len(arguments) > 2 else None
            )
            if window is None or window < 1:
                raise ValueError("Qlib Quantile requires a positive literal window")
            lookbacks.append(_expression_lookback(arguments[0]) + window - 1)
        else:
            lookbacks.append(argument_lookback)
    return max(lookbacks)


@dataclass(frozen=True, slots=True)
class CompiledExpression:
    expression: str
    canonical_expression: str
    expression_sha256: str
    required_fields: tuple[str, ...]
    functions: tuple[str, ...]
    max_lookback_days: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": QLIB_EXPRESSION_CONTRACT_VERSION,
            "expression": self.expression,
            "canonical_expression": self.canonical_expression,
            "expression_sha256": self.expression_sha256,
            "required_fields": list(self.required_fields),
            "functions": list(self.functions),
            "max_lookback_days": self.max_lookback_days,
        }


def compile_qlib_expression(
    expression: str,
    *,
    allowed_fields: Iterable[str] | None = None,
) -> CompiledExpression:
    candidate = str(expression).strip()
    if not candidate or len(candidate) > 4096:
        raise ValueError("Qlib factor expression must contain 1-4096 characters")
    if not _SAFE_EXPRESSION_RE.fullmatch(candidate):
        raise ValueError("Qlib factor expression contains unsupported characters")
    depth = 0
    for character in candidate:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Qlib factor expression has unbalanced parentheses")
    if depth:
        raise ValueError("Qlib factor expression has unbalanced parentheses")

    functions = tuple(sorted(set(_FUNCTION_RE.findall(candidate))))
    unsupported = sorted(set(functions) - _ALLOWED_FUNCTIONS)
    if unsupported:
        raise ValueError(
            "Qlib factor expression uses unsupported functions: " + ", ".join(unsupported)
        )

    field_catalog = set(allowed_fields or qlib_research_field_catalog())
    required_fields = tuple(sorted(set(_FIELD_RE.findall(candidate))))
    if not required_fields:
        raise ValueError("Qlib factor expression must reference at least one governed field")
    if any(field.lower().startswith("label") for field in required_fields):
        raise ValueError("Qlib factor expressions cannot reference labels")
    unknown_fields = sorted(set(required_fields) - field_catalog)
    if unknown_fields:
        raise ValueError(
            "Qlib factor expression references unavailable fields: "
            + ", ".join(unknown_fields)
        )
    max_lookback = _expression_lookback(candidate)
    normalized = canonical_expression(candidate)
    return CompiledExpression(
        expression=candidate,
        canonical_expression=normalized,
        expression_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        required_fields=required_fields,
        functions=functions,
        max_lookback_days=max_lookback,
    )


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    id: str
    name: str
    expression: str
    expression_sha256: str
    required_fields: tuple[str, ...]
    max_lookback_days: int
    economic_family: str
    family_tags: tuple[str, ...]
    aliases: tuple[str, ...]
    source_refs: tuple[str, ...]
    availability_policy: str
    definition_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": FACTOR_LIBRARY_CONTRACT_VERSION,
            "id": self.id,
            "name": self.name,
            "expression": self.expression,
            "expression_sha256": self.expression_sha256,
            "required_fields": list(self.required_fields),
            "max_lookback_days": self.max_lookback_days,
            "economic_family": self.economic_family,
            "family_tags": list(self.family_tags),
            "aliases": list(self.aliases),
            "source_refs": list(self.source_refs),
            "availability_policy": self.availability_policy,
            "qlib_commit": QLIB_COMMIT,
            "definition_sha256": self.definition_sha256,
        }


def _alpha360_records() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for field in ("close", "open", "high", "low", "vwap", "volume"):
        for offset in range(59, -1, -1):
            name = f"{field.upper()}{offset}"
            if field == "volume":
                expression = (
                    f"Ref($volume, {offset})/($volume+1e-12)"
                    if offset
                    else "$volume/($volume+1e-12)"
                )
                family = "liquidity"
            else:
                expression = (
                    f"Ref(${field}, {offset})/$close"
                    if offset
                    else f"${field}/$close"
                )
                family = "market_state"
            result.append(
                {
                    "alias": f"alpha360:{name}",
                    "name": name,
                    "expression": expression,
                    "family": family,
                    "tags": ("alpha360", field),
                    "source_ref": f"qlib@{QLIB_COMMIT}:Alpha360:{name}",
                }
            )
    return result


def _alpha158_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add(name: str, expression: str, family: str, *tags: str) -> None:
        records.append(
            {
                "alias": f"alpha158:{name}",
                "name": name,
                "expression": expression,
                "family": family,
                "tags": ("alpha158", *tags),
                "source_ref": f"qlib@{QLIB_COMMIT}:Alpha158:{name}",
            }
        )

    for name, expression in (
        ("KMID", "($close-$open)/$open"),
        ("KLEN", "($high-$low)/$open"),
        ("KMID2", "($close-$open)/($high-$low+1e-12)"),
        ("KUP", "($high-Greater($open, $close))/$open"),
        ("KUP2", "($high-Greater($open, $close))/($high-$low+1e-12)"),
        ("KLOW", "(Less($open, $close)-$low)/$open"),
        ("KLOW2", "(Less($open, $close)-$low)/($high-$low+1e-12)"),
        ("KSFT", "(2*$close-$high-$low)/$open"),
        ("KSFT2", "(2*$close-$high-$low)/($high-$low+1e-12)"),
    ):
        add(name, expression, "price_action", "kbar")
    for field in ("open", "high", "low", "vwap"):
        add(f"{field.upper()}0", f"${field}/$close", "market_state", field)

    windows = (5, 10, 20, 30, 60)
    for window in windows:
        rolling = (
            ("ROC", f"Ref($close, {window})/$close", "trend"),
            ("MA", f"Mean($close, {window})/$close", "trend"),
            ("STD", f"Std($close, {window})/$close", "volatility_risk"),
            ("BETA", f"Slope($close, {window})/$close", "trend"),
            ("RSQR", f"Rsquare($close, {window})", "trend"),
            ("RESI", f"Resi($close, {window})/$close", "trend"),
            ("MAX", f"Max($high, {window})/$close", "trend"),
            ("MIN", f"Min($low, {window})/$close", "trend"),
            ("QTLU", f"Quantile($close, {window}, 0.8)/$close", "trend"),
            ("QTLD", f"Quantile($close, {window}, 0.2)/$close", "trend"),
            ("RANK", f"Rank($close, {window})", "mean_reversion"),
            (
                "RSV",
                f"($close-Min($low, {window}))/(Max($high, {window})-Min($low, {window})+1e-12)",
                "mean_reversion",
            ),
            ("IMAX", f"IdxMax($high, {window})/{window}", "trend"),
            ("IMIN", f"IdxMin($low, {window})/{window}", "trend"),
            (
                "IMXD",
                f"(IdxMax($high, {window})-IdxMin($low, {window}))/{window}",
                "trend",
            ),
            ("CORR", f"Corr($close, Log($volume+1), {window})", "liquidity"),
            (
                "CORD",
                f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {window})",
                "liquidity",
            ),
            ("CNTP", f"Mean($close>Ref($close, 1), {window})", "trend"),
            ("CNTN", f"Mean($close<Ref($close, 1), {window})", "trend"),
            (
                "CNTD",
                f"Mean($close>Ref($close, 1), {window})-Mean($close<Ref($close, 1), {window})",
                "trend",
            ),
            (
                "SUMP",
                f"Sum(Greater($close-Ref($close, 1), 0), {window})/"
                f"(Sum(Abs($close-Ref($close, 1)), {window})+1e-12)",
                "trend",
            ),
            (
                "SUMN",
                f"Sum(Greater(Ref($close, 1)-$close, 0), {window})/"
                f"(Sum(Abs($close-Ref($close, 1)), {window})+1e-12)",
                "trend",
            ),
            (
                "SUMD",
                f"(Sum(Greater($close-Ref($close, 1), 0), {window})-"
                f"Sum(Greater(Ref($close, 1)-$close, 0), {window}))/"
                f"(Sum(Abs($close-Ref($close, 1)), {window})+1e-12)",
                "trend",
            ),
            ("VMA", f"Mean($volume, {window})/($volume+1e-12)", "liquidity"),
            ("VSTD", f"Std($volume, {window})/($volume+1e-12)", "liquidity"),
            (
                "WVMA",
                f"Std(Abs($close/Ref($close, 1)-1)*$volume, {window})/"
                f"(Mean(Abs($close/Ref($close, 1)-1)*$volume, {window})+1e-12)",
                "liquidity",
            ),
            (
                "VSUMP",
                f"Sum(Greater($volume-Ref($volume, 1), 0), {window})/"
                f"(Sum(Abs($volume-Ref($volume, 1)), {window})+1e-12)",
                "liquidity",
            ),
            (
                "VSUMN",
                f"Sum(Greater(Ref($volume, 1)-$volume, 0), {window})/"
                f"(Sum(Abs($volume-Ref($volume, 1)), {window})+1e-12)",
                "liquidity",
            ),
            (
                "VSUMD",
                f"(Sum(Greater($volume-Ref($volume, 1), 0), {window})-"
                f"Sum(Greater(Ref($volume, 1)-$volume, 0), {window}))/"
                f"(Sum(Abs($volume-Ref($volume, 1)), {window})+1e-12)",
                "liquidity",
            ),
        )
        for prefix, expression, family in rolling:
            add(f"{prefix}{window}", expression, family, prefix.lower())
    if len(records) != 158:
        raise RuntimeError(f"pinned Alpha158 definition count changed: {len(records)}")
    return records


_SEED_SPECS = (
    ("ret_5d", "$close/Ref($close,5)-1", "trend"),
    ("ret_20d", "$close/Ref($close,20)-1", "trend"),
    ("ret_60d", "$close/Ref($close,60)-1", "trend"),
    (
        "rsi_14",
        "Sum(Greater($close-Ref($close,1),0),14)/(Sum(Abs($close-Ref($close,1)),14)+1e-12)",
        "mean_reversion",
    ),
    (
        "kdj_j_9",
        "3*EMA(($close-Min($low,9))/(Max($high,9)-Min($low,9)+1e-12),3)-2*EMA(EMA(($close-Min($low,9))/(Max($high,9)-Min($low,9)+1e-12),3),3)",
        "mean_reversion",
    ),
    ("boll_z_20", "($close-Mean($close,20))/(Std($close,20)+1e-12)", "mean_reversion"),
    ("realized_vol_20", "Std($close/Ref($close,1)-1,20)", "volatility_risk"),
    (
        "downside_vol_20",
        "Std(Less($close/Ref($close,1)-1,0),20)",
        "volatility_risk",
    ),
    ("range_vol_20", "Mean(($high-$low)/($close+1e-12),20)", "volatility_risk"),
    (
        "max_drawdown_60",
        "Min($close/(Max($close,60)+1e-12)-1,60)",
        "volatility_risk",
    ),
    ("turnover_mean_20", "Mean($turnover_rate_f,20)", "liquidity"),
    (
        "turnover_accel_5_20",
        "Mean($turnover_rate_f,5)/(Mean($turnover_rate_f,20)+1e-12)-1",
        "liquidity",
    ),
    (
        "amount_accel_5_20",
        "Mean($amount,5)/(Mean($amount,20)+1e-12)-1",
        "liquidity",
    ),
    (
        "amihud_20",
        "Mean(Abs($close/Ref($close,1)-1)/($amount+1e-12),20)",
        "liquidity",
    ),
    ("price_volume_corr_20", "Corr($close,Log($volume+1),20)", "liquidity"),
    ("earnings_yield_ttm", "1/($pe_ttm+1e-12)", "value"),
    ("book_to_price", "1/($pb+1e-12)", "value"),
    ("sales_yield_ttm", "1/($ps_ttm+1e-12)", "value"),
    ("roe_ttm", "$fund_roe", "quality"),
    ("debt_to_assets", "$fund_debt_to_assets", "quality"),
    ("ocf_to_revenue", "$fund_ocf_to_revenue", "quality"),
    ("revenue_growth_yoy", "$fund_revenue_yoy", "growth"),
    ("netprofit_growth_yoy", "$fund_netprofit_yoy", "growth"),
    ("operating_profit_growth_yoy", "$fund_op_profit_yoy", "growth"),
)


def _seed_records() -> list[dict[str, Any]]:
    if len(_SEED_SPECS) != 24:
        raise RuntimeError("platform seed factor contract must contain exactly 24 factors")
    return [
        {
            "alias": f"platform_seed:{name}",
            "name": name,
            "expression": expression,
            "family": family,
            "tags": ("platform_seed", family),
            "source_ref": f"quantlab:{FACTOR_LIBRARY_CONTRACT_VERSION}:{name}",
        }
        for name, expression, family in _SEED_SPECS
    ]


def _availability_policy(fields: tuple[str, ...]) -> str:
    catalog = qlib_research_field_catalog()
    availabilities = {catalog[field]["availability"] for field in fields}
    if "next_session_after_announcement" in availabilities:
        return "next_session_after_announcement"
    return "after_same_session_close"


def _build_definitions() -> tuple[FactorDefinition, ...]:
    records = [*_alpha360_records(), *_alpha158_records(), *_seed_records()]
    alpha20 = {
        "RESI5",
        "WVMA5",
        "RSQR5",
        "KLEN",
        "RSQR10",
        "CORR5",
        "CORD5",
        "CORR10",
        "ROC60",
        "RESI10",
        "VSTD5",
        "RSQR60",
        "CORR60",
        "WVMA60",
        "STD5",
        "RSQR20",
        "CORD60",
        "CORD10",
        "CORR20",
        "KLOW",
    }
    for record in records:
        if record["alias"].startswith("alpha158:") and record["name"] in alpha20:
            record.setdefault("extra_aliases", []).append(f"alpha20:{record['name']}")
            record.setdefault("extra_sources", []).append(
                f"rdagent@pinned:Alpha20:{record['name']}"
            )

    merged: dict[str, dict[str, Any]] = {}
    for record in records:
        compiled = compile_qlib_expression(record["expression"])
        current = merged.setdefault(
            compiled.expression_sha256,
            {
                "name": record["name"],
                "compiled": compiled,
                "family": record["family"],
                "tags": set(),
                "aliases": set(),
                "source_refs": set(),
            },
        )
        current["tags"].update(record["tags"])
        current["aliases"].add(record["alias"])
        current["aliases"].update(record.get("extra_aliases", ()))
        current["source_refs"].add(record["source_ref"])
        current["source_refs"].update(record.get("extra_sources", ()))
        if current["family"] != record["family"]:
            previous_family = current["family"]
            current["family"] = "mixed"
            current["tags"].update({previous_family, record["family"]})

    definitions: list[FactorDefinition] = []
    for expression_sha256, item in sorted(merged.items()):
        compiled: CompiledExpression = item["compiled"]
        aliases = tuple(sorted(item["aliases"]))
        source_refs = tuple(sorted(item["source_refs"]))
        family = str(item["family"])
        if family not in ECONOMIC_FAMILIES:
            raise RuntimeError(f"unsupported economic family {family}")
        identity = {
            "contract_version": FACTOR_LIBRARY_CONTRACT_VERSION,
            "expression_sha256": expression_sha256,
            "economic_family": family,
            "aliases": aliases,
            "source_refs": source_refs,
            "required_fields": compiled.required_fields,
            "max_lookback_days": compiled.max_lookback_days,
            "qlib_commit": QLIB_COMMIT,
        }
        definitions.append(
            FactorDefinition(
                id=f"factor-{expression_sha256[:24]}",
                name=str(item["name"]),
                expression=compiled.expression,
                expression_sha256=expression_sha256,
                required_fields=compiled.required_fields,
                max_lookback_days=compiled.max_lookback_days,
                economic_family=family,
                family_tags=tuple(sorted(item["tags"])),
                aliases=aliases,
                source_refs=source_refs,
                availability_policy=_availability_policy(compiled.required_fields),
                definition_sha256=canonical_sha256(identity),
            )
        )
    return tuple(definitions)


FACTOR_DEFINITIONS = _build_definitions()
FACTOR_DEFINITION_BY_ID = {item.id: item for item in FACTOR_DEFINITIONS}
FACTOR_DEFINITION_BY_ALIAS = {
    alias: item for item in FACTOR_DEFINITIONS for alias in item.aliases
}


def list_factor_definitions() -> list[dict[str, Any]]:
    return [item.to_dict() for item in FACTOR_DEFINITIONS]


def get_factor_definition(identifier: str) -> dict[str, Any]:
    item = FACTOR_DEFINITION_BY_ID.get(identifier) or FACTOR_DEFINITION_BY_ALIAS.get(
        identifier
    )
    if item is None:
        raise KeyError(identifier)
    return item.to_dict()


def qlib_expression_contract() -> dict[str, Any]:
    return {
        "contract_version": QLIB_EXPRESSION_CONTRACT_VERSION,
        "allowed_functions": sorted(_ALLOWED_FUNCTIONS),
        "fields": qlib_research_field_catalog(),
        "rules": {
            "future_ref_forbidden": True,
            "negative_time_offset_forbidden": True,
            "label_fields_forbidden": True,
            "required_output": [
                "hypothesis",
                "qlib_expression",
                "required_fields",
                "suggested_economic_family",
            ],
        },
    }


def library_release_definition() -> dict[str, Any]:
    members = [item.definition_sha256 for item in FACTOR_DEFINITIONS]
    definition = {
        "contract_version": FACTOR_LIBRARY_CONTRACT_VERSION,
        "id": "unified-factor-library-v1",
        "qlib_commit": QLIB_COMMIT,
        "member_definition_sha256": sorted(members),
        "member_count": len(members),
        "source_alias_counts": {
            "alpha158": len(_alpha158_records()),
            "alpha360": len(_alpha360_records()),
            "alpha20": 20,
            "platform_seed": len(_SEED_SPECS),
        },
    }
    return {**definition, "definition_sha256": canonical_sha256(definition)}


def feature_expression_map(prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    marker = f"{prefix}:"
    for alias, definition in FACTOR_DEFINITION_BY_ALIAS.items():
        if alias.startswith(marker):
            result[alias.removeprefix(marker)] = definition.expression
    return dict(sorted(result.items()))
