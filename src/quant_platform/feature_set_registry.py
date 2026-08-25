from __future__ import annotations

from typing import Any

from .model_research_governance import canonical_sha256

# This is the exact, fixed 20-feature baseline used by the pinned RD-Agent
# quantitative workflow.  It is copied into QuantLab so a future upstream
# change cannot silently alter a governed fin_model/fin_quant comparison.
RDAGENT_ALPHA20: dict[str, str] = {
    "RESI5": "Resi($close, 5)/$close",
    "WVMA5": (
        "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/"
        "(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)"
    ),
    "RSQR5": "Rsquare($close, 5)",
    "KLEN": "($high-$low)/$open",
    "RSQR10": "Rsquare($close, 10)",
    "CORR5": "Corr($close, Log($volume+1), 5)",
    "CORD5": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)",
    "CORR10": "Corr($close, Log($volume+1), 10)",
    "ROC60": "Ref($close, 60)/$close",
    "RESI10": "Resi($close, 10)/$close",
    "VSTD5": "Std($volume, 5)/($volume+1e-12)",
    "RSQR60": "Rsquare($close, 60)",
    "CORR60": "Corr($close, Log($volume+1), 60)",
    "WVMA60": (
        "Std(Abs($close/Ref($close, 1)-1)*$volume, 60)/"
        "(Mean(Abs($close/Ref($close, 1)-1)*$volume, 60)+1e-12)"
    ),
    "STD5": "Std($close, 5)/$close",
    "RSQR20": "Rsquare($close, 20)",
    "CORD60": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)",
    "CORD10": "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)",
    "CORR20": "Corr($close, Log($volume+1), 20)",
    "KLOW": "(Less($open, $close)-$low)/$open",
}


def _record(feature_set_id: str, name: str, features: dict[str, str]) -> dict[str, Any]:
    definition = {
        "contract_version": "governed-feature-set-v1",
        "id": feature_set_id,
        "name": name,
        "features": dict(features),
    }
    return {**definition, "definition_sha256": canonical_sha256(definition)}


FEATURE_SETS: dict[str, dict[str, Any]] = {
    "governed-baseline": _record(
        "governed-baseline",
        "Pinned RD-Agent Alpha20 baseline",
        RDAGENT_ALPHA20,
    )
}


def get_feature_set(feature_set_id: str) -> dict[str, Any]:
    try:
        item = FEATURE_SETS[feature_set_id]
    except KeyError as exc:
        raise ValueError(f"unknown governed feature set {feature_set_id!r}") from exc
    return {
        **item,
        "features": dict(item["features"]),
    }


def list_feature_sets() -> list[dict[str, Any]]:
    return [
        {
            "id": item["id"],
            "name": item["name"],
            "feature_count": len(item["features"]),
            "definition_sha256": item["definition_sha256"],
            "contract_version": item["contract_version"],
        }
        for item in FEATURE_SETS.values()
    ]
