from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True, slots=True)
class FetchSpec:
    dataset: str
    api_name: str
    scope: dict[str, Any]
    params: dict[str, Any]
    fields: tuple[str, ...] = ()
    allow_empty: bool = False
    max_attempts: int = 5

    @property
    def unit_key(self) -> str:
        material = canonical_json(
            {
                "dataset": self.dataset,
                "api_name": self.api_name,
                "scope": self.scope,
                "params": self.params,
                "fields": self.fields,
            }
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ProviderResult:
    api_name: str
    columns: list[str]
    rows: list[dict[str, Any]]
    raw_body: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkUnit:
    unit_key: str
    spec: FetchSpec
    attempts: int


@dataclass(slots=True)
class UnitResult:
    output_path: str
    row_count: int
    sha256: str
