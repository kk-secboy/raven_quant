from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from .checkpoint import CheckpointStore
from .execution_data import validate_and_normalize
from .models import ProviderResult, WorkUnit
from .provider import ProviderError
from .storage import ParquetStore


class Provider(Protocol):
    def fetch(
        self, api_name: str, params: dict[str, object], fields: tuple[str, ...] = ()
    ) -> ProviderResult: ...


@dataclass(slots=True)
class RunSummary:
    succeeded: int = 0
    failed: int = 0
    rows: int = 0


class DownloadRunner:
    def __init__(
        self,
        *,
        checkpoint: CheckpointStore,
        storage: ParquetStore,
        provider: Provider,
        workers: int,
        on_result: Callable[[str, bool, int], None] | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.storage = storage
        self.provider = provider
        self.workers = max(1, workers)
        self.on_result = on_result
        self._lock = threading.Lock()

    def run(self, datasets: set[str] | None = None) -> RunSummary:
        self.checkpoint.reset_stale()
        summary = RunSummary()
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="fetch") as pool:
            futures = [pool.submit(self._worker, summary, datasets) for _ in range(self.workers)]
            for future in futures:
                future.result()
        return summary

    def _worker(self, summary: RunSummary, datasets: set[str] | None) -> None:
        while True:
            unit = self.checkpoint.claim(datasets=datasets)
            if unit is None:
                return
            self._execute(unit, summary)

    def _execute(self, unit: WorkUnit, summary: RunSummary) -> None:
        try:
            response = self.provider.fetch(unit.spec.api_name, unit.spec.params, unit.spec.fields)
            response = validate_and_normalize(unit.spec, response)
            if not response.rows and not unit.spec.allow_empty:
                raise ProviderError(
                    f"{unit.spec.dataset} returned an empty successful response",
                    retryable=True,
                )
            result = self.storage.write_unit(unit.spec.dataset, unit.unit_key, response)
            self.checkpoint.succeed(unit.unit_key, result)
            with self._lock:
                summary.succeeded += 1
                summary.rows += result.row_count
            if self.on_result:
                self.on_result(unit.spec.dataset, True, result.row_count)
        except Exception as exc:
            retry_after = 30 if isinstance(exc, ProviderError) and exc.retryable else 0
            terminal = isinstance(exc, ProviderError) and not exc.retryable
            self.checkpoint.fail(
                unit.unit_key,
                str(exc),
                retry_after_seconds=retry_after,
                terminal=terminal,
            )
            with self._lock:
                summary.failed += 1
            if self.on_result:
                self.on_result(unit.spec.dataset, False, 0)
