from __future__ import annotations

import json
import random
import threading
import time
from typing import Any

import requests

from .models import ProviderResult
from .rate_limit import GlobalRateGate


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        rate_limited: bool = False,
        status_code: int | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.rate_limited = rate_limited
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


def _records(columns: list[str], rows: Any) -> list[dict[str, Any]]:
    if rows is None:
        return []
    if isinstance(rows, list) and (not rows or isinstance(rows[0], dict)):
        return list(rows)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if len(row) != len(columns):
            raise ProviderError(
                f"row {index} has {len(row)} values for {len(columns)} columns",
                retryable=False,
            )
        result.append(dict(zip(columns, row, strict=True)))
    return result


def decode_response(api_name: str, body: bytes, status_code: int = 200) -> ProviderResult:
    try:
        root = json.loads(body)
    except json.JSONDecodeError as exc:
        preview = body[:160].decode("utf-8", errors="replace")
        raise ProviderError(
            f"provider returned non-JSON content (HTTP {status_code}): {preview}",
            retryable=status_code == 429 or status_code >= 500 or status_code == 200,
            rate_limited=status_code == 429,
            status_code=status_code,
        ) from exc

    code = root.get("code", 0)
    message = str(root.get("msg") or root.get("message") or "").strip()
    message_lower = message.lower()
    explicit_rate_limit = any(
        token in message_lower
        for token in (
            "rate limit",
            "request limit",
            "too many request",
            "too frequent",
            "frequency limit",
            "频率",
            "限频",
            "请求过于频繁",
            "请求次数超限",
            "每分钟",
            "冷却",
        )
    )
    maintenance = any(
        token in message_lower
        for token in (
            "服务升级",
            "系统维护",
            "稍后再试",
            "maintenance",
            "temporarily unavailable",
            "service unavailable",
        )
    )
    rate_limited = status_code == 429 or explicit_rate_limit
    no_data = status_code == 200 and any(
        token in message_lower
        for token in (
            "指定数据不存在",
            "data does not exist",
            "no data exists",
        )
    )
    if no_data:
        # Some Tushare endpoints report an empty but otherwise valid slice as
        # code 50101 instead of returning code=0 with an empty items array.
        # Preserve this as a successful empty response so the work unit's
        # allow_empty policy decides whether it is acceptable.  Do not treat
        # every 50101 this way: the same code is also used for pagination and
        # parameter errors that must remain visible to recovery logic.
        return ProviderResult(
            api_name=api_name,
            columns=[],
            rows=[],
            raw_body=body,
            metadata={"provider_code": code, "provider_message": message},
        )
    if status_code < 200 or status_code >= 300 or str(code) not in {"0", "", "None"}:
        permission_error = any(
            token in message_lower
            for token in (
                "permission",
                "forbidden",
                "unauthorized",
                "权限",
                "无权",
                "积分不足",
            )
        )
        validation_error = any(
            token in message_lower
            for token in (
                "invalid parameter",
                "missing parameter",
                "bad request",
                "参数错误",
                "参数不能为空",
                "必填参数",
                "格式错误",
            )
        )
        transient = status_code in {502, 503, 504} or maintenance
        raise ProviderError(
            f"provider error code={code} http={status_code}: {message or 'unknown error'}",
            retryable=rate_limited or transient or not (permission_error or validation_error),
            rate_limited=rate_limited,
            status_code=status_code,
        )

    payload = root.get("data") if isinstance(root.get("data"), dict) else root
    columns = payload.get("fields") or payload.get("columns") or []
    rows = payload.get("items")
    if rows is None:
        rows = payload.get("rows")
    if not columns and rows:
        if isinstance(rows[0], dict):
            columns = list(rows[0])
        else:
            raise ProviderError("successful response has rows but no columns", retryable=True)
    metadata = payload.get("response_meta") or root.get("response_meta") or {}
    return ProviderResult(
        api_name=api_name,
        columns=list(columns),
        rows=_records(list(columns), rows or []),
        raw_body=body,
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


class TushareHttpProvider:
    """Tushare wire-protocol client compatible with the shared-note relay."""

    def __init__(
        self,
        *,
        api_url: str,
        token: str,
        rate_gate: GlobalRateGate,
        timeout_seconds: float = 60.0,
        max_attempts: int = 5,
        cooldown_seconds: float = 180.0,
    ) -> None:
        self.api_url = api_url
        self.token = token
        self.rate_gate = rate_gate
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.cooldown_seconds = cooldown_seconds
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def fetch(
        self, api_name: str, params: dict[str, Any], fields: tuple[str, ...] = ()
    ) -> ProviderResult:
        payload: dict[str, Any] = {
            "api_name": api_name,
            "token": self.token,
            "params": params,
        }
        if fields:
            payload["fields"] = ",".join(fields)

        last_error: ProviderError | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.rate_gate.wait()
            try:
                response = self._session().post(
                    self.api_url,
                    json=payload,
                    timeout=(10.0, self.timeout_seconds),
                    headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
                )
                return decode_response(api_name, response.content, response.status_code)
            except requests.RequestException as exc:
                last_error = ProviderError(str(exc), retryable=True)
            except ProviderError as exc:
                last_error = exc

            if last_error is None or not last_error.retryable:
                break
            if last_error.rate_limited:
                self.rate_gate.cooldown(self.cooldown_seconds)
                last_error.retry_after_seconds = max(1, int(self.cooldown_seconds))
                break
            if attempt >= self.max_attempts:
                break
            delay = min(30.0, (2 ** (attempt - 1)) + random.uniform(0.0, 1.0))
            time.sleep(delay)
        assert last_error is not None
        raise last_error
