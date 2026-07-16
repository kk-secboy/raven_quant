import json

import pytest
import requests

from quant_data.provider import ProviderError, TushareHttpProvider, decode_response
from quant_data.rate_limit import GlobalRateGate

pytestmark = pytest.mark.no_database


def test_decodes_standard_tushare_shape() -> None:
    body = json.dumps(
        {
            "code": 0,
            "msg": "",
            "data": {"fields": ["ts_code", "close"], "items": [["000001.SZ", 10.5]]},
        }
    ).encode()
    result = decode_response("daily", body)
    assert result.rows == [{"ts_code": "000001.SZ", "close": 10.5}]


def test_decodes_relay_shape() -> None:
    body = json.dumps(
        {"columns": ["ts_code", "trade_date"], "rows": [["000001.SZ", "20240102"]]}
    ).encode()
    result = decode_response("daily", body)
    assert result.columns == ["ts_code", "trade_date"]
    assert result.rows[0]["trade_date"] == "20240102"


def test_permission_error_is_not_retryable() -> None:
    body = json.dumps({"code": -1, "msg": "没有权限访问该接口"}).encode()
    with pytest.raises(ProviderError) as raised:
        decode_response("income", body)
    assert raised.value.retryable is False


def test_rate_limit_error_is_retryable() -> None:
    body = json.dumps({"code": 429, "msg": "请求频率超限，进入冷却"}).encode()
    with pytest.raises(ProviderError) as raised:
        decode_response("daily", body, 429)
    assert raised.value.retryable is True
    assert raised.value.rate_limited is True


@pytest.mark.parametrize(
    "message,status_code",
    [
        ("服务升级中,请稍后再试", 200),
        ("Service temporarily unavailable", 503),
    ],
)
def test_provider_maintenance_uses_local_retry(
    message: str, status_code: int
) -> None:
    body = json.dumps({"code": 503, "msg": message}).encode()
    with pytest.raises(ProviderError) as raised:
        decode_response("stock_basic", body, status_code)
    assert raised.value.retryable is True
    assert raised.value.rate_limited is False


class _Response:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self.content = json.dumps(payload).encode()


class _Session:
    def __init__(self, responses) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class _Gate:
    def __init__(self) -> None:
        self.waits = 0
        self.cooldowns = 0

    def wait(self) -> None:
        self.waits += 1

    def cooldown(self, _seconds: float) -> bool:
        self.cooldowns += 1
        return self.cooldowns == 1


def _provider(
    session: _Session, gate: _Gate, *, max_attempts: int = 5
) -> TushareHttpProvider:
    provider = TushareHttpProvider(
        api_url="https://example.invalid",
        token="test",
        rate_gate=gate,
        max_attempts=max_attempts,
        cooldown_seconds=180,
    )
    provider._local.session = session
    return provider


def test_http_429_establishes_one_cooldown_and_returns_immediately() -> None:
    session = _Session([_Response(429, {"code": 429, "msg": "too many requests"})])
    gate = _Gate()

    with pytest.raises(ProviderError) as raised:
        _provider(session, gate, max_attempts=1).fetch("daily", {})

    assert session.calls == 1
    assert gate.cooldowns == 1
    assert raised.value.retry_after_seconds == 180


def test_http_503_uses_short_local_backoff_without_global_cooldown(monkeypatch) -> None:
    session = _Session(
        [
            _Response(503, {"code": 503, "msg": "service unavailable"}),
            _Response(200, {"code": 0, "data": {"fields": [], "items": []}}),
        ]
    )
    gate = _Gate()
    sleeps = []
    monkeypatch.setattr("quant_data.provider.time.sleep", sleeps.append)

    result = _provider(session, gate).fetch("daily", {})

    assert result.rows == []
    assert session.calls == 2
    assert gate.cooldowns == 0
    assert len(sleeps) == 1
    assert sleeps[0] < 3


def test_permission_failure_is_not_retried() -> None:
    session = _Session([_Response(200, {"code": -1, "msg": "权限不足"})])
    gate = _Gate()

    with pytest.raises(ProviderError) as raised:
        _provider(session, gate).fetch("income", {})

    assert raised.value.retryable is False
    assert session.calls == 1
    assert gate.cooldowns == 0


def test_network_timeout_uses_local_retry(monkeypatch) -> None:
    session = _Session(
        [
            requests.Timeout("slow"),
            _Response(200, {"code": 0, "data": {"fields": [], "items": []}}),
        ]
    )
    gate = _Gate()
    monkeypatch.setattr("quant_data.provider.time.sleep", lambda _delay: None)

    assert _provider(session, gate).fetch("daily", {}).rows == []
    assert session.calls == 2
    assert gate.cooldowns == 0


def test_shared_cooldown_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr("quant_data.rate_limit.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("quant_data.rate_limit.random.uniform", lambda *_args: 0.0)
    gate = GlobalRateGate(90)

    assert gate.cooldown(180) is True
    first_deadline = gate._cooldown_until
    assert gate.cooldown(180) is False
    assert gate._cooldown_until == first_deadline
