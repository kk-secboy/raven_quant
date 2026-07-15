import json

import pytest

from quant_data.provider import ProviderError, decode_response


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
def test_provider_maintenance_uses_shared_cooldown(
    message: str, status_code: int
) -> None:
    body = json.dumps({"code": 503, "msg": message}).encode()
    with pytest.raises(ProviderError) as raised:
        decode_response("stock_basic", body, status_code)
    assert raised.value.retryable is True
    assert raised.value.rate_limited is True
