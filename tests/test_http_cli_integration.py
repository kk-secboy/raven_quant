from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from typer.testing import CliRunner

from quant_data.cli import app


def value_for(field: str):
    values = {
        "ts_code": "000001.SZ",
        "symbol": "000001",
        "name": "Ping An Bank",
        "area": "Shenzhen",
        "industry": "Bank",
        "market": "Main Board",
        "exchange": "SSE",
        "list_status": "L",
        "list_date": "19910403",
        "delist_date": "",
        "is_hs": "S",
        "cal_date": "20240102",
        "pretrade_date": "20231229",
        "is_open": 1,
        "trade_date": "20240102",
        "ann_date": "20240102",
        "end_date": "20231231",
        "pre_date": "20240102",
        "actual_date": "20240102",
        "modify_date": "20240102",
        "index_code": "801010.SI",
        "con_code": "000001.SZ",
        "in_date": "20240102",
        "out_date": "",
        "is_new": "Y",
    }
    return values.get(field, 1.0)


class FakeTushareHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.__class__.requests.append(payload)
        api_name = payload["api_name"]
        fields = [item for item in payload.get("fields", "").split(",") if item]
        params = payload.get("params", {})
        if api_name == "trade_cal":
            fields = fields or ["exchange", "cal_date", "is_open", "pretrade_date"]
            rows = [[value_for(field) for field in fields]]
        elif api_name == "stock_basic" and params.get("list_status") in {"L", "D"}:
            rows = [[value_for(field) for field in fields]]
        elif api_name == "stock_basic":
            rows = []
        elif api_name in {
            "daily",
            "adj_factor",
            "daily_basic",
            "stk_limit",
            "index_daily",
            "index_weight",
            "fund_daily",
            "fund_adj",
        }:
            fields = fields or ["ts_code", "trade_date", "up_limit", "down_limit"]
            rows = [[value_for(field) for field in fields]]
        elif api_name == "fund_basic":
            fields = ["ts_code", "name", "market", "list_date"]
            rows = [["510300.SH", "CSI 300 ETF", "E", "20120201"]]
        elif api_name == "index_classify":
            fields = ["index_code", "industry_name", "level", "src"]
            rows = [["801010.SI", "Agriculture", params["level"], "SW2021"]]
        elif api_name == "disclosure_date":
            rows = [[value_for(field) for field in fields]]
        elif api_name == "index_member_all":
            fields = ["index_code", "con_code", "in_date", "out_date", "is_new"]
            rows = [[value_for(field) for field in fields]]
        else:
            fields = fields or ["ts_code", "trade_date"]
            rows = []
        body = json.dumps(
            {"code": 0, "msg": "", "data": {"fields": fields, "items": rows}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        return


def test_bootstrap_cli_over_real_http(tmp_path: Path, monkeypatch) -> None:
    FakeTushareHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTushareHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("TUSHARE_API_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("TUSHARE_TOKEN", "test-secret")
        monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
        monkeypatch.setenv("REQUESTS_PER_MINUTE", "60000")
        monkeypatch.setenv("DOWNLOAD_WORKERS", "2")
        monkeypatch.setenv("MAX_REQUEST_ATTEMPTS", "1")
        result = CliRunner().invoke(
            app,
            [
                "bootstrap",
                "--profile",
                "full",
                "--start",
                "2024-01-02",
                "--end",
                "2024-01-02",
                "--snapshot-name",
                "integration",
                "--download-only",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not (tmp_path / "data" / "snapshots").exists()
        verified = CliRunner().invoke(app, ["verify"])
        assert verified.exit_code == 0, verified.output
        snapshot = CliRunner().invoke(
            app,
            [
                "snapshot",
                "--name",
                "integration",
                "--start",
                "2024-01-02",
                "--end",
                "2024-01-02",
                "--profile",
                "full",
            ],
        )
        assert snapshot.exit_code == 0, snapshot.output
        assert (tmp_path / "data" / "snapshots" / "integration" / "manifest.json").exists()
        assert all(request["token"] == "test-secret" for request in FakeTushareHandler.requests)
        called = {request["api_name"] for request in FakeTushareHandler.requests}
        assert {
            "stock_basic",
            "trade_cal",
            "daily",
            "adj_factor",
            "daily_basic",
            "suspend_d",
            "stk_limit",
            "limit_list_d",
            "index_daily",
            "index_weight",
            "moneyflow",
            "margin_detail",
            "hsgt_top10",
            "top_list",
            "top_inst",
            "fund_basic",
            "fund_daily",
            "fund_adj",
            "index_classify",
            "index_member_all",
            "disclosure_date",
            "income_vip",
            "balancesheet_vip",
            "cashflow_vip",
            "fina_indicator_vip",
            "forecast_vip",
            "express_vip",
            "namechange",
            "dividend",
            "repurchase",
            "share_float",
            "pledge_stat",
            "pledge_detail",
            "stk_holdertrade",
            "anns_d",
            "news",
        } <= called
        assert (
            not {
                "income",
                "balancesheet",
                "cashflow",
                "fina_indicator",
                "forecast",
                "express",
            }
            & called
        )
        daily_calls = [
            request for request in FakeTushareHandler.requests if request["api_name"] == "daily"
        ]
        assert daily_calls == [
            {
                "api_name": "daily",
                "token": "test-secret",
                "params": {"trade_date": "20240102"},
                "fields": (
                    "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
                ),
            }
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
