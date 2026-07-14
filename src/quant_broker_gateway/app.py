from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import GatewaySettings
from .protocols import BrokerAdapter
from .qmt import QmtBrokerAdapter
from .security import HmacRequestVerifier, SignatureError
from .store import GatewayStore, GatewayStoreError

LOGGER = logging.getLogger(__name__)
MAX_BODY_BYTES = 2_000_000


def create_app(
    settings: GatewaySettings | None = None,
    adapter: BrokerAdapter | None = None,
) -> FastAPI:
    resolved = settings or GatewaySettings.from_env()
    resolved.validate()
    provider = adapter or QmtBrokerAdapter(
        mini_path=resolved.qmt_mini_path,
        account_id=resolved.qmt_account_id,
        account_ref=resolved.account_ref,
        session_id=resolved.qmt_session_id,
        volume_multiplier=resolved.volume_multiplier,
        max_quote_age_seconds=resolved.max_quote_age_seconds,
    )
    store = GatewayStore(
        resolved.database_url,
        provider,
        account_ref=resolved.account_ref,
        max_slice_lateness_seconds=resolved.max_slice_lateness_seconds,
        cancel_after_seconds=resolved.cancel_after_seconds,
        max_replacements=resolved.max_replacements,
        max_reprice_bps=resolved.max_reprice_bps,
    )
    verifier = HmacRequestVerifier(
        resolved.hmac_secret,
        max_clock_skew_seconds=resolved.max_clock_skew_seconds,
        nonce_claim=store.claim_nonce,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        stop = asyncio.Event()

        async def dispatch_loop() -> None:
            while not stop.is_set():
                try:
                    while await asyncio.to_thread(store.run_due_once):
                        pass
                    await asyncio.to_thread(store.maintain_active_once)
                except Exception:
                    LOGGER.exception("QMT execution-slice loop failed")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=resolved.poll_seconds)
                except TimeoutError:
                    continue

        task = asyncio.create_task(dispatch_loop())
        yield
        stop.set()
        await task
        await asyncio.to_thread(provider.close)

    app = FastAPI(title="QuantLab QMT Sandbox Gateway", version="1", lifespan=lifespan)
    app.state.gateway_store = store
    app.state.gateway_settings = resolved

    @app.middleware("http")
    async def verify_request(request: Request, call_next: Any) -> Any:
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "request body is too large"}, status_code=413)
        path = request.url.path
        if request.url.query:
            path += f"?{request.url.query}"
        try:
            verifier.verify(
                method=request.method,
                path=path,
                body=body,
                timestamp=request.headers.get("X-QuantLab-Timestamp", ""),
                nonce=request.headers.get("X-QuantLab-Nonce", ""),
                signature=request.headers.get("X-QuantLab-Signature", ""),
            )
        except SignatureError:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        try:
            return store.health()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="QMT sandbox is unavailable") from exc

    @app.post("/v1/orders")
    async def submit_order(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
            return store.accept_parent(payload)
        except GatewayStoreError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON order payload") from exc

    @app.get("/v1/snapshot")
    def snapshot(account_ref: str) -> dict[str, Any]:
        try:
            return store.snapshot(account_ref)
        except GatewayStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="QMT snapshot is unavailable") from exc

    @app.get("/v1/market-evidence")
    def market_evidence(instrument: str) -> dict[str, Any]:
        try:
            return store.market_evidence(instrument)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="QMT market evidence is unavailable"
            ) from exc

    return app
