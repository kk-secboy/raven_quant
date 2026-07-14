from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any
from urllib.parse import quote

import requests


def signed_get(base_url: str, path: str, secret: str, timeout: float) -> dict[str, Any]:
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16)
    message = f"{timestamp}.{nonce}.GET.{path}.".encode()
    signature = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    response = requests.get(
        f"{base_url.rstrip('/')}{path}",
        headers={
            "X-QuantLab-Timestamp": timestamp,
            "X-QuantLab-Nonce": nonce,
            "X-QuantLab-Signature": signature,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"gateway returned a non-object for {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only acceptance checks for a running QMT sandbox gateway"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8790")
    parser.add_argument("--account-ref", required=True)
    parser.add_argument("--instrument", default="SH600000")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    secret = os.getenv("BROKER_HMAC_SECRET", "").strip()
    if len(secret) < 32:
        raise SystemExit("BROKER_HMAC_SECRET must be exported before running acceptance")
    paths = {
        "health": "/v1/health",
        "snapshot": f"/v1/snapshot?account_ref={quote(args.account_ref, safe='')}",
        "market_evidence": (
            f"/v1/market-evidence?instrument={quote(args.instrument, safe='')}"
        ),
    }
    results = {
        name: signed_get(args.url, path, secret, args.timeout) for name, path in paths.items()
    }
    for name, payload in results.items():
        if payload.get("status") != "ok" or payload.get("environment") != "sandbox":
            raise SystemExit(f"{name} did not attest sandbox readiness")
    if results["snapshot"].get("account_ref") != args.account_ref:
        raise SystemExit("snapshot account reference mismatch")
    if results["market_evidence"].get("instrument") != args.instrument:
        raise SystemExit("market-evidence instrument mismatch")
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
