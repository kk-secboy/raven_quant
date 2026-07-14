from __future__ import annotations

import argparse
import getpass
import os
import tempfile
from datetime import date
from pathlib import Path

import requests

OFFICIAL_API_URL = "https://api.tushare.pro"


def validate_token(api_url: str, token: str, timeout: float) -> None:
    today = date.today().strftime("%Y%m%d")
    response = requests.post(
        api_url,
        json={
            "api_name": "trade_cal",
            "token": token,
            "params": {"exchange": "SSE", "start_date": today, "end_date": today},
            "fields": "exchange,cal_date,is_open,pretrade_date",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("code") != 0:
        message = payload.get("msg") if isinstance(payload, dict) else "invalid response"
        raise RuntimeError(f"Tushare rejected the credential: {message}")


def update_env(env_file: Path, api_url: str, token: str) -> None:
    original = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    lines = original.splitlines()
    replacements = {"TUSHARE_API_URL": api_url, "TUSHARE_TOKEN": token}
    found: set[str] = set()
    output: list[str] = []
    for line in lines:
        key = (
            line.split("=", 1)[0].strip()
            if "=" in line and not line.lstrip().startswith("#")
            else ""
        )
        if key in replacements:
            output.append(f"{key}={replacements[key]}")
            found.add(key)
        else:
            output.append(line)
    for key, value in replacements.items():
        if key not in found:
            output.append(f"{key}={value}")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{env_file.name}.", dir=env_file.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(output).rstrip() + "\n")
        os.replace(temporary, env_file)
        try:
            env_file.chmod(0o600)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a Tushare token and store it without echoing the secret"
    )
    parser.add_argument("--env-file", type=Path, default=Path("deploy/.env"))
    parser.add_argument("--api-url", default=OFFICIAL_API_URL)
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args()
    token = getpass.getpass("Tushare token (input is hidden): ").strip()
    if len(token) < 8 or any(character.isspace() for character in token):
        raise SystemExit("token format is invalid")
    validate_token(args.api_url, token, args.timeout)
    update_env(args.env_file.resolve(), args.api_url, token)
    print(f"Tushare credential validated and stored in {args.env_file.resolve()}")


if __name__ == "__main__":
    main()
