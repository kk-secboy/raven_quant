from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

SHANGHAI_TZ = timezone(timedelta(hours=8))
EXPECTED_LEVEL_COUNTS = {"L1": 31, "L2": 134, "L3": 346}
MEMBER_FIELDS = (
    "l1_code",
    "l1_name",
    "l2_code",
    "l2_name",
    "l3_code",
    "l3_name",
    "ts_code",
    "name",
    "in_date",
    "out_date",
    "is_new",
)
FOLDER_ALIASES = {
    "850817.SI": "集成电路封测_封测",
    "850822.SI": "印制电路板_PCB",
}


class RelayError(RuntimeError):
    pass


def safe_component(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value.strip())
    cleaned = cleaned.rstrip(" .")
    return cleaned or "UNNAMED"


def canonical_sw_code(value: Any) -> str:
    code = str(value or "").strip().upper()
    if code and "." not in code:
        code = f"{code}.SI"
    return code


def folder_code(value: str) -> str:
    return canonical_sw_code(value).removesuffix(".SI")


def normalize_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    fields = data.get("fields") or data.get("columns") or []
    items = data.get("items") or data.get("rows") or []
    if not isinstance(fields, list) or not isinstance(items, list):
        return []
    return [
        {
            str(field): item[index] if index < len(item) else None
            for index, field in enumerate(fields)
        }
        for item in items
        if isinstance(item, list)
    ]


class RelayClient:
    def __init__(
        self,
        api_url: str,
        token: str,
        *,
        pause_seconds: float,
        timeout_seconds: float,
        max_attempts: int,
    ) -> None:
        self.api_url = api_url
        self.token = token
        self.pause_seconds = pause_seconds
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.session = requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "Accept-Encoding": "gzip"}
        )

    def fetch(self, api_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        body = {"api_name": api_name, "token": self.token, "params": params}
        last_message = "unknown relay failure"
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.post(
                    self.api_url,
                    json=body,
                    timeout=(10.0, self.timeout_seconds),
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                last_message = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_attempts:
                    break
                time.sleep(min(30.0, 2.0**attempt))
                continue

            code = payload.get("code") if isinstance(payload, dict) else None
            if str(code) == "0":
                rows = normalize_rows(payload)
                time.sleep(self.pause_seconds)
                return rows

            message = str(payload.get("msg") or "relay rejected request")
            last_message = f"api={api_name} code={code} message={message}"
            wait_match = re.search(r"(\d+)\s*秒", message)
            retryable = str(code) in {"403", "429", "500", "502", "503", "504"}
            if not retryable or attempt >= self.max_attempts:
                break
            wait_seconds = (
                int(wait_match.group(1)) + 2
                if wait_match
                else (55 if str(code) == "403" and "IP" in message else min(30, 2**attempt))
            )
            print(
                f"relay temporarily unavailable for {api_name}; retrying in {wait_seconds}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait_seconds)
        raise RelayError(last_message)


def newest_members(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for raw in rows:
        row = {field: str(raw.get(field) or "").strip() for field in MEMBER_FIELDS}
        row["l1_code"] = canonical_sw_code(row["l1_code"])
        row["l2_code"] = canonical_sw_code(row["l2_code"])
        row["l3_code"] = canonical_sw_code(row["l3_code"])
        row["ts_code"] = row["ts_code"].upper()
        if not row["l3_code"] or not row["ts_code"]:
            continue
        if row["is_new"] and row["is_new"].upper() != "Y":
            continue
        key = (row["l3_code"], row["ts_code"])
        previous = selected.get(key)
        if previous is None or row["in_date"] > previous["in_date"]:
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (row["l3_code"], row["ts_code"]))


def taxonomy_rows(classifications: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    l1_by_industry = {
        str(row.get("industry_code") or "").strip(): row for row in classifications["L1"]
    }
    l2_by_industry = {
        str(row.get("industry_code") or "").strip(): row for row in classifications["L2"]
    }
    output: list[dict[str, str]] = []
    for raw in classifications["L3"]:
        l3_industry_code = str(raw.get("industry_code") or "").strip()
        l2_industry_code = str(raw.get("parent_code") or "").strip()
        l2 = l2_by_industry.get(l2_industry_code, {})
        l1_industry_code = str(l2.get("parent_code") or "").strip()
        if not l1_industry_code and len(l3_industry_code) >= 2:
            l1_industry_code = f"{l3_industry_code[:2]}0000"
        l1 = l1_by_industry.get(l1_industry_code, {})
        l1_code = canonical_sw_code(l1.get("index_code"))
        l2_code = canonical_sw_code(l2.get("index_code"))
        l3_code = canonical_sw_code(raw.get("index_code"))
        if not all((l1_code, l2_code, l3_code)):
            raise ValueError(f"incomplete taxonomy ancestry for {l3_industry_code}")
        output.append(
            {
                "l1_industry_code": l1_industry_code,
                "l1_code": l1_code,
                "l1_name": str(l1.get("industry_name") or "").strip(),
                "l2_industry_code": l2_industry_code,
                "l2_code": l2_code,
                "l2_name": str(l2.get("industry_name") or "").strip(),
                "l3_industry_code": l3_industry_code,
                "l3_code": l3_code,
                "l3_name": str(raw.get("industry_name") or "").strip(),
                "is_pub": str(raw.get("is_pub") or "").strip(),
                "src": str(raw.get("src") or "SW2021").strip(),
            }
        )
    return sorted(output, key=lambda row: row["l3_industry_code"])


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stock_markdown(row: dict[str, str], generated_at: str) -> str:
    symbol, _, exchange = row["ts_code"].partition(".")

    def quoted(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    return (
        "---\n"
        "schema_version: 1\n"
        f"as_of: {quoted(generated_at[:10])}\n"
        f"ts_code: {quoted(row['ts_code'])}\n"
        f"symbol: {quoted(symbol)}\n"
        f"exchange: {quoted(exchange)}\n"
        f"name: {quoted(row['name'])}\n"
        'classification_kind: "sw2021_l3"\n'
        f"l1_code: {quoted(row['l1_code'])}\n"
        f"l1_name: {quoted(row['l1_name'])}\n"
        f"official_l2_code: {quoted(row['l2_code'])}\n"
        f"official_l2_name: {quoted(row['l2_name'])}\n"
        f"official_l3_code: {quoted(row['l3_code'])}\n"
        f"official_l3_name: {quoted(row['l3_name'])}\n"
        f"in_date: {quoted(row['in_date']) if row['in_date'] else 'null'}\n"
        f"out_date: {quoted(row['out_date']) if row['out_date'] else 'null'}\n"
        "is_current: true\n"
        'source_api: "index_member_all"\n'
        "---\n\n"
        f"# {row['name']}（{row['ts_code']}）\n\n"
        f"当前属于 **{row['l1_name']} / {row['l2_name']} / {row['l3_name']}**。\n"
    )


def build_tree(
    root: Path,
    taxonomy: list[dict[str, str]],
    members: list[dict[str, str]],
    listed_stocks: dict[str, dict[str, str]],
    *,
    api_url: str,
    replace: bool,
) -> dict[str, Any]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "industries"
    if target.exists() and not replace:
        raise FileExistsError(f"{target} already exists; pass --replace to rebuild")

    generated_at = datetime.now(SHANGHAI_TZ).isoformat(timespec="seconds")
    stage = root / f".industry-tree-staging-{uuid.uuid4().hex}"
    stage_industries = stage / "industries"
    stage_industries.mkdir(parents=True)

    taxonomy_by_l3 = {row["l3_code"]: row for row in taxonomy}
    industry_paths: dict[str, Path] = {}
    for row in taxonomy:
        l1_folder = f"{row['l1_industry_code']}_{safe_component(row['l1_name'])}"
        l3_name = FOLDER_ALIASES.get(row["l3_code"], row["l3_name"])
        l3_folder = f"{row['l3_industry_code']}_{safe_component(l3_name)}"
        path = stage_industries / l1_folder / l3_folder
        path.mkdir(parents=True, exist_ok=True)
        industry_paths[row["l3_code"]] = path

    electronic = next(
        (
            path.parent
            for code, path in industry_paths.items()
            if taxonomy_by_l3[code]["l1_name"] == "电子"
        ),
        None,
    )
    if electronic is not None:
        mlcc = electronic / "z001_custom_MLCC"
        mlcc.mkdir(exist_ok=True)
        (mlcc / "README.md").write_text(
            "# MLCC（自定义赛道）\n\n"
            "MLCC 不是申万 SW2021 独立行业，申万仅提供“被动元件”分类。\n"
            "本目录暂不复制整个被动元件板块，待取得公司产品级证据后再生成股票文件。\n",
            encoding="utf-8",
        )

    enriched_members: list[dict[str, str]] = []
    missing_taxonomy: list[str] = []
    for member in members:
        taxon = taxonomy_by_l3.get(member["l3_code"])
        path = industry_paths.get(member["l3_code"])
        if taxon is None or path is None:
            missing_taxonomy.append(member["l3_code"])
            continue
        merged = dict(member)
        for key in ("l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name"):
            if not merged.get(key):
                merged[key] = taxon[key]
        listing = listed_stocks.get(merged["ts_code"])
        if listing is None:
            continue
        if listing.get("name"):
            merged["name"] = listing["name"]
        stock_path = path / f"{safe_component(merged['ts_code'])}.md"
        stock_path.write_text(stock_markdown(merged, generated_at), encoding="utf-8")
        enriched_members.append(merged)

    if missing_taxonomy:
        missing = ", ".join(sorted(set(missing_taxonomy)))
        raise ValueError(f"members reference unknown L3 classifications: {missing}")

    per_industry = Counter(row["l3_code"] for row in enriched_members)
    unique_stocks = {row["ts_code"] for row in enriched_members}
    stock_to_l3: dict[str, set[str]] = defaultdict(set)
    for row in enriched_members:
        stock_to_l3[row["ts_code"]].add(row["l3_code"])
    multi_classified = {
        code: sorted(levels) for code, levels in stock_to_l3.items() if len(levels) > 1
    }
    empty_industries = [row["l3_code"] for row in taxonomy if per_industry[row["l3_code"]] == 0]

    for row in taxonomy:
        member_codes = sorted(
            item["ts_code"] for item in enriched_members if item["l3_code"] == row["l3_code"]
        )
        segment = {
            "schema_version": 1,
            "classification_kind": "sw2021_l3",
            "l1_industry_code": row["l1_industry_code"],
            "l1_code": row["l1_code"],
            "l1_name": row["l1_name"],
            "official_l2_industry_code": row["l2_industry_code"],
            "official_l2_code": row["l2_code"],
            "official_l2_name": row["l2_name"],
            "segment_industry_code": row["l3_industry_code"],
            "segment_code": row["l3_code"],
            "segment_name": row["l3_name"],
            "is_published_index": row["is_pub"] == "1",
            "member_count": len(member_codes),
            "members_sha256": hashlib.sha256("\n".join(member_codes).encode()).hexdigest(),
        }
        (industry_paths[row["l3_code"]] / "_segment.json").write_text(
            json.dumps(segment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    taxonomy_by_l1: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in taxonomy:
        taxonomy_by_l1[row["l1_code"]].append(row)
    for rows in taxonomy_by_l1.values():
        first = rows[0]
        l1_path = industry_paths[first["l3_code"]].parent
        industry = {
            "schema_version": 1,
            "classification_kind": "sw2021_l1",
            "industry_code": first["l1_industry_code"],
            "index_code": first["l1_code"],
            "name": first["l1_name"],
            "segment_count": len(rows),
            "member_file_count": sum(per_industry[row["l3_code"]] for row in rows),
        }
        (l1_path / "_industry.json").write_text(
            json.dumps(industry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    manifest: dict[str, Any] = {
        "generated_at": generated_at,
        "taxonomy": "SW2021",
        "source_api": "index_classify + index_member_all",
        "source_url": api_url,
        "membership_scope": "current only (is_new=Y)",
        "listing_scope": "currently listed only (stock_basic list_status=L)",
        "l1_count": len({row["l1_code"] for row in taxonomy}),
        "l2_count": len({row["l2_code"] for row in taxonomy}),
        "l3_count": len(taxonomy),
        "membership_file_count": len(enriched_members),
        "unique_stock_count": len(unique_stocks),
        "empty_l3_count": len(empty_industries),
        "empty_l3_codes": empty_industries,
        "multi_classified_stock_count": len(multi_classified),
        "multi_classified_stocks": multi_classified,
        "custom_tracks": {
            "MLCC": {
                "status": "pending_product_level_evidence",
                "official_parent": "850823.SI 被动元件",
                "stock_files": 0,
            }
        },
    }

    write_csv(
        stage / "taxonomy.csv",
        taxonomy,
        (
            "l1_industry_code",
            "l1_code",
            "l1_name",
            "l2_industry_code",
            "l2_code",
            "l2_name",
            "l3_industry_code",
            "l3_code",
            "l3_name",
            "is_pub",
            "src",
        ),
    )
    write_csv(stage / "memberships.csv", enriched_members, MEMBER_FIELDS)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (stage / "manifest.json").write_text(
        manifest_text, encoding="utf-8"
    )
    (stage_industries / "_manifest.json").write_text(
        manifest_text, encoding="utf-8"
    )
    (stage / "README.md").write_text(
        "# 申万行业股票目录\n\n"
        "目录结构为 `一级行业/细分行业/股票代码.md`。细分行业使用申万 SW2021 "
        "三级分类扁平化为第二层；股票文件仅表示当前有效行业成分，不表示投资建议。\n\n"
        f"生成时间：{generated_at}\n\n"
        f"- 一级行业：{manifest['l1_count']}\n"
        f"- 二级行业（文件元数据）：{manifest['l2_count']}\n"
        f"- 细分行业目录：{manifest['l3_count']}\n"
        f"- 行业成员文件：{manifest['membership_file_count']}\n"
        f"- 不重复股票：{manifest['unique_stock_count']}\n"
        f"- 无当前成分的细分行业：{manifest['empty_l3_count']}\n\n"
        "MLCC 为自定义产品赛道，未直接复制申万“被动元件”成分；详见电子行业下的 "
        "`z001_custom_MLCC/README.md`。\n",
        encoding="utf-8",
    )

    if target.exists():
        resolved_target = target.resolve()
        if resolved_target.parent != root or resolved_target.name != "industries":
            raise RuntimeError(f"refusing to replace unexpected path: {resolved_target}")
        shutil.rmtree(resolved_target)
    os.replace(stage_industries, target)
    for filename in ("taxonomy.csv", "memberships.csv", "manifest.json", "README.md"):
        os.replace(stage / filename, root / filename)
    stage.rmdir()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a SW2021 L1/L3 directory tree with one Markdown file "
            "per current stock member."
        )
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent, help="output root"
    )
    parser.add_argument("--api-url", default=os.getenv("TUSHARE_API_URL", ""))
    parser.add_argument("--pause", type=float, default=0.75)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    api_url = str(args.api_url).strip()
    if not api_url or not token:
        raise SystemExit("TUSHARE_API_URL and TUSHARE_TOKEN must be provided via the environment")
    client = RelayClient(
        api_url,
        token,
        pause_seconds=max(0.0, args.pause),
        timeout_seconds=max(1.0, args.timeout),
        max_attempts=max(1, args.max_attempts),
    )

    classifications: dict[str, list[dict[str, Any]]] = {}
    for level, expected in EXPECTED_LEVEL_COUNTS.items():
        rows = client.fetch("index_classify", {"level": level, "src": "SW2021"})
        if len(rows) != expected:
            raise ValueError(f"index_classify {level}: expected {expected} rows, got {len(rows)}")
        classifications[level] = rows
        print(f"index_classify {level}: {len(rows)} rows", flush=True)

    taxonomy = taxonomy_rows(classifications)
    l1_codes = sorted({row["l1_code"] for row in taxonomy})
    all_members: list[dict[str, Any]] = []
    for l1_code in l1_codes:
        rows = client.fetch("index_member_all", {"l1_code": l1_code, "is_new": "Y"})
        if len(rows) >= 2_000:
            raise ValueError(f"index_member_all {l1_code} hit the 2000-row cap")
        all_members.extend(rows)
        print(f"index_member_all {l1_code}: {len(rows)} rows", flush=True)

    members = newest_members(all_members)
    listed_rows = client.fetch("stock_basic", {"list_status": "L"})
    if len(listed_rows) >= 6_000:
        raise ValueError("stock_basic list_status=L hit the 6000-row cap")
    listed_stocks = {
        str(row.get("ts_code") or "").strip().upper(): {
            "name": str(row.get("name") or "").strip(),
            "list_status": str(row.get("list_status") or "L").strip().upper(),
        }
        for row in listed_rows
        if str(row.get("ts_code") or "").strip()
    }
    print(f"stock_basic listed: {len(listed_stocks)} rows", flush=True)
    manifest = build_tree(
        args.root,
        taxonomy,
        members,
        listed_stocks,
        api_url=api_url,
        replace=args.replace,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
