"""Governed GICS sector-leader baskets for global-reference factors.

The production probe (2026-09-02) confirmed no sector index coverage in
index_global and no industry classification column in us_basic, so sector
transmission is proxied by equal-weight baskets of well-known US-listed
leaders built from us_daily.  This module is the single governed mapping:
versioned, hash-bound into factor manifests, and deliberately explicit — one
comment per sector noting the selection rationale.  Any membership change is
a semantic change and must bump SECTOR_LEADER_MAP_VERSION.

GICS 一级 11 行业,每行业 3–5 只广为人知的美股上市龙头(裸 ticker,与
us_daily 的 ts_code 约定一致;TSM 为ADR)。
"""

from __future__ import annotations

import hashlib
import json

SECTOR_LEADER_MAP_VERSION = "gics-sector-leaders.v1"

# slug -> (GICS 一级行业中文名, 龙头成员)
SECTOR_LEADERS: dict[str, tuple[str, tuple[str, ...]]] = {
    # 信息技术:全球算力/软件/晶圆代工龙头(原半导体篮子并入本行业)
    "information_technology": (
        "信息技术",
        ("NVDA", "AAPL", "MSFT", "AVGO", "TSM"),
    ),
    # 通信服务:搜索/社交/流媒体/移动运营商龙头
    "communication_services": ("通信服务", ("GOOGL", "META", "NFLX", "TMUS")),
    # 可选消费:电商/电动车/家装/餐饮龙头
    "consumer_discretionary": ("可选消费", ("AMZN", "TSLA", "HD", "MCD")),
    # 必需消费:零售/日化/饮料龙头
    "consumer_staples": ("必需消费", ("WMT", "PG", "KO", "PEP", "COST")),
    # 能源:一体化油气与油服龙头
    "energy": ("能源", ("XOM", "CVX", "COP", "SLB")),
    # 金融:银行与支付网络龙头
    "financials": ("金融", ("JPM", "V", "MA", "BAC")),
    # 医疗:保险/制药/创新药龙头
    "health_care": ("医疗", ("UNH", "JNJ", "LLY", "ABBV")),
    # 工业:航空发动机/工程机械/自动化/物流龙头
    "industrials": ("工业", ("GE", "CAT", "HON", "UPS")),
    # 材料:工业气体/涂料/铜矿龙头
    "materials": ("材料", ("LIN", "SHW", "FCX")),
    # 房地产:物流/通信铁塔/数据中心 REIT 龙头
    "real_estate": ("房地产", ("PLD", "AMT", "EQIX")),
    # 公用事业:新能源电力与 regulated 电力龙头
    "utilities": ("公用事业", ("NEE", "SO", "DUK")),
}

SECTOR_SLUGS = tuple(SECTOR_LEADERS)


def sector_factor_name(slug: str) -> str:
    """Factor name for one sector slug; fails closed on unknown sectors."""

    if slug not in SECTOR_LEADERS:
        raise ValueError(
            f"unknown GICS sector slug {slug!r}; expected one of {list(SECTOR_SLUGS)}"
        )
    return f"global_ref_sector_{slug}_ret"


def sector_leader_map_identity() -> dict[str, str]:
    """Canonical identity of the governed map: version plus content sha256."""

    canonical = {
        "version": SECTOR_LEADER_MAP_VERSION,
        "sectors": {
            slug: {"label": label, "members": list(members)}
            for slug, (label, members) in SECTOR_LEADERS.items()
        },
    }
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "version": SECTOR_LEADER_MAP_VERSION,
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    }
