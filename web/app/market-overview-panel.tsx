"use client";

import { CSSProperties, FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";

type Quote = {
  ts_code: string;
  name?: string;
  trade_date?: string;
  close?: number | null;
  pct_chg?: number | null;
  amount?: number | null;
  product?: string;
  industry?: string | null;
  asset_type?: string;
};

type Sector = {
  industry: string;
  members: number;
  pct_chg: number | null;
  amount: number | null;
  advance_ratio: number | null;
};

type Pulse = {
  trade_date: string;
  average_pct_chg: number | null;
  advance_ratio: number | null;
  amount: number | null;
};

type MarketOverview = {
  status: "ready" | "not_ready";
  message?: string;
  source: {
    snapshot_name: string | null;
    as_of: string | null;
    generated_at: string;
    is_realtime: boolean;
    freshness: "current" | "delayed" | "historical" | "unavailable";
    calendar_days_behind: number | null;
    available_datasets: string[];
  };
  breadth: {
    instruments?: number;
    advances?: number;
    declines?: number;
    unchanged?: number;
    limit_up?: number;
    limit_down?: number;
    average_pct_chg?: number | null;
    median_pct_chg?: number | null;
    amount?: number | null;
  };
  indices: Quote[];
  pulse: Pulse[];
  sectors: Sector[];
  etfs: Quote[];
  futures: Quote[];
  watchlist: Quote[];
};

type MarketOverviewPanelProps = {
  api: string;
  onOpenData: () => void;
};

const defaultWatchlist = "000300.SH,000905.SH,000852.SH,000016.SH,510300.SH,159919.SZ,510500.SH,512100.SH";

function number(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: digits }).format(value);
}

function percent(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function money(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const yuan = value * 1000;
  if (Math.abs(yuan) >= 1e12) return `${(yuan / 1e12).toFixed(2)} 万亿`;
  if (Math.abs(yuan) >= 1e8) return `${(yuan / 1e8).toFixed(1)} 亿`;
  if (Math.abs(yuan) >= 1e4) return `${(yuan / 1e4).toFixed(1)} 万`;
  return number(yuan, 0);
}

function changeClass(value: number | null | undefined) {
  return value === null || value === undefined ? "flat" : value > 0 ? "up" : value < 0 ? "down" : "flat";
}

function shortDate(value: string | null | undefined) {
  if (!value) return "—";
  const parsed = new Date(`${value}T00:00:00`);
  return Number.isNaN(parsed.getTime()) ? value : `${parsed.getMonth() + 1}/${parsed.getDate()}`;
}

export function MarketOverviewPanel({ api, onOpenData }: MarketOverviewPanelProps) {
  const [market, setMarket] = useState<MarketOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [watchlistText, setWatchlistText] = useState(defaultWatchlist);
  const [querySymbols, setQuerySymbols] = useState(defaultWatchlist);

  const refresh = useCallback(async () => {
    try {
      const response = await apiFetch(`${api}/api/market/overview?symbols=${encodeURIComponent(querySymbols)}`, { cache: "no-store" });
      if (!response.ok) throw new Error("行情聚合接口暂不可用");
      setMarket(await response.json());
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "行情聚合接口暂不可用");
    } finally {
      setLoading(false);
    }
  }, [api, querySymbols]);

  useEffect(() => {
    const initial = window.setTimeout(refresh, 0);
    const timer = window.setInterval(refresh, 30_000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
  }, [refresh]);

  const breadthTotal = (market?.breadth.advances ?? 0) + (market?.breadth.declines ?? 0) + (market?.breadth.unchanged ?? 0);
  const advanceRatio = breadthTotal ? Math.round((market?.breadth.advances ?? 0) / breadthTotal * 100) : 0;
  const pulseScale = Math.max(1, ...((market?.pulse ?? []).map((item) => Math.abs(item.average_pct_chg ?? 0))));
  const sectorScale = Math.max(1, ...((market?.sectors ?? []).map((item) => Math.abs(item.pct_chg ?? 0))));
  const leaders = useMemo(() => (market?.sectors ?? []).filter((item) => (item.pct_chg ?? 0) >= 0).slice(0, 5), [market]);
  const laggards = useMemo(() => [...(market?.sectors ?? [])].filter((item) => (item.pct_chg ?? 0) < 0).reverse().slice(0, 5), [market]);

  function applyWatchlist(event: FormEvent) {
    event.preventDefault();
    const symbols = watchlistText.split(/[,，\s]+/).map((item) => item.trim().toUpperCase()).filter(Boolean).slice(0, 30);
    setQuerySymbols(symbols.join(",") || defaultWatchlist);
  }

  if (loading && !market) {
    return <section className="market-empty"><span className="market-loading" /><h2>正在整理研究行情</h2><p>从最新不可变快照读取指数、市场宽度和资产行情。</p></section>;
  }

  if (!market || market.status !== "ready") {
    return <section className="market-empty"><span className="market-empty-mark">M</span><h2>行情面板等待数据快照</h2><p>{market?.message ?? error ?? "暂时无法读取研究行情。"}</p><button className="primary" onClick={onOpenData}>前往数据中心</button></section>;
  }

  const freshnessLabel = ({ current: "数据较新", delayed: "存在延迟", historical: "历史研究数据", unavailable: "不可用" } as const)[market.source.freshness];

  return <div className="market-page">
    <section className="market-source-bar">
      <div><span className={`market-live-dot ${market.source.freshness}`} /><strong>研究行情</strong><small>不可变快照 · 非实时行情</small></div>
      <div><span>行情日期</span><strong>{market.source.as_of ?? "—"}</strong></div>
      <div><span>数据状态</span><strong>{freshnessLabel}</strong></div>
      <div className="market-source-name"><span>数据版本</span><strong>{market.source.snapshot_name}</strong></div>
      <button onClick={refresh}>刷新行情</button>
    </section>

    {error ? <div className="notice">{error}，当前仍展示上一次成功结果。</div> : null}

    <section className="market-hero">
      <article className="breadth-card">
        <div className="market-card-heading"><div><p className="eyebrow">MARKET BREADTH</p><h2>市场温度</h2></div><span>{market.breadth.instruments ?? breadthTotal} 只股票</span></div>
        <div className="breadth-main">
          <div className="breadth-gauge" style={{ "--breadth": `${advanceRatio}%` } as CSSProperties}><strong>{advanceRatio}%</strong><span>上涨占比</span></div>
          <div className="breadth-copy"><strong className={changeClass(market.breadth.average_pct_chg)}>{percent(market.breadth.average_pct_chg)}</strong><span>全市场平均涨跌</span><small>中位数 {percent(market.breadth.median_pct_chg)}</small></div>
        </div>
        <div className="breadth-legend"><span><i className="up" />上涨 {number(market.breadth.advances, 0)}</span><span><i className="flat" />平盘 {number(market.breadth.unchanged, 0)}</span><span><i className="down" />下跌 {number(market.breadth.declines, 0)}</span></div>
      </article>

      <article className="index-board">
        <div className="market-card-heading"><div><p className="eyebrow">CORE BENCHMARKS</p><h2>主要指数</h2></div><span>{market.source.as_of}</span></div>
        <div className="index-grid">
          {market.indices.map((item) => <div className="index-tile" key={item.ts_code}><span>{item.name}</span><strong>{number(item.close)}</strong><em className={changeClass(item.pct_chg)}>{percent(item.pct_chg)}</em><small>{item.ts_code}</small></div>)}
          {!market.indices.length ? <div className="market-inline-empty">当前快照没有主要指数行情。</div> : null}
        </div>
      </article>
    </section>

    <section className="market-stat-strip">
      <article><span>两市成交额</span><strong>{money(market.breadth.amount)}</strong></article>
      <article><span>上涨家数</span><strong className="up">{number(market.breadth.advances, 0)}</strong></article>
      <article><span>下跌家数</span><strong className="down">{number(market.breadth.declines, 0)}</strong></article>
      <article><span>涨停参考</span><strong className="up">{number(market.breadth.limit_up, 0)}</strong></article>
      <article><span>跌停参考</span><strong className="down">{number(market.breadth.limit_down, 0)}</strong></article>
      <article><span>数据延迟</span><strong>{market.source.calendar_days_behind ?? "—"}<small> 天</small></strong></article>
    </section>

    <section className="market-grid">
      <article className="market-card pulse-card">
        <div className="market-card-heading"><div><h2>20 日市场脉搏</h2><p>每日全市场平均涨跌，适合观察风险偏好，不代表可交易信号。</p></div></div>
        <div className="pulse-chart" aria-label="20 日市场平均涨跌">
          {market.pulse.map((item) => {
            const value = item.average_pct_chg ?? 0;
            const height = Math.max(8, Math.abs(value) / pulseScale * 72);
            return <div className="pulse-column" key={item.trade_date} title={`${item.trade_date} ${percent(value)}`}><span className={changeClass(value)} style={{ height }} /><small>{shortDate(item.trade_date)}</small></div>;
          })}
        </div>
        <div className="pulse-foot"><span>过去 20 个交易日</span><span>绿色为上涨 · 红色为下跌</span></div>
      </article>

      <article className="market-card sector-card">
        <div className="market-card-heading"><div><h2>行业强弱</h2><p>按成分股等权平均涨跌排序。</p></div></div>
        <div className="sector-columns">
          <div><span className="sector-title">领涨</span>{leaders.map((item) => <SectorRow item={item} scale={sectorScale} key={`up-${item.industry}`} />)}</div>
          <div><span className="sector-title">承压</span>{laggards.map((item) => <SectorRow item={item} scale={sectorScale} key={`down-${item.industry}`} />)}</div>
        </div>
        {!market.sectors.length ? <div className="market-inline-empty">需要 stock_basic 行业字段才能生成行业强弱。</div> : null}
      </article>

      <article className="market-card asset-list-card">
        <div className="market-card-heading"><div><h2>活跃 ETF</h2><p>按最近交易日成交额展示。</p></div></div>
        <QuoteList items={market.etfs} empty="当前快照没有 ETF 日线。" />
      </article>

      <article className="market-card asset-list-card">
        <div className="market-card-heading"><div><h2>股指期货</h2><p>IF、IC、IM、IH 各取活跃合约。</p></div></div>
        <QuoteList items={market.futures} empty="期货数据完成后将在这里出现。" />
      </article>
    </section>

    <section className="market-card watchlist-card">
      <div className="market-card-heading"><div><h2>自选与策略观察池</h2><p>最多 30 个代码；这里只观察快照行情，不会触发交易。</p></div><form onSubmit={applyWatchlist}><input aria-label="自选代码" value={watchlistText} onChange={(event) => setWatchlistText(event.target.value)} /><button type="submit">应用</button></form></div>
      <div className="market-table-wrap"><table><thead><tr><th>标的</th><th>类型</th><th>行业</th><th>收盘价</th><th>涨跌幅</th><th>成交额</th><th>交易日</th></tr></thead><tbody>{market.watchlist.map((item) => <tr key={item.ts_code}><td><strong>{item.name ?? item.ts_code}</strong><small>{item.ts_code}</small></td><td>{({ stock: "股票", etf: "ETF", index: "指数" } as Record<string, string>)[item.asset_type ?? ""] ?? item.asset_type ?? "—"}</td><td>{item.industry ?? "—"}</td><td>{number(item.close)}</td><td><span className={changeClass(item.pct_chg)}>{percent(item.pct_chg)}</span></td><td>{money(item.amount)}</td><td>{item.trade_date ?? "—"}</td></tr>)}</tbody></table>{!market.watchlist.length ? <div className="market-inline-empty">所选代码在当前快照中没有行情。</div> : null}</div>
    </section>
  </div>;
}

function SectorRow({ item, scale }: { item: Sector; scale: number }) {
  const width = Math.max(4, Math.abs(item.pct_chg ?? 0) / scale * 100);
  return <div className="sector-row"><div><strong>{item.industry}</strong><small>{item.members} 只 · 上涨 {number(item.advance_ratio, 0)}%</small></div><div className="sector-track"><i className={changeClass(item.pct_chg)} style={{ width: `${width}%` }} /></div><span className={changeClass(item.pct_chg)}>{percent(item.pct_chg)}</span></div>;
}

function QuoteList({ items, empty }: { items: Quote[]; empty: string }) {
  if (!items.length) return <div className="market-inline-empty">{empty}</div>;
  return <div className="asset-quote-list">{items.map((item) => <div key={item.ts_code}><div><strong>{item.name ?? item.ts_code}</strong><small>{item.ts_code}</small></div><span>{number(item.close)}</span><em className={changeClass(item.pct_chg)}>{percent(item.pct_chg)}</em></div>)}</div>;
}
