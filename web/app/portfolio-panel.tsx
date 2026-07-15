"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { apiFetch } from "./api-client";

type Strategy = { id: string; versions: { id: string; status: string }[] };
type Dataset = { name: string; ready: boolean; reproducible: boolean };
type Holding = {
  instrument: string; weight: number; previous_weight: number; weight_change: number;
  action: string; reason: string;
};
type Snapshot = {
  id: string; as_of_date: string; effective_date?: string | null; status: string;
  policy_version: string; backtest_engine_version: string; cost_model: Record<string, number>;
  holdings: Holding[]; snapshot?: { changes?: Holding[]; risk_summary?: { expected_turnover: number } };
};
type RecommendationPortfolio = {
  id: string; name: string; status: string; strategy_version_id: string; dataset: string;
  hypothetical_initial_value: number; latest_snapshot?: Snapshot | null; snapshots: Snapshot[];
  hypothetical_performance: { trade_date: string; hypothetical_value: number; daily_return: number; drawdown: number }[];
};

const pct = (value?: number | null) => value == null ? "—" : `${(value * 100).toFixed(2)}%`;

export function PortfolioPanel({ api }: { api: string }) {
  const [portfolios, setPortfolios] = useState<RecommendationPortfolio[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [name, setName] = useState("沪深300 推荐组合");
  const [versionId, setVersionId] = useState("");
  const [dataset, setDataset] = useState("");
  const [asOfDate, setAsOfDate] = useState(new Date().toISOString().slice(0, 10));
  const [message, setMessage] = useState("");

  const load = useCallback(async () => {
    const [portfolioResponse, strategyResponse, datasetResponse] = await Promise.all([
      apiFetch(`${api}/api/recommendation-portfolios`, { cache: "no-store" }),
      apiFetch(`${api}/api/strategies`, { cache: "no-store" }),
      apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" }),
    ]);
    if (!portfolioResponse.ok) throw new Error("recommendations unavailable");
    const nextPortfolios = await portfolioResponse.json() as RecommendationPortfolio[];
    const nextStrategies = strategyResponse.ok ? await strategyResponse.json() as Strategy[] : [];
    const nextDatasets = datasetResponse.ok ? await datasetResponse.json() as Dataset[] : [];
    setPortfolios(nextPortfolios); setStrategies(nextStrategies); setDatasets(nextDatasets);
    if (!selectedId && nextPortfolios.length) setSelectedId(nextPortfolios[0].id);
    if (!versionId) {
      const approved = nextStrategies.flatMap((item) => item.versions).find((item) => item.status === "approved");
      if (approved) setVersionId(approved.id);
    }
    if (!dataset) {
      const ready = nextDatasets.find((item) => item.ready && item.reproducible);
      if (ready) setDataset(ready.name);
    }
  }, [api, dataset, selectedId, versionId]);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      load().catch(() => setMessage("无法读取推荐组合。"));
    }, 0);
    return () => window.clearTimeout(initial);
  }, [load]);
  const selected = portfolios.find((item) => item.id === selectedId) ?? portfolios[0];
  const snapshot = selected?.latest_snapshot;

  async function create(event: FormEvent) {
    event.preventDefault();
    const response = await apiFetch(`${api}/api/recommendation-portfolios`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, strategy_version_id: versionId, dataset, hypothetical_initial_value: 5_000_000 }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "推荐组合创建失败"); return; }
    setSelectedId(body.id); setMessage("推荐组合已创建。"); await load();
  }

  async function refresh(event: FormEvent) {
    event.preventDefault(); if (!selected) return;
    const response = await apiFetch(`${api}/api/recommendation-portfolios/${selected.id}/refresh`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ as_of_date: asOfDate }),
    });
    const body = await response.json();
    setMessage(response.ok ? "推荐刷新任务已进入队列。" : body.detail ?? "推荐刷新失败");
    await load();
  }

  return <div className="portfolio-page">
    {message && <div className="notice">{message}</div>}
    <section className="portfolio-hero">
      <form className="portfolio-launcher" onSubmit={create}>
        <div className="card-heading"><div><span>RECOMMENDATION TRACKING</span><strong>创建推荐组合</strong></div><span className="status-chip verified">RESEARCH ONLY</span></div>
        <label>名称<input value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label>已审批v2策略<select value={versionId} onChange={(event) => setVersionId(event.target.value)}>{strategies.flatMap((item) => item.versions).filter((item) => item.status === "approved").map((item) => <option key={item.id} value={item.id}>{item.id.slice(0, 16)}</option>)}</select></label>
        <label>Qlib数据集<select value={dataset} onChange={(event) => setDataset(event.target.value)}>{datasets.filter((item) => item.ready && item.reproducible).map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
        <button className="primary" disabled={!versionId || !dataset || name.length < 3}>创建推荐组合</button>
      </form>
      <article className="portfolio-summary">
        <div className="card-heading"><div><span>当前组合</span><strong>{selected?.name ?? "尚无推荐组合"}</strong></div><span className="status-chip">{selected?.status ?? "empty"}</span></div>
        <label>组合<select value={selected?.id ?? ""} onChange={(event) => setSelectedId(event.target.value)}>{portfolios.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        <form className="rebalance-form" onSubmit={refresh}><label>信号日期<input type="date" value={asOfDate} onChange={(event) => setAsOfDate(event.target.value)} /></label><button className="primary" disabled={!selected || !asOfDate}>刷新推荐</button></form>
        <small>只生成研究建议，不产生订单、成交或券商指令。</small>
      </article>
    </section>
    <section className="metric-strip portfolio-metrics">
      <div><span>建议生效日</span><strong>{snapshot?.effective_date ?? "—"}</strong></div>
      <div><span>推荐股票</span><strong>{snapshot?.holdings.length ?? 0}</strong></div>
      <div><span>建议换手</span><strong>{pct(snapshot?.snapshot?.risk_summary?.expected_turnover)}</strong></div>
      <div><span>Policy</span><strong>{snapshot?.policy_version ?? "—"}</strong></div>
      <div><span>Qlib引擎</span><strong>{snapshot?.backtest_engine_version ?? "—"}</strong></div>
    </section>
    <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">TARGET WEIGHTS</p><h2>推荐股票与建议调整</h2></div><span>{snapshot?.status ?? "尚未刷新"}</span></div>
      <div className="table-wrap"><table className="portfolio-table"><thead><tr><th>证券</th><th>目标权重</th><th>原权重</th><th>变化</th><th>建议</th><th>原因</th></tr></thead><tbody>{snapshot?.holdings.map((item) => <tr key={item.instrument}><td><code>{item.instrument}</code></td><td>{pct(item.weight)}</td><td>{pct(item.previous_weight)}</td><td>{pct(item.weight_change)}</td><td>{({ increase: "增加", decrease: "减少", hold: "维持" } as Record<string, string>)[item.action] ?? item.action}</td><td>{item.reason}</td></tr>)}</tbody></table>{!snapshot?.holdings.length && <div className="empty">尚无推荐快照。</div>}</div>
    </section>
    {snapshot && <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">COST ASSUMPTIONS</p><h2>统一成本假设</h2></div></div><div className="metric-strip portfolio-metrics"><div><span>买入费率</span><strong>{pct(snapshot.cost_model.buy_commission_rate)}</strong></div><div><span>卖出费率</span><strong>{pct(snapshot.cost_model.sell_commission_rate)}</strong></div><div><span>固定滑点</span><strong>{pct(snapshot.cost_model.fixed_slippage_rate)}</strong></div><div><span>参与率上限</span><strong>{pct(snapshot.cost_model.max_volume_participation)}</strong></div><div><span>最低佣金</span><strong>¥{snapshot.cost_model.min_commission}</strong></div></div></section>}
  </div>;
}
