"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";
import { StrategyAllocationPanel } from "./strategy-allocation-panel";

type Dataset = { name: string; ready: boolean; start_date: string; end_date: string; lineage_verified?: boolean };
type StrategyVersion = { id: string; version: number; status: string; config?: Record<string, number | string> };
type Strategy = { id: string; name: string; versions: StrategyVersion[] };
type Position = { instrument: string; industry?: string | null; take_profit_stage: number; quantity: number; market_price: number; market_value: number; weight: number; realized_pnl: number; unrealized_pnl: number };
type NavRow = { trade_date: string; nav: number; daily_return: number; drawdown: number; turnover: number; fees: number };
type Batch = { id: string; as_of_date: string; trade_date?: string | null; status: string; error?: string | null };
type Order = { id: string; instrument: string; side: string; requested_quantity: number; target_weight: number; status: string; reason?: string | null };
type RiskEvent = {
  id: number; rule: string; severity: string; event_type: string; observed?: number;
  limit_value?: number; status: string; created_at: string; details?: Record<string, unknown>;
  acknowledged_by?: string | null; resolved_by?: string | null; resolution_reason?: string | null;
};
type Contributor = { instrument: string; pnl: number };
type Review = { id: string; trade_date: string; status: string; summary: {
  ending_nav: number; net_pnl: number; daily_return: number; active_return?: number | null;
  drawdown: number; exposure: number; turnover: number; fees: number; fill_rate: number;
  requested_orders: number; filled_orders: number; rejected_orders: number;
  risk_event_count: number; critical_risk_count: number; next_portfolio_status: string;
  best_contributors: Contributor[]; worst_contributors: Contributor[];
  rejections: { instrument: string; reason: string }[]; risk_rules: string[];
} };
type Portfolio = {
  id: string; name: string; strategy_version_id: string; dataset: string; status: string;
  dataset_roll_policy: "pinned" | "latest_compatible";
  initial_cash: number; cash: number; nav: number; high_water_mark: number;
  positions: Position[]; nav_history: NavRow[]; batches: Batch[]; orders: Order[]; risk_events: RiskEvent[]; reviews: Review[];
};

const money = (value: number) => new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 0 }).format(value);
const pct = (value: number | undefined) => typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";

export function PortfolioPanel({ api }: { api: string }) {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [versionId, setVersionId] = useState("");
  const [dataset, setDataset] = useState("");
  const [name, setName] = useState("沪深300 模拟组合");
  const [initialCash, setInitialCash] = useState(5_000_000);
  const [autoRollDataset, setAutoRollDataset] = useState(true);
  const [signalDate, setSignalDate] = useState("");
  const [rebalanceSlippage, setRebalanceSlippage] = useState(0.0005);
  const [riskResolution, setRiskResolution] = useState("");
  const [message, setMessage] = useState("");

  async function load() {
    try {
      const responses = await Promise.all([
        apiFetch(`${api}/api/strategies`, { cache: "no-store" }),
        apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" }),
        apiFetch(`${api}/api/portfolios`, { cache: "no-store" }),
      ]);
      if (responses.some((response) => !response.ok)) throw new Error("API unavailable");
      const nextStrategies: Strategy[] = await responses[0].json();
      const nextDatasets: Dataset[] = await responses[1].json();
      const nextPortfolios: Portfolio[] = await responses[2].json();
      const approved = nextStrategies.flatMap((strategy) => strategy.versions.filter((version) => version.status === "approved"));
      setStrategies(nextStrategies); setDatasets(nextDatasets); setPortfolios(nextPortfolios);
      if (!versionId && approved.length) setVersionId(approved[0].id);
      if (!dataset && nextDatasets.length) {
        setDataset(nextDatasets[0].name);
        if (!nextDatasets[0].lineage_verified) setAutoRollDataset(false);
      }
      if (!signalDate && nextDatasets.length) setSignalDate(nextDatasets[0].end_date);
      if (!selectedId && nextPortfolios.length) setSelectedId(nextPortfolios[0].id);
    } catch { setMessage("无法读取模拟组合控制面，请确认 Python 后端正在运行。"); }
  }

  useEffect(() => {
    const initial = window.setTimeout(load, 0); const timer = window.setInterval(load, 8000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const approvedVersions = useMemo(() => strategies.flatMap((strategy) => strategy.versions
    .filter((version) => version.status === "approved")
    .map((version) => ({ strategy, version }))), [strategies]);
  const selected = portfolios.find((item) => item.id === selectedId) ?? portfolios[0];
  const selectedVersion = approvedVersions.find(({ version }) => version.id === selected?.strategy_version_id)?.version;
  const latestNav = selected?.nav_history[0];
  const latestReview = selected?.reviews[0];
  const activeBatch = selected?.batches.some((item) => ["queued", "running"].includes(item.status));
  const riskPending = selected ? ["liquidation_pending", "risk_reduction_pending"].includes(selected.status) : false;

  async function createPortfolio(event: FormEvent) {
    event.preventDefault();
    const response = await apiFetch(`${api}/api/portfolios`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name, strategy_version_id: versionId, dataset, initial_cash: initialCash,
        dataset_roll_policy: autoRollDataset && datasets.find((item) => item.name === dataset)?.lineage_verified ? "latest_compatible" : "pinned",
        actor: "local-operator",
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "模拟组合创建失败"); return; }
    setMessage(`模拟组合 ${body.name} 已创建，只允许使用已审批策略。`); setSelectedId(body.id); await load();
  }

  async function rebalance(event: FormEvent) {
    event.preventDefault(); if (!selected) return;
    const response = await apiFetch(`${api}/api/portfolios/${selected.id}/rebalance`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ as_of_date: signalDate, slippage: rebalanceSlippage }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "日终再平衡创建失败"); return; }
    setMessage(`信号批次 ${body.id.slice(0, 8)} 已进入 Qlib Worker；同一信号日重复提交不会重复记账。`); await load();
  }

  async function changeStatus(status: "active" | "paused") {
    if (!selected) return;
    const response = await apiFetch(`${api}/api/portfolios/${selected.id}/status`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "组合状态更新失败"); return; }
    setMessage(status === "paused" ? "组合已暂停，不再接受新的日终批次。" : "组合已恢复为可运行状态。"); await load();
  }

  async function actOnRisk(item: RiskEvent, action: "acknowledge" | "resolve") {
    if (!selected) return;
    const response = await apiFetch(`${api}/api/portfolios/${selected.id}/risk-events/${item.id}/${action}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        actor: "local-operator",
        ...(action === "resolve" ? { reason: riskResolution } : {}),
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "风险事件处置失败"); return; }
    setMessage(action === "acknowledge" ? "风险事件已确认并记录责任人。" : "风险事件已完成处置；恢复组合仍需单独操作。");
    if (action === "resolve") setRiskResolution("");
    await load();
  }

  return <>
    {message && <div className="notice">{message}</div>}
    <StrategyAllocationPanel api={api} strategies={strategies} datasets={datasets} />
    <section className="portfolio-hero">
      <form className="portfolio-launcher" onSubmit={createPortfolio}>
        <div className="card-heading"><div><span>受治理模拟组合</span><strong>从已审批策略开始</strong></div><span className="status-chip verified">POSTGRESQL LEDGER</span></div>
        <label>组合名称<input value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label>已审批策略<select value={versionId} onChange={(event) => setVersionId(event.target.value)}>{approvedVersions.length ? approvedVersions.map(({ strategy, version }) => <option key={version.id} value={version.id}>{strategy.name} · v{version.version}</option>) : <option value="">尚无已审批策略</option>}</select></label>
        <label>Qlib 快照<select value={dataset} onChange={(event) => { const value = event.target.value; setDataset(value); const item = datasets.find((candidate) => candidate.name === value); if (item) { setSignalDate(item.end_date); if (!item.lineage_verified) setAutoRollDataset(false); } }}>{datasets.map((item) => <option key={item.name} value={item.name}>{item.name} · 截至 {item.end_date}{item.lineage_verified ? " · 已验证血缘" : " · 固定快照"}</option>)}</select></label>
        <label className="policy-toggle"><input type="checkbox" checked={autoRollDataset} disabled={!datasets.find((item) => item.name === dataset)?.lineage_verified} onChange={(event) => setAutoRollDataset(event.target.checked)} /><span>自动推进到同血缘、追加式验证的新 Qlib 快照</span></label>
        <label>初始资金<input type="number" min="100000" step="100000" value={initialCash} onChange={(event) => setInitialCash(Number(event.target.value))} /></label>
        <button className="primary" disabled={!versionId || !dataset || name.length < 3}>创建模拟组合</button>
      </form>
      <article className="portfolio-summary">
        <div className="card-heading"><div><span>组合账本</span><strong>{selected?.name ?? "等待创建组合"}</strong></div>{selected && <span className={`state ${selected.status === "active" ? "ready" : "partial"}`}>{selected.status}</span>}</div>
        <label>当前组合<select value={selected?.id ?? ""} onChange={(event) => setSelectedId(event.target.value)}>{portfolios.length ? portfolios.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.status}</option>) : <option value="">尚无组合</option>}</select></label>
        {selected && <small className="muted">数据策略：{selected.dataset_roll_policy === "latest_compatible" ? "同血缘自动推进" : "固定不可变快照"}</small>}
        <div className="portfolio-kpis"><div><span>净值</span><strong>{money(selected?.nav ?? 0)}</strong></div><div><span>现金</span><strong>{money(selected?.cash ?? 0)}</strong></div><div><span>持仓</span><strong>{selected?.positions.length ?? 0}</strong></div><div><span>回撤</span><strong>{pct(latestNav?.drawdown)}</strong></div></div>
        {selectedVersion?.config && <small className="muted">硬风控：日亏 {pct(Number(selectedVersion.config.max_daily_loss ?? 0.03))} · 止损 {pct(Number(selectedVersion.config.stop_loss ?? 0.07))} · 止盈 {pct(Number(selectedVersion.config.take_profit ?? 0.20))} · 行业 {pct(Number(selectedVersion.config.max_industry_weight ?? 0.30))}</small>}
        <form className="rebalance-form" onSubmit={rebalance}><label>信号日期<input type="date" value={signalDate} onChange={(event) => setSignalDate(event.target.value)} /></label><label>模拟滑点（bp）<input type="number" min="0" max="200" step="1" value={rebalanceSlippage * 10000} onChange={(event) => setRebalanceSlippage(Number(event.target.value) / 10000)} /></label><button className="primary" disabled={!selected || !["active", "liquidation_pending", "risk_reduction_pending"].includes(selected.status) || !signalDate || activeBatch}>{riskPending ? "执行待处理风险动作" : "运行日终再平衡"}</button></form>
        {selected && <button className="secondary-action" disabled={riskPending} onClick={() => changeStatus(selected.status === "active" ? "paused" : "active")}>{riskPending ? "风险动作待执行" : selected.status === "active" ? "暂停组合" : "恢复组合"}</button>}
      </article>
    </section>
    <section className="metric-strip portfolio-metrics"><div><span>初始资金</span><strong>{money(selected?.initial_cash ?? 0)}</strong></div><div><span>累计收益</span><strong>{selected ? pct(selected.nav / selected.initial_cash - 1) : "—"}</strong></div><div><span>当日收益</span><strong>{pct(latestNav?.daily_return)}</strong></div><div><span>当日换手</span><strong>{pct(latestNav?.turnover)}</strong></div><div><span>累计风险事件</span><strong>{selected?.risk_events.length ?? 0}</strong></div><div><span>最近批次</span><strong>{selected?.batches[0]?.status ?? "—"}</strong></div></section>
    {latestReview && <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">POST-TRADE REVIEW</p><h2>盘后复盘 · {latestReview.trade_date}</h2></div><span className={`state ${latestReview.status === "ok" ? "ready" : latestReview.status === "action_required" ? "failed" : "partial"}`}>{latestReview.status}</span></div><div className="metric-strip portfolio-metrics"><div><span>当日净损益</span><strong>{money(latestReview.summary.net_pnl)}</strong></div><div><span>相对基准</span><strong>{pct(latestReview.summary.active_return ?? undefined)}</strong></div><div><span>成交率</span><strong>{pct(latestReview.summary.fill_rate)}</strong></div><div><span>费用</span><strong>{money(latestReview.summary.fees)}</strong></div><div><span>风险事件</span><strong>{latestReview.summary.risk_event_count}</strong></div><div><span>下一状态</span><strong>{latestReview.summary.next_portfolio_status}</strong></div></div><div className="portfolio-lower"><div><h3>主要正贡献</h3><div className="risk-list">{latestReview.summary.best_contributors.map((item) => <article key={`best-${item.instrument}`}><div><strong>{item.instrument}</strong><small>{money(item.pnl)}</small></div></article>)}</div></div><div><h3>主要负贡献 / 拒单</h3><div className="risk-list">{latestReview.summary.worst_contributors.filter((item) => item.pnl < 0).map((item) => <article key={`worst-${item.instrument}`}><div><strong>{item.instrument}</strong><small>{money(item.pnl)}</small></div></article>)}{latestReview.summary.rejections.slice(0, 5).map((item) => <article key={`reject-${item.instrument}`}><div><strong>{item.instrument}</strong><small>{item.reason}</small></div></article>)}</div></div></div></section>}
    <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">PAPER POSITIONS</p><h2>当前持仓</h2></div><span>{selected?.positions.length ?? 0} 只</span></div><div className="table-wrap"><table className="portfolio-table"><thead><tr><th>证券</th><th>行业（交易日）</th><th>止盈阶段</th><th>数量</th><th>市价</th><th>市值</th><th>权重</th><th>未实现盈亏</th><th>已实现盈亏</th></tr></thead><tbody>{selected?.positions.map((item) => <tr key={item.instrument}><td><code>{item.instrument}</code></td><td>{item.industry ?? "待校验"}</td><td>{item.take_profit_stage ? `阶段 ${item.take_profit_stage}` : "未触发"}</td><td>{item.quantity.toLocaleString()}</td><td>{item.market_price.toFixed(2)}</td><td>{money(item.market_value)}</td><td>{pct(item.weight)}</td><td>{money(item.unrealized_pnl)}</td><td>{money(item.realized_pnl)}</td></tr>)}</tbody></table>{!selected?.positions.length && <div className="empty">尚无持仓。运行第一个日终再平衡后，下一交易日开盘模拟成交会写入这里。</div>}</div></section>
    <section className="portfolio-lower">
      <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">ORDER LEDGER</p><h2>订单与批次</h2></div><span>{selected?.orders.length ?? 0} 笔订单</span></div><div className="table-wrap"><table className="portfolio-table"><thead><tr><th>证券</th><th>方向</th><th>数量</th><th>目标权重</th><th>状态</th><th>原因</th></tr></thead><tbody>{selected?.orders.slice(0, 20).map((item) => <tr key={item.id}><td><code>{item.instrument}</code></td><td>{item.side === "buy" ? "买入" : "卖出"}</td><td>{item.requested_quantity.toLocaleString()}</td><td>{pct(item.target_weight)}</td><td><span className={`state ${item.status === "filled" ? "ready" : "failed"}`}>{item.status}</span></td><td>{item.reason ?? "—"}</td></tr>)}</tbody></table>{!selected?.orders.length && <div className="empty compact">尚无订单。</div>}</div></section>
      <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">RISK EVENTS</p><h2>风险事件</h2></div><span>{selected?.risk_events.length ?? 0} 条</span></div><label>处置结论<textarea value={riskResolution} onChange={(event) => setRiskResolution(event.target.value)} placeholder="说明风险动作、仓位/敞口复核和恢复依据（至少 10 字）" /></label><div className="risk-list">{selected?.risk_events.map((item) => <article key={item.id}><span className={`job-state ${item.severity === "critical" ? "failed" : "running"}`} /><div><strong>{item.rule}</strong><small>{item.event_type} · 观察值 {pct(item.observed)} / 阈值 {pct(item.limit_value)}{item.resolution_reason ? ` · ${item.resolution_reason}` : ""}</small></div><span>{item.severity} · {item.status}</span><div>{item.status === "open" && <button className="inline-action" onClick={() => actOnRisk(item, "acknowledge")}>确认</button>}{["open", "acknowledged"].includes(item.status) && <button className="inline-action" disabled={riskResolution.trim().length < 10} onClick={() => actOnRisk(item, "resolve")}>完成处置</button>}</div></article>)}{!selected?.risk_events.length && <div className="empty compact">当前没有风险事件；越过硬阈值时组合会自动暂停。</div>}</div></section>
    </section>
  </>;
}
