"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { apiFetch } from "./api-client";

type Strategy = { id: string; versions: { id: string; status: string }[] };
type Dataset = { name: string; ready: boolean; reproducible: boolean; frequency?: string };
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
  construction_notional: number; latest_snapshot?: Snapshot | null; snapshots: Snapshot[];
};
type SimulationNav = {
  trade_date: string; cash: number; market_value: number; nav: number; daily_return: number;
  drawdown: number; has_stale_prices: boolean; status: string; performance_certified: boolean;
};
type SimulationPosition = {
  instrument: string; quantity: number; available_quantity: number; average_cost: number;
  market_price?: number | null; market_date?: string | null; stale: boolean; market_value: number;
};
type SimulationPortfolio = {
  id: string; name: string; recommendation_portfolio_id: string; status: string; cash: number;
  nav: number; execution_algorithm: string; execution_dataset: string; daily_dataset: string;
  cost_schedule_version: string; latest_nav?: SimulationNav | null;
};

const pct = (value?: number | null) => value == null ? "—" : `${(value * 100).toFixed(2)}%`;

export function PortfolioPanel({ api }: { api: string }) {
  const [portfolios, setPortfolios] = useState<RecommendationPortfolio[]>([]);
  const [simulations, setSimulations] = useState<SimulationPortfolio[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [name, setName] = useState("沪深300 推荐组合");
  const [versionId, setVersionId] = useState("");
  const [dataset, setDataset] = useState("");
  const [asOfDate, setAsOfDate] = useState(new Date().toISOString().slice(0, 10));
  const [message, setMessage] = useState("");
  const [selectedSimulationId, setSelectedSimulationId] = useState("");
  const [simulationName, setSimulationName] = useState("A股真实模拟账户");
  const [executionDataset, setExecutionDataset] = useState("");
  const [initialCash, setInitialCash] = useState(5_000_000);
  const [executionAlgorithm, setExecutionAlgorithm] = useState<"twap" | "vwap">("twap");
  const [simulationNav, setSimulationNav] = useState<SimulationNav[]>([]);
  const [simulationPositions, setSimulationPositions] = useState<SimulationPosition[]>([]);

  const load = useCallback(async () => {
    const [portfolioResponse, strategyResponse, datasetResponse, simulationResponse] = await Promise.all([
      apiFetch(`${api}/api/recommendation-portfolios`, { cache: "no-store" }),
      apiFetch(`${api}/api/strategies`, { cache: "no-store" }),
      apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" }),
      apiFetch(`${api}/api/simulation-portfolios`, { cache: "no-store" }),
    ]);
    if (!portfolioResponse.ok) throw new Error("recommendations unavailable");
    const nextPortfolios = await portfolioResponse.json() as RecommendationPortfolio[];
    const nextStrategies = strategyResponse.ok ? await strategyResponse.json() as Strategy[] : [];
    const nextDatasets = datasetResponse.ok ? await datasetResponse.json() as Dataset[] : [];
    const nextSimulations = simulationResponse.ok
      ? await simulationResponse.json() as SimulationPortfolio[] : [];
    setPortfolios(nextPortfolios); setStrategies(nextStrategies); setDatasets(nextDatasets);
    setSimulations(nextSimulations);
    if (!selectedId && nextPortfolios.length) setSelectedId(nextPortfolios[0].id);
    if (!versionId) {
      const approved = nextStrategies.flatMap((item) => item.versions).find((item) => item.status === "approved");
      if (approved) setVersionId(approved.id);
    }
    if (!dataset) {
      const ready = nextDatasets.find((item) => item.ready && item.reproducible && item.frequency === "day");
      if (ready) setDataset(ready.name);
    }
    if (!executionDataset) {
      const readyExecution = nextDatasets.find((item) => item.ready && item.reproducible && item.frequency === "5min");
      if (readyExecution) setExecutionDataset(readyExecution.name);
    }
    if (!selectedSimulationId && nextSimulations.length) setSelectedSimulationId(nextSimulations[0].id);
  }, [api, dataset, executionDataset, selectedId, selectedSimulationId, versionId]);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      load().catch(() => setMessage("无法读取推荐组合。"));
    }, 0);
    return () => window.clearTimeout(initial);
  }, [load]);
  useEffect(() => {
    if (!selectedSimulationId) return;
    Promise.all([
      apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}`, { cache: "no-store" }),
      apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}/nav`, { cache: "no-store" }),
      apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}/positions`, { cache: "no-store" }),
    ]).then(async ([detailResponse, navResponse, positionResponse]) => {
      if (detailResponse.ok) {
        const detail = await detailResponse.json() as SimulationPortfolio;
        setSimulations((current) => current.map((item) => item.id === detail.id ? detail : item));
      }
      setSimulationNav(navResponse.ok ? await navResponse.json() as SimulationNav[] : []);
      setSimulationPositions(positionResponse.ok ? await positionResponse.json() as SimulationPosition[] : []);
    }).catch(() => setMessage("无法读取模拟账户账本。"));
  }, [api, selectedSimulationId]);
  const selected = portfolios.find((item) => item.id === selectedId) ?? portfolios[0];
  const snapshot = selected?.latest_snapshot;
  const selectedSimulation = simulations.find((item) => item.id === selectedSimulationId);
  const latestSimulationNav = simulationNav.at(-1) ?? selectedSimulation?.latest_nav;

  async function create(event: FormEvent) {
    event.preventDefault();
    const response = await apiFetch(`${api}/api/recommendation-portfolios`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, strategy_version_id: versionId, dataset, construction_notional: 5_000_000 }),
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

  async function createSimulation(event: FormEvent) {
    event.preventDefault(); if (!selected) return;
    const response = await apiFetch(`${api}/api/simulation-portfolios`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: simulationName, recommendation_portfolio_id: selected.id,
        execution_dataset: executionDataset, initial_cash: initialCash,
        execution_algorithm: executionAlgorithm, cost_schedule_version: "cn-effective-cost-v1",
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "模拟账户创建失败"); return; }
    setSelectedSimulationId(body.id); setMessage("模拟账户已创建，激活后从下一份推荐快照开始撮合。");
    await load();
  }

  async function setSimulationStatus(status: "active" | "pause") {
    if (!selectedSimulation) return;
    const response = await apiFetch(
      `${api}/api/simulation-portfolios/${selectedSimulation.id}/${status === "active" ? "activate" : "pause"}`,
      { method: "POST" },
    );
    const body = await response.json();
    setMessage(response.ok ? `模拟账户已${status === "active" ? "激活" : "暂停"}。` : body.detail ?? "状态更新失败");
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
    <section className="data-panel">
      <div className="panel-heading"><div><p className="eyebrow">TRUE SIMULATION LEDGER</p><h2>真实模拟撮合账户</h2></div><span className={`state ${latestSimulationNav?.performance_certified ? "ready" : "partial"}`}>{latestSimulationNav?.performance_certified ? "绩效可认证" : "尚无可认证净值"}</span></div>
      <div className="portfolio-hero">
        <form className="portfolio-launcher" onSubmit={createSimulation}>
          <label>账户名称<input value={simulationName} onChange={(event) => setSimulationName(event.target.value)} /></label>
          <label>5 分钟执行数据<select value={executionDataset} onChange={(event) => setExecutionDataset(event.target.value)}>{datasets.filter((item) => item.ready && item.reproducible && item.frequency === "5min").map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
          <label>初始现金<input type="number" min="100000" step="100000" value={initialCash} onChange={(event) => setInitialCash(Number(event.target.value))} /></label>
          <label>执行算法<select value={executionAlgorithm} onChange={(event) => setExecutionAlgorithm(event.target.value as "twap" | "vwap")}><option value="twap">TWAP</option><option value="vwap">VWAP</option></select></label>
          <button className="primary" disabled={!selected || !executionDataset || simulationName.length < 3}>创建模拟账户</button>
        </form>
        <article className="portfolio-summary">
          <label>模拟账户<select value={selectedSimulation?.id ?? ""} onChange={(event) => setSelectedSimulationId(event.target.value)}><option value="">尚未选择</option>{simulations.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
          <div className="metric-strip portfolio-metrics"><div><span>状态</span><strong>{selectedSimulation?.status ?? "—"}</strong></div><div><span>现金</span><strong>¥{Number(selectedSimulation?.cash ?? 0).toFixed(2)}</strong></div><div><span>NAV</span><strong>¥{Number(latestSimulationNav?.nav ?? selectedSimulation?.nav ?? 0).toFixed(2)}</strong></div><div><span>行情状态</span><strong>{latestSimulationNav?.has_stale_prices ? "陈旧/降级" : latestSimulationNav ? "正常" : "待运行"}</strong></div></div>
          <div className="button-row"><button type="button" className="primary" disabled={!selectedSimulation || selectedSimulation.status === "active"} onClick={() => setSimulationStatus("active")}>激活</button><button type="button" disabled={!selectedSimulation || selectedSimulation.status === "paused"} onClick={() => setSimulationStatus("pause")}>暂停</button></div>
          <small>推荐只提供目标权重；此处净值只来自 T+1 分钟撮合、成交、费用、现金和持仓账本。</small>
        </article>
      </div>
      <div className="table-wrap"><table><thead><tr><th>证券</th><th>持仓</th><th>可卖</th><th>成本</th><th>行情日</th><th>市值</th><th>估值状态</th></tr></thead><tbody>{simulationPositions.map((position) => <tr key={position.instrument}><td><code>{position.instrument}</code></td><td>{position.quantity}</td><td>{position.available_quantity}</td><td>{Number(position.average_cost).toFixed(4)}</td><td>{position.market_date ?? "—"}</td><td>¥{Number(position.market_value).toFixed(2)}</td><td><span className={`state ${position.stale ? "failed" : "ready"}`}>{position.stale ? "stale" : "current"}</span></td></tr>)}</tbody></table>{selectedSimulation && !simulationPositions.length && <div className="empty">账户尚无持仓；等待下一次成功的推荐快照进入 T+1 撮合。</div>}</div>
    </section>
    {snapshot && <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">COST ASSUMPTIONS</p><h2>统一成本假设</h2></div></div><div className="metric-strip portfolio-metrics"><div><span>买入费率</span><strong>{pct(snapshot.cost_model.buy_commission_rate)}</strong></div><div><span>卖出费率</span><strong>{pct(snapshot.cost_model.sell_commission_rate)}</strong></div><div><span>固定滑点</span><strong>{pct(snapshot.cost_model.fixed_slippage_rate)}</strong></div><div><span>参与率上限</span><strong>{pct(snapshot.cost_model.max_volume_participation)}</strong></div><div><span>最低佣金</span><strong>¥{snapshot.cost_model.min_commission}</strong></div></div></section>}
  </div>;
}
