"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch, type PaperTargetProjection } from "./api-client";

type StrategyVersion = {
  id: string; status: string; strategy_type: string; execution_frequency?: string;
  config?: { execution_frequency?: string; execution_method?: string };
};
type Strategy = { id: string; name: string; versions: StrategyVersion[] };
type Allocation = { id: string; name: string; status: string };
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
  position_side?: "long" | "short";
};
type SimulationPortfolio = {
  id: string; name: string; recommendation_portfolio_id?: string | null;
  source_type: "recommendation" | "strategy_version" | "allocation"; source_id: string;
  execution_adapter: "long_only" | "pair"; execution_frequency: "1min" | "5min";
  execution_contract_hash: string; status: string; cash: number;
  nav: number; execution_algorithm: string; execution_dataset: string; daily_dataset: string;
  cost_schedule_version: string; benchmark?: string | null; latest_nav?: SimulationNav | null;
  simulation_mode?: "paper" | "shadow_pair"; synthetic_short_exposure?: boolean;
  financing_enabled?: boolean; real_trading_eligible?: boolean;
};
type SimulationPerformance = {
  nav_days: number;
  unitized: {
    status: string; twr?: number | null; max_drawdown?: number | null;
    recovery_trading_days?: number | null;
  };
  statistics: {
    status: string; twr?: number | null; cagr?: number | null;
    annualized_volatility?: number | null; sharpe_ratio?: number | null;
    sortino_ratio?: number | null; metric_status?: Record<string, string>;
  };
  relative_performance: {
    status: string; annualized_excess_return?: number | null;
    information_ratio?: number | null; tracking_error?: number | null;
    benchmark?: string | null; broken_from?: string | null;
  };
  xirr: { status: string; rate?: number | null };
};
type SimulationBatch = {
  id: string; signal_date: string; trade_date: string; status: string;
  execution_adapter: string; created_at: string; error?: string | null;
};
type SimulationOrder = {
  id: string; instrument: string; side: string; requested_quantity: number;
  filled_quantity: number; status: string; reject_reason?: string | null;
  requested_value: number; filled_value: number; created_at: string;
};
type SimulationFill = {
  id: string; instrument: string; side: string; executed_at: string;
  quantity: number; price: number; gross_value: number; fee: number;
};
type SimulationEvent = {
  id: string; trade_date: string; severity: string; event_type: string;
  instrument?: string | null; reason: string; created_at: string;
};
type SimulationCashFlow = {
  id: string; trade_date: string; flow_type: string; amount: number;
  balance_after: number; created_at: string;
};

const pct = (value?: number | null) => value == null ? "—" : `${(value * 100).toFixed(2)}%`;
const decimal = (value?: number | null) => value == null ? "—" : value.toFixed(2);
const money = (value?: number | null) => value == null
  ? "—"
  : new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 2 }).format(value);
const shortTime = (value?: string | null) => value
  ? new Date(value).toLocaleString("zh-CN", { hour12: false })
  : "—";

async function jsonResponse<T>(request: Promise<Response>): Promise<T> {
  const response = await request;
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export function PortfolioPanel({ api }: { api: string }) {
  const [portfolios, setPortfolios] = useState<RecommendationPortfolio[]>([]);
  const [simulations, setSimulations] = useState<SimulationPortfolio[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [allocations, setAllocations] = useState<Allocation[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [name, setName] = useState("沪深300 推荐组合");
  const [versionId, setVersionId] = useState("");
  const [dataset, setDataset] = useState("");
  const [asOfDate, setAsOfDate] = useState(new Date().toISOString().slice(0, 10));
  const [message, setMessage] = useState("");
  const [loadMessage, setLoadMessage] = useState("");
  const [ledgerMessage, setLedgerMessage] = useState("");
  const [selectedSimulationId, setSelectedSimulationId] = useState("");
  const [simulationName, setSimulationName] = useState("A股模拟账户");
  const [simulationSourceType, setSimulationSourceType] = useState<"recommendation" | "strategy_version" | "allocation">("recommendation");
  const [simulationSourceId, setSimulationSourceId] = useState("");
  const [executionFrequency, setExecutionFrequency] = useState<"1min" | "5min">("5min");
  const [executionDataset, setExecutionDataset] = useState("");
  const [initialCash, setInitialCash] = useState(5_000_000);
  const [simulationNav, setSimulationNav] = useState<SimulationNav[]>([]);
  const [simulationPositions, setSimulationPositions] = useState<SimulationPosition[]>([]);
  const [simulationPerformance, setSimulationPerformance] = useState<SimulationPerformance | null>(null);
  const [simulationBatches, setSimulationBatches] = useState<SimulationBatch[]>([]);
  const [simulationOrders, setSimulationOrders] = useState<SimulationOrder[]>([]);
  const [simulationFills, setSimulationFills] = useState<SimulationFill[]>([]);
  const [simulationEvents, setSimulationEvents] = useState<SimulationEvent[]>([]);
  const [simulationCashFlows, setSimulationCashFlows] = useState<SimulationCashFlow[]>([]);
  const [paperTarget, setPaperTarget] = useState<PaperTargetProjection | null>(null);

  const load = useCallback(async () => {
    let nextPortfolios: RecommendationPortfolio[] | undefined;
    let nextStrategies: Strategy[] | undefined;
    let nextAllocations: Allocation[] | undefined;
    const resources = [
      {
        label: "推荐组合",
        request: jsonResponse<RecommendationPortfolio[]>(
          apiFetch(`${api}/api/recommendation-portfolios`, { cache: "no-store" }),
        ).then((items) => {
          nextPortfolios = items;
          setPortfolios(items);
          setSelectedId((current) => current && items.some((item) => item.id === current)
            ? current
            : items[0]?.id ?? "");
        }),
      },
      {
        label: "策略",
        request: jsonResponse<Strategy[]>(apiFetch(`${api}/api/strategies`, { cache: "no-store" })).then((items) => {
          nextStrategies = items;
          setStrategies(items);
          const approved = items.flatMap((item) => item.versions).find((item) => item.status === "approved");
          setVersionId((current) => current || approved?.id || "");
        }),
      },
      {
        label: "Qlib 数据集",
        request: jsonResponse<Dataset[]>(apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" })).then((items) => {
          setDatasets(items);
          const ready = items.find((item) => item.ready && item.reproducible && item.frequency === "day");
          const readyExecution = items.find((item) => item.ready && item.reproducible && item.frequency === "5min");
          setDataset((current) => current || ready?.name || "");
          setExecutionDataset((current) => current || readyExecution?.name || "");
        }),
      },
      {
        label: "模拟账户",
        request: jsonResponse<SimulationPortfolio[]>(
          apiFetch(`${api}/api/simulation-portfolios`, { cache: "no-store" }),
        ).then((items) => {
          setSimulations(items);
          setSelectedSimulationId((current) => current && items.some((item) => item.id === current)
            ? current
            : items[0]?.id ?? "");
        }),
      },
      {
        label: "模拟候选",
        request: jsonResponse<PaperTargetProjection>(
          apiFetch(`${api}/api/autopilot/paper-target`, { cache: "no-store" }),
        ).then((item) => setPaperTarget(item)),
      },
      {
        label: "策略分配",
        request: jsonResponse<Allocation[]>(
          apiFetch(`${api}/api/strategy-allocations`, { cache: "no-store" }),
        ).then((items) => {
          nextAllocations = items;
          setAllocations(items);
        }),
      },
    ];
    const results = await Promise.allSettled(resources.map((item) => item.request));
    if (nextPortfolios && nextStrategies && nextAllocations) {
      const hasRecommendation = nextPortfolios.length > 0;
      const hasStrategy = nextStrategies.some((strategy) => strategy.versions.some(
        (version) => version.status === "approved" && version.strategy_type !== "pair",
      ));
      const hasAllocation = nextAllocations.some((item) => item.status === "active");
      setSimulationSourceType((current) => {
        if (
          (current === "recommendation" && hasRecommendation)
          || (current === "strategy_version" && hasStrategy)
          || (current === "allocation" && hasAllocation)
        ) return current;
        if (hasRecommendation) return "recommendation";
        if (hasStrategy) return "strategy_version";
        if (hasAllocation) return "allocation";
        return current;
      });
    }
    const failed = results.flatMap((result, index) => result.status === "rejected" ? [resources[index].label] : []);
    setLoadMessage(
      failed.length === 0
        ? ""
        : failed.length === resources.length
          ? "推荐与模拟目录暂时无法更新，仍保留上次成功数据。"
          : `部分目录暂未更新，已保留上次成功内容：${failed.join("、")}。`,
    );
  }, [api]);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      load().catch(() => setLoadMessage("推荐与模拟目录暂时无法更新，仍保留上次成功数据。"));
    }, 0);
    return () => window.clearTimeout(initial);
  }, [load]);
  useEffect(() => {
    if (!selectedSimulationId) return;
    let disposed = false;
    async function refreshLedger() {
      const resources = [
        {
          label: "账户",
          request: jsonResponse<SimulationPortfolio>(
            apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}`, { cache: "no-store" }),
          ).then((detail) => {
            if (disposed) return;
            setSimulations((current) => current.map((item) => item.id === detail.id ? detail : item));
          }),
        },
        {
          label: "净值",
          request: jsonResponse<SimulationNav[]>(
            apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}/nav?limit=5000`, { cache: "no-store" }),
          ).then((items) => { if (!disposed) setSimulationNav(items); }),
        },
        {
          label: "持仓",
          request: jsonResponse<SimulationPosition[]>(
            apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}/positions`, { cache: "no-store" }),
          ).then((items) => { if (!disposed) setSimulationPositions(items); }),
        },
        {
          label: "绩效",
          request: jsonResponse<SimulationPerformance>(
            apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}/performance`, { cache: "no-store" }),
          ).then((item) => { if (!disposed) setSimulationPerformance(item); }),
        },
        {
          label: "批次",
          request: jsonResponse<SimulationBatch[]>(
            apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}/batches?limit=100`, { cache: "no-store" }),
          ).then((items) => { if (!disposed) setSimulationBatches(items); }),
        },
        {
          label: "订单",
          request: jsonResponse<SimulationOrder[]>(
            apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}/orders?limit=200`, { cache: "no-store" }),
          ).then((items) => { if (!disposed) setSimulationOrders(items); }),
        },
        {
          label: "成交",
          request: jsonResponse<SimulationFill[]>(
            apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}/fills?limit=200`, { cache: "no-store" }),
          ).then((items) => { if (!disposed) setSimulationFills(items); }),
        },
        {
          label: "事件",
          request: jsonResponse<SimulationEvent[]>(
            apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}/events?limit=200`, { cache: "no-store" }),
          ).then((items) => { if (!disposed) setSimulationEvents(items); }),
        },
        {
          label: "资金流水",
          request: jsonResponse<SimulationCashFlow[]>(
            apiFetch(`${api}/api/simulation-portfolios/${selectedSimulationId}/cash_flows?limit=200`, { cache: "no-store" }),
          ).then((items) => { if (!disposed) setSimulationCashFlows(items); }),
        },
      ];
      const results = await Promise.allSettled(resources.map((item) => item.request));
      if (disposed) return;
      const failed = results.flatMap((result, index) => result.status === "rejected" ? [resources[index].label] : []);
      setLedgerMessage(
        failed.length === 0
          ? ""
          : failed.length === resources.length
            ? "模拟账本暂时无法更新，仍显示上次成功数据。"
            : `模拟账本部分数据暂未更新，已保留上次成功内容：${failed.join("、")}。`,
      );
    }
    let timer: number | undefined;
    async function pollLedger() {
      try {
        await refreshLedger();
      } catch {
        if (!disposed) setLedgerMessage("模拟账本暂时无法更新，仍显示上次成功数据。");
      } finally {
        if (!disposed) timer = window.setTimeout(pollLedger, 30_000);
      }
    }
    void pollLedger();
    return () => {
      disposed = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [api, selectedSimulationId]);
  const selected = portfolios.find((item) => item.id === selectedId) ?? portfolios[0];
  const snapshot = selected?.latest_snapshot;
  const selectedSimulation = simulations.find((item) => item.id === selectedSimulationId);
  const latestSimulationNav = simulationNav.at(-1) ?? selectedSimulation?.latest_nav;
  const openOrders = simulationOrders.filter((item) => ["planned", "open"].includes(item.status));
  const failedBatches = simulationBatches.filter((item) => item.status === "failed");
  const navPoints = simulationNav.slice(-90);
  const navValues = navPoints.map((item) => Number(item.nav)).filter(Number.isFinite);
  const navMin = navValues.length ? Math.min(...navValues) : 0;
  const navMax = navValues.length ? Math.max(...navValues) : 0;
  const navPath = navPoints.map((item, index) => {
    const x = navPoints.length <= 1 ? 0 : (index / (navPoints.length - 1)) * 100;
    const range = navMax - navMin;
    const y = range <= 0 ? 50 : 94 - ((Number(item.nav) - navMin) / range) * 88;
    return `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  const approvedLongVersions = useMemo(() => strategies.flatMap((strategy) => strategy.versions
    .filter((version) => version.status === "approved" && version.strategy_type !== "pair")
    .map((version) => ({ id: version.id, label: `${strategy.name} · ${version.id.slice(0, 10)}`, version }))), [strategies]);
  const activeAllocations = useMemo(() => allocations.filter((item) => item.status === "active"), [allocations]);
  const sourceOptions = simulationSourceType === "recommendation"
    ? portfolios.map((item) => ({ id: item.id, label: item.name }))
    : simulationSourceType === "strategy_version"
      ? approvedLongVersions
      : activeAllocations.map((item) => ({ id: item.id, label: item.name }));
  const activeSimulationSourceId = sourceOptions.some((item) => item.id === simulationSourceId)
    ? simulationSourceId
    : sourceOptions[0]?.id ?? "";
  const governedVersion = simulationSourceType === "strategy_version"
    ? approvedLongVersions.find((item) => item.id === activeSimulationSourceId)?.version
    : simulationSourceType === "recommendation"
      ? strategies.flatMap((item) => item.versions).find(
        (version) => version.id === portfolios.find((item) => item.id === activeSimulationSourceId)?.strategy_version_id,
      )
      : undefined;
  const governedExecutionFrequency = governedVersion?.execution_frequency
    ?? governedVersion?.config?.execution_frequency;
  const governedExecutionAlgorithm = simulationSourceType === "allocation"
    ? "twap"
    : governedVersion?.config?.execution_method ?? "—";
  const activeExecutionFrequency = governedExecutionFrequency === "1min"
    ? "1min"
    : governedExecutionFrequency === "5min"
      ? "5min"
      : executionFrequency;
  const executionDatasets = datasets.filter((item) =>
    item.ready && item.reproducible && item.frequency === activeExecutionFrequency,
  );
  const activeExecutionDataset = executionDatasets.some((item) => item.name === executionDataset)
    ? executionDataset
    : executionDatasets[0]?.name ?? "";

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
    event.preventDefault(); if (!activeSimulationSourceId) return;
    const response = await apiFetch(`${api}/api/simulation-portfolios`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: simulationName,
        source_type: simulationSourceType,
        source_id: activeSimulationSourceId,
        execution_dataset: activeExecutionDataset,
        execution_frequency: activeExecutionFrequency,
        execution_adapter: "long_only",
        initial_cash: initialCash,
        cost_schedule_version: "cn-effective-cost-v1",
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "模拟账户创建失败"); return; }
    setSelectedSimulationId(body.id); setMessage("统一模拟账户已创建；只消费受治理来源并按执行契约撮合。");
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
    {loadMessage && <div className="notice">{loadMessage}</div>}
    {ledgerMessage && <div className="notice">{ledgerMessage}</div>}
    {message && <div className="notice">{message}</div>}
    <section className="recommendation-command">
      <article className="portfolio-summary">
        <div className="card-heading"><div><span>AUTOMATIC RECOMMENDATION</span><strong>{selected?.name ?? "等待合格策略"}</strong></div><span className="status-chip">{selected?.status ?? "等待门禁"}</span></div>
        <label>组合<select value={selected?.id ?? ""} onChange={(event) => setSelectedId(event.target.value)}>{portfolios.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
        <small>{selected ? "交易日数据发布后自动刷新推荐，并交给模拟账本执行。" : "研究、独立验证和正式回测通过后，系统会自动创建推荐组合。"}</small>
      </article>
      <details className="advanced-config recommendation-manual-tools"><summary>高级操作 · 手工创建或重算推荐</summary><div className="manual-recommendation-grid">
        <form className="portfolio-launcher" onSubmit={create}>
          <label>名称<input value={name} onChange={(event) => setName(event.target.value)} /></label>
          <label>已审批v2策略<select value={versionId} onChange={(event) => setVersionId(event.target.value)}>{strategies.flatMap((item) => item.versions).filter((item) => item.status === "approved").map((item) => <option key={item.id} value={item.id}>{item.id.slice(0, 16)}</option>)}</select></label>
          <label>Qlib数据集<select value={dataset} onChange={(event) => setDataset(event.target.value)}>{datasets.filter((item) => item.ready && item.reproducible).map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
          <button className="primary" disabled={!versionId || !dataset || name.length < 3}>创建推荐组合</button>
        </form>
        <form className="portfolio-launcher" onSubmit={refresh}><label>重算日期<input type="date" value={asOfDate} onChange={(event) => setAsOfDate(event.target.value)} /></label><button className="primary" disabled={!selected || !asOfDate}>重算当前推荐</button></form>
      </div></details>
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
      <div className="panel-heading"><div><p className="eyebrow">UNIFIED SIMULATION LEDGER</p><h2>统一持久模拟盘</h2><p>当前自动驾驶只运行多头账户；旧影子配对账户仅保留为只读历史证据。</p></div><span className={`state ${latestSimulationNav?.performance_certified ? "ready" : "partial"}`}>{latestSimulationNav?.performance_certified ? "绩效可认证" : "尚无可认证净值"}</span></div>
      <div className="simulation-command">
        <article className="portfolio-summary simulation-account-bar">
          <div><span className="eyebrow">ACCOUNT CONTROL</span><strong>{selectedSimulation?.name ?? "等待自动创建模拟账户"}</strong><small>{selectedSimulation ? `来源 ${selectedSimulation.source_type} · ${selectedSimulation.daily_dataset}` : "系统只会在独立验证、最终 OOS 和正式成本回测全部通过后创建账户，不生成演示数据。"}</small></div>
          <label>模拟账户<select value={selectedSimulation?.id ?? ""} onChange={(event) => setSelectedSimulationId(event.target.value)}><option value="">尚未选择</option>{simulations.map((item) => <option key={item.id} value={item.id}>{item.simulation_mode === "shadow_pair" ? "[影子配对] " : "[多头] "}{item.name}</option>)}</select></label>
          <div className="button-row"><button type="button" className="primary" disabled={!selectedSimulation || selectedSimulation.status === "active" || selectedSimulation.simulation_mode === "shadow_pair"} onClick={() => setSimulationStatus("active")}>恢复运行</button><button type="button" disabled={!selectedSimulation || selectedSimulation.status === "paused"} onClick={() => setSimulationStatus("pause")}>暂停</button></div>
        </article>
        <details className="advanced-config simulation-manual-tools">
          <summary>高级操作 · 手工创建受治理账户</summary>
          <form className="portfolio-launcher" onSubmit={createSimulation}>
          <label>账户名称<input value={simulationName} onChange={(event) => setSimulationName(event.target.value)} /></label>
          <label>受治理来源<select value={simulationSourceType} onChange={(event) => { setSimulationSourceType(event.target.value as typeof simulationSourceType); setSimulationSourceId(""); }}><option value="recommendation">推荐组合</option><option value="strategy_version">已审批策略版本</option><option value="allocation">已审批核心 / 卫星分配</option></select></label>
          <label>来源版本<select value={activeSimulationSourceId} onChange={(event) => setSimulationSourceId(event.target.value)}><option value="">无可用来源</option>{sourceOptions.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
          <label>执行频率<select value={activeExecutionFrequency} disabled={governedExecutionFrequency === "1min" || governedExecutionFrequency === "5min"} onChange={(event) => setExecutionFrequency(event.target.value as "1min" | "5min")}><option value="1min">1 分钟</option><option value="5min">5 分钟</option></select></label>
          <label>{activeExecutionFrequency === "1min" ? "1 分钟" : "5 分钟"}执行数据<select value={activeExecutionDataset} onChange={(event) => setExecutionDataset(event.target.value)}><option value="">无可用数据</option>{executionDatasets.map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
          <label>初始现金<input type="number" min="100000" step="100000" value={initialCash} onChange={(event) => setInitialCash(Number(event.target.value))} /></label>
          <label>受控执行算法<input value={governedExecutionAlgorithm.toUpperCase()} disabled /></label>
          <button className="primary" disabled={!activeSimulationSourceId || !activeExecutionDataset || simulationName.length < 3}>创建模拟账户</button>
          </form>
        </details>
      </div>
      <section className="paper-target-readonly workspace-card">
        <div className="panel-heading"><div><p className="eyebrow">AUTOPILOT PAPER TARGETS · READ ONLY</p><h2>今日模拟候选与目标权重</h2><p>{paperTarget?.message ?? "正在读取自动驾驶模拟账户。"}</p></div><span className={`state ${paperTarget?.status === "ready" ? "ready" : paperTarget?.status === "blocked_invalid_order_plan" ? "failed" : "partial"}`}>{paperTarget?.status === "ready" ? "模拟候选" : paperTarget?.status === "blocked_invalid_order_plan" ? "已阻断" : "等待"}</span></div>
        {paperTarget?.status === "ready" ? <>
          <div className="autopilot-paper-target-meta"><span>信号日 <strong>{paperTarget.signal_date}</strong></span><span>拟执行日 <strong>{paperTarget.trade_date}</strong></span><span>现金目标 <strong>{pct(paperTarget.cash_weight)}</strong></span><span>批次 <code>{paperTarget.batch?.id.slice(0, 12)}</code></span></div>
          <div className="table-wrap"><table className="portfolio-table compact-ledger-table"><thead><tr><th>证券</th><th>目标权重</th><th>上次目标</th><th>调整</th><th>依据</th></tr></thead><tbody>{paperTarget.targets.slice(0, 30).map((item) => <tr key={item.instrument}><td><code>{item.instrument}</code></td><td>{pct(item.target_weight)}</td><td>{pct(item.previous_weight)}</td><td>{({ add: "新增", remove: "移除", increase: "增加", decrease: "减少", hold: "维持" } as Record<string, string>)[item.action]}</td><td>{item.reason}</td></tr>)}</tbody></table></div>
          <small className="autopilot-paper-target-note">只读模拟订单计划：它不会开启 RecommendationStore，也不会创建真实订单、融资或券商连接。</small>
        </> : <div className="empty">{paperTarget?.blocker ? `订单计划校验失败：${paperTarget.blocker}` : "等待自动驾驶通过全部门禁后创建模拟账户并生成首份订单计划。"}</div>}
      </section>
      {selectedSimulation?.simulation_mode === "shadow_pair" && <div className="notice"><strong>历史影子账户 · 只读</strong>：空头、借券费和可做空性均为旧的模拟假设；该账户不能恢复、下单、推荐或进入当前自动驾驶。</div>}
      <div className="metric-strip portfolio-metrics simulation-ledger-kpis"><div><span>账户状态</span><strong>{selectedSimulation?.status ?? "等待门禁"}</strong></div><div><span>总资产</span><strong>{money(Number(latestSimulationNav?.nav ?? selectedSimulation?.nav ?? 0) || null)}</strong></div><div><span>可用现金</span><strong>{money(Number(selectedSimulation?.cash ?? 0) || null)}</strong></div><div><span>持仓数</span><strong>{simulationPositions.length}</strong></div><div><span>待成交订单</span><strong>{openOrders.length}</strong></div><div><span>异常批次</span><strong className={failedBatches.length ? "danger-text" : ""}>{failedBatches.length}</strong></div></div>
      {!selectedSimulation && <div className="simulation-blocker"><strong>当前不会伪造一个空模拟盘</strong><span>{sourceOptions.length ? "已有受治理来源，等待自动编排创建并激活账户。" : "尚无通过最终 OOS 与正式成本回测的策略；自动研究通过硬门禁后会自行创建账户。"}</span></div>}
      {selectedSimulation && <section className="simulation-ledger-grid">
        <article className="workspace-card nav-card">
          <div className="panel-heading"><div><p className="eyebrow">NAV · LAST 90 DAYS</p><h2>账户净值</h2></div><strong>{money(Number(latestSimulationNav?.nav ?? selectedSimulation.nav))}</strong></div>
          <div className="nav-chart">{navPath ? <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="近90日净值曲线"><defs><linearGradient id="navArea" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#67a33d" stopOpacity=".28"/><stop offset="1" stopColor="#67a33d" stopOpacity="0"/></linearGradient></defs><path className="nav-area" d={`${navPath} L100,100 L0,100 Z`}/><path className="nav-line" d={navPath}/></svg> : <div className="empty">首个交易日完成后显示真实净值，不绘制示例曲线。</div>}</div>
          <div className="nav-chart-axis"><span>{navPoints[0]?.trade_date ?? "—"}</span><span>{navPoints.at(-1)?.trade_date ?? "—"}</span></div>
        </article>
        <article className="workspace-card ledger-health-card">
          <div className="panel-heading"><div><p className="eyebrow">EXECUTION HEALTH</p><h2>执行与对账</h2></div><span className={`state ${failedBatches.length || latestSimulationNav?.has_stale_prices ? "failed" : "ready"}`}>{failedBatches.length ? "需要处理" : "正常"}</span></div>
          <dl className="ledger-health-list"><div><dt>最近交易批次</dt><dd>{simulationBatches.at(-1)?.trade_date ?? "尚未生成"}</dd></div><div><dt>最近成交</dt><dd>{shortTime(simulationFills.at(-1)?.executed_at)}</dd></div><div><dt>现金流水</dt><dd>{simulationCashFlows.length} 条</dd></div><div><dt>行情估值</dt><dd>{latestSimulationNav?.has_stale_prices ? "存在陈旧价格" : latestSimulationNav ? "完整" : "待首日运行"}</dd></div></dl>
          <small>每 30 秒自动刷新；T+1、停牌、涨跌停、费用和滑点均由账本记录。</small>
        </article>
      </section>}
      {selectedSimulation && <section className="workspace-card">
        <div className="panel-heading"><div><p className="eyebrow">CERTIFIED ACCOUNT PERFORMANCE</p><h2>账户单位净值绩效</h2><p>收益、回撤和风险指标来自扣除费用后的单位化 TWR；入金和出金不会制造收益。</p></div><span className={`state ${simulationPerformance?.statistics.status === "ok" ? "ready" : "partial"}`}>{simulationPerformance?.statistics.status ?? "证据不足"}</span></div>
        <div className="metric-strip portfolio-metrics">
          <div><span>累计 TWR</span><strong>{pct(simulationPerformance?.statistics.twr)}</strong></div>
          <div><span>CAGR</span><strong>{pct(simulationPerformance?.statistics.cagr)}</strong></div>
          <div><span>年化波动</span><strong>{pct(simulationPerformance?.statistics.annualized_volatility)}</strong></div>
          <div><span>Sharpe</span><strong>{decimal(simulationPerformance?.statistics.sharpe_ratio)}</strong></div>
          <div><span>Sortino</span><strong>{decimal(simulationPerformance?.statistics.sortino_ratio)}</strong></div>
          <div><span>最大回撤</span><strong>{pct(simulationPerformance?.unitized.max_drawdown)}</strong></div>
        </div>
        <div className="metric-strip portfolio-metrics">
          <div><span>XIRR · 资金体验</span><strong>{pct(simulationPerformance?.xirr.rate)}</strong></div>
          <div><span>恢复交易日</span><strong>{simulationPerformance?.unitized.recovery_trading_days ?? "—"}</strong></div>
          <div><span>净值天数</span><strong>{simulationPerformance?.nav_days ?? 0}</strong></div>
          <div><span>年化超额 · {simulationPerformance?.relative_performance.benchmark ?? selectedSimulation.benchmark ?? "未绑定基准"}</span><strong>{pct(simulationPerformance?.relative_performance.annualized_excess_return)}</strong></div>
          <div><span>Information Ratio</span><strong>{decimal(simulationPerformance?.relative_performance.information_ratio)}</strong></div>
          <div><span>Tracking Error</span><strong>{pct(simulationPerformance?.relative_performance.tracking_error)}</strong></div>
        </div>
        {simulationPerformance?.relative_performance.status === "benchmark_not_configured" && <div className="notice">账户政策基准尚未配置：相对收益、IR 和 Tracking Error 明确保持未定义，不以 0 冒充。</div>}
        {simulationPerformance?.relative_performance.status === "unavailable_broken_benchmark_chain" && <div className="notice">基准证据链在 {simulationPerformance.relative_performance.broken_from ?? "未知日期"} 缺失或中断：相对收益、IR 和 Tracking Error 暂不发布，补齐逐日基准行情后才会恢复。</div>}
      </section>}
      <div className="table-wrap"><table><thead><tr><th>证券</th><th>方向</th><th>持仓</th><th>可卖</th><th>成本</th><th>行情日</th><th>市值</th><th>估值状态</th></tr></thead><tbody>{simulationPositions.map((position) => <tr key={position.instrument}><td><code>{position.instrument}</code></td><td>{position.position_side === "short" ? "模拟空头" : "多头"}</td><td>{position.quantity}</td><td>{position.available_quantity}</td><td>{Number(position.average_cost).toFixed(4)}</td><td>{position.market_date ?? "—"}</td><td>¥{Number(position.market_value).toFixed(2)}</td><td><span className={`state ${position.stale ? "failed" : "ready"}`}>{position.stale ? "stale" : "current"}</span></td></tr>)}</tbody></table>{selectedSimulation && !simulationPositions.length && <div className="empty">账户尚无持仓；等待下一次受治理信号进入模拟撮合。</div>}</div>
      {selectedSimulation && <section className="ledger-tables-grid">
        <article className="workspace-card"><div className="panel-heading"><div><p className="eyebrow">ORDER BOOK</p><h2>最近订单</h2></div><span>{simulationOrders.length} 条</span></div><div className="table-wrap compact-ledger-table"><table><thead><tr><th>时间</th><th>证券</th><th>方向</th><th>申请 / 成交</th><th>状态</th></tr></thead><tbody>{simulationOrders.slice(-12).reverse().map((order) => <tr key={order.id}><td>{shortTime(order.created_at)}</td><td><code>{order.instrument}</code></td><td>{({ buy: "买入", sell: "卖出", sell_short: "模拟卖空", buy_to_cover: "模拟回补" } as Record<string, string>)[order.side] ?? order.side}</td><td>{order.requested_quantity} / {order.filled_quantity}</td><td><span className={`state ${order.status === "filled" ? "ready" : order.status === "rejected" ? "failed" : "partial"}`}>{order.status}</span>{order.reject_reason && <small>{order.reject_reason}</small>}</td></tr>)}</tbody></table>{!simulationOrders.length && <div className="empty">尚无订单；系统不会为展示而生成虚假记录。</div>}</div></article>
        <article className="workspace-card"><div className="panel-heading"><div><p className="eyebrow">FILL LEDGER</p><h2>最近成交</h2></div><span>{simulationFills.length} 条</span></div><div className="table-wrap compact-ledger-table"><table><thead><tr><th>成交时间</th><th>证券</th><th>方向</th><th>数量 × 价格</th><th>费用</th></tr></thead><tbody>{simulationFills.slice(-12).reverse().map((fill) => <tr key={fill.id}><td>{shortTime(fill.executed_at)}</td><td><code>{fill.instrument}</code></td><td>{({ buy: "买入", sell: "卖出", sell_short: "模拟卖空", buy_to_cover: "模拟回补" } as Record<string, string>)[fill.side] ?? fill.side}</td><td>{fill.quantity} × {Number(fill.price).toFixed(3)}</td><td>{money(Number(fill.fee))}</td></tr>)}</tbody></table>{!simulationFills.length && <div className="empty">尚无模拟撮合成交。</div>}</div></article>
      </section>}
      {selectedSimulation && <section className="workspace-card simulation-audit-timeline"><div className="panel-heading"><div><p className="eyebrow">IMMUTABLE AUDIT TRAIL</p><h2>运行事件与阻断原因</h2></div><span>{simulationEvents.length} 条</span></div><div className="audit-event-list">{simulationEvents.slice(-20).reverse().map((event) => <div key={event.id} className={`audit-event ${event.severity}`}><span>{event.trade_date}</span><strong>{event.event_type}</strong><p>{event.instrument ? `${event.instrument} · ` : ""}{event.reason}</p><time>{shortTime(event.created_at)}</time></div>)}{!simulationEvents.length && <div className="empty">暂无异常或审计事件。</div>}</div></section>}
    </section>
    {snapshot && <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">COST ASSUMPTIONS</p><h2>统一成本假设</h2></div></div><div className="metric-strip portfolio-metrics"><div><span>买入费率</span><strong>{pct(snapshot.cost_model.buy_commission_rate)}</strong></div><div><span>卖出费率</span><strong>{pct(snapshot.cost_model.sell_commission_rate)}</strong></div><div><span>固定滑点</span><strong>{pct(snapshot.cost_model.fixed_slippage_rate)}</strong></div><div><span>参与率上限</span><strong>{pct(snapshot.cost_model.max_volume_participation)}</strong></div><div><span>最低佣金</span><strong>¥{snapshot.cost_model.min_commission}</strong></div></div></section>}
  </div>;
}
