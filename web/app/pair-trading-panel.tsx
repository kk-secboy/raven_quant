"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";

type QlibDataset = { name: string; ready: boolean; reproducible: boolean; lineage_verified?: boolean; start_date?: string | null; end_date?: string | null; trading_days: number };
type Snapshot = { name: string; invalid?: boolean; lineage_id?: string | null; datasets?: Record<string, { rows?: number }> };
type PairVersion = {
  id: string; version: number; status: string; strategy_type: string; created_by: string;
  config: Record<string, number | string>;
  pair?: { leg_y: string; leg_x: string; asset_class: string; shorting_mode: string } | null;
};
type PairStrategy = { id: string; name: string; description: string; status: string; versions: PairVersion[] };
type PairBacktest = {
  id: string; strategy_version_id: string; dataset: string; execution_dataset?: string | null;
  status: string; periods: { start: string; end: string }; metrics?: Record<string, unknown> | null; error?: string | null;
};
type ReadinessCheck = { id: string; title: string; status: "pass" | "block"; evidence: string; remediation?: string | null };
type ReadinessProfile = { id: string; status: "ready" | "blocked"; passed: number; total: number; checks: ReadinessCheck[] };
type PairNav = {
  trade_date: string; nav: number; cash: number; daily_return: number; drawdown: number;
  gross_exposure: number; net_exposure: number; turnover: number; fees: number; borrow_cost: number;
  zscore: number; correlation: number; cointegration_pvalue: number;
};
type PairBatch = { id: string; as_of_date: string; trade_date?: string | null; status: string; error?: string | null };
type PairOrder = { id: string; leg: string; instrument: string; side: string; requested_quantity: number; target_quantity: number; status: string; reason?: string | null };
type PairRiskEvent = { id: number; rule: string; severity: string; event_type: string; observed?: number | null; limit_value?: number | null; status: string; resolution_reason?: string | null };
type PairReview = { id: string; trade_date: string; status: string; summary: { action: string; reason?: string | null; rejection?: string | null; orders: number; fills: number; fees: number; borrow_cost: number; daily_return: number; drawdown: number; gross_exposure: number; net_exposure: number; zscore: number } };
type PairPortfolio = {
  id: string; name: string; strategy_version_id: string; dataset: string; execution_snapshot: string;
  dataset_roll_policy: "pinned" | "latest_compatible"; execution_roll_policy: "pinned" | "latest_compatible";
  minute_dataset: string; shortability_dataset: string; status: string; initial_cash: number; cash: number; nav: number;
  position_direction: number; quantity_y: number; quantity_x: number; holding_days: number;
  nav_history: PairNav[]; batches: PairBatch[]; orders: PairOrder[]; risk_events: PairRiskEvent[]; reviews: PairReview[];
  version: PairVersion;
};
type Schedule = { id: string; name: string; kind: string; status: string; desired_status: string; payload: { pair_portfolio_id?: string } };

const pct = (value: unknown) => typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
const decimal = (value: unknown) => typeof value === "number" ? value.toFixed(3) : "—";
const count = (value: unknown) => typeof value === "number" ? new Intl.NumberFormat("zh-CN").format(value) : "—";
const money = (value: number | undefined) => new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 0 }).format(value ?? 0);

function errorText(body: { detail?: unknown }, fallback: string) {
  if (typeof body.detail === "string") return body.detail;
  if (body.detail && typeof body.detail === "object" && "message" in body.detail) return String((body.detail as { message: unknown }).message);
  return fallback;
}

export function PairTradingPanel({ api }: { api: string }) {
  const [view, setView] = useState("research");
  const [datasets, setDatasets] = useState<QlibDataset[]>([]);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [strategies, setStrategies] = useState<PairStrategy[]>([]);
  const [backtests, setBacktests] = useState<PairBacktest[]>([]);
  const [readiness, setReadiness] = useState<ReadinessProfile | null>(null);
  const [paperReadiness, setPaperReadiness] = useState<ReadinessProfile | null>(null);
  const [portfolios, setPortfolios] = useState<PairPortfolio[]>([]);
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [name, setName] = useState("沪深300 ETF 配对");
  const [description, setDescription] = useState("沪深300 ETF 之间的协整与 Kalman 动态价差策略，使用次日分钟窗口执行。");
  const [legY, setLegY] = useState("SH510300");
  const [legX, setLegX] = useState("SZ159919");
  const [assetClass, setAssetClass] = useState("etf");
  const [formationWindow, setFormationWindow] = useState(60);
  const [minCorrelation, setMinCorrelation] = useState(0.80);
  const [maxCointegrationPvalue, setMaxCointegrationPvalue] = useState(0.05);
  const [cointegrationRecheckDays, setCointegrationRecheckDays] = useState(5);
  const [entryZscore, setEntryZscore] = useState(1.50);
  const [exitZscore, setExitZscore] = useState(0.50);
  const [stopZscore, setStopZscore] = useState(3.00);
  const [maxHoldingDays, setMaxHoldingDays] = useState(5);
  const [initialCapital, setInitialCapital] = useState(5_000_000);
  const [pairGrossFraction, setPairGrossFraction] = useState(0.20);
  const [maxVolumeParticipation, setMaxVolumeParticipation] = useState(0.01);
  const [minCapacityFillRatio, setMinCapacityFillRatio] = useState(0.95);
  const [openCost, setOpenCost] = useState(0.0005);
  const [closeCost, setCloseCost] = useState(0.0015);
  const [minCommission, setMinCommission] = useState(5);
  const [slippage, setSlippage] = useState(0.0005);
  const [annualBorrowRate, setAnnualBorrowRate] = useState(0.08);
  const [lotSize, setLotSize] = useState(100);
  const [kalmanProcessVariance, setKalmanProcessVariance] = useState(0.00001);
  const [kalmanObservationVariance, setKalmanObservationVariance] = useState(0.001);
  const [minHedgeRatio, setMinHedgeRatio] = useState(0.10);
  const [maxHedgeRatio, setMaxHedgeRatio] = useState(10.0);
  const [maxDrawdown, setMaxDrawdown] = useState(0.10);
  const [minSharpeRatio, setMinSharpeRatio] = useState(0.0);
  const [minClosedTrades, setMinClosedTrades] = useState(5);
  const [minBacktestDays, setMinBacktestDays] = useState(252);
  const [minRollingCointegrationPassRate, setMinRollingCointegrationPassRate] = useState(0.80);
  const [minRobustnessPassRate, setMinRobustnessPassRate] = useState(0.75);
  const [selectedVersion, setSelectedVersion] = useState("");
  const [dataset, setDataset] = useState("");
  const [snapshot, setSnapshot] = useState("");
  const [minuteDataset, setMinuteDataset] = useState("");
  const [shortabilityDataset, setShortabilityDataset] = useState("");
  const [autoRollDaily, setAutoRollDaily] = useState(true);
  const [autoRollExecution, setAutoRollExecution] = useState(true);
  const [start, setStart] = useState("2024-01-01");
  const [end, setEnd] = useState(new Date().toISOString().slice(0, 10));
  const [approvalReason, setApprovalReason] = useState("");
  const [portfolioName, setPortfolioName] = useState("沪深300 ETF 双腿模拟组合");
  const [selectedPortfolioId, setSelectedPortfolioId] = useState("");
  const [signalDate, setSignalDate] = useState("");
  const [pairScheduleTime, setPairScheduleTime] = useState("15:30");
  const [pairScheduleMisfireGrace, setPairScheduleMisfireGrace] = useState(1800);
  const [riskResolution, setRiskResolution] = useState("");
  const [message, setMessage] = useState("");

  function chooseSnapshot(name: string, source = snapshots) {
    setSnapshot(name);
    const selected = source.find((item) => item.name === name);
    const names = Object.keys(selected?.datasets ?? {});
    if (!selected?.lineage_id) setAutoRollExecution(false);
    setMinuteDataset(names.find((item) => /1m|minute|分钟/i.test(item)) ?? "");
    setShortabilityDataset(names.find((item) => /margin_eligibility|shortable|shortability|融券/i.test(item)) ?? "");
  }

  async function load() {
    try {
      const responses = await Promise.all([
        apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" }),
        apiFetch(`${api}/api/snapshots`, { cache: "no-store" }),
        apiFetch(`${api}/api/strategies`, { cache: "no-store" }),
        apiFetch(`${api}/api/backtests`, { cache: "no-store" }),
        apiFetch(`${api}/api/operations/readiness`, { cache: "no-store" }),
        apiFetch(`${api}/api/pair-portfolios`, { cache: "no-store" }),
        apiFetch(`${api}/api/schedules`, { cache: "no-store" }),
      ]);
      if (responses.some((response) => !response.ok)) throw new Error("API unavailable");
      const nextDatasets: QlibDataset[] = await responses[0].json();
      const nextSnapshots: Snapshot[] = await responses[1].json();
      const allStrategies: PairStrategy[] = await responses[2].json();
      const nextStrategies = allStrategies.filter((item) => item.versions.some((version) => version.strategy_type === "pair"));
      const readinessBody: { profiles: ReadinessProfile[] } = await responses[4].json();
      const nextPortfolios: PairPortfolio[] = await responses[5].json();
      const nextSchedules: Schedule[] = await responses[6].json();
      setDatasets(nextDatasets);
      const validSnapshots = nextSnapshots.filter((item) => !item.invalid);
      setSnapshots(validSnapshots);
      setStrategies(nextStrategies);
      setBacktests(await responses[3].json());
      setReadiness(readinessBody.profiles.find((item) => item.id === "pair_research") ?? null);
      setPaperReadiness(readinessBody.profiles.find((item) => item.id === "pair_paper") ?? null);
      setPortfolios(nextPortfolios);
      setSchedules(nextSchedules.filter((item) => item.kind === "pair_paper_rebalance"));
      if (!dataset) {
        const eligible = nextDatasets.find((item) => item.ready && item.reproducible);
        if (eligible) {
          setDataset(eligible.name);
          if (!eligible.lineage_verified) setAutoRollDaily(false);
          if (eligible.start_date) setStart(eligible.start_date);
          if (eligible.end_date) { setEnd(eligible.end_date); if (!signalDate) setSignalDate(eligible.end_date); }
        }
      }
      if (!snapshot && validSnapshots.length) chooseSnapshot(validSnapshots[0].name, validSnapshots);
      if (!selectedVersion && nextStrategies.length) setSelectedVersion(nextStrategies[0].versions[0]?.id ?? "");
      if (!selectedPortfolioId && nextPortfolios.length) setSelectedPortfolioId(nextPortfolios[0].id);
    } catch {
      setMessage("无法读取配对交易控制面，请确认 Python 后端正在运行。");
    }
  }

  useEffect(() => {
    const initial = window.setTimeout(load, 0);
    const timer = window.setInterval(load, 8000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const versions = strategies.flatMap((strategy) => strategy.versions.filter((version) => version.strategy_type === "pair").map((version) => ({ strategy, version })));
  const current = versions.find((item) => item.version.id === selectedVersion);
  const currentBacktest = backtests.find((item) => item.strategy_version_id === selectedVersion);
  const active = useMemo(() => backtests.some((item) => versions.some(({ version }) => version.id === item.strategy_version_id) && ["queued", "running"].includes(item.status)), [backtests, versions]);
  const snapshotDatasets = Object.keys(snapshots.find((item) => item.name === snapshot)?.datasets ?? {});
  const metrics = currentBacktest?.metrics ?? {};
  const evidence = typeof metrics.initial_pair_evidence === "object" && metrics.initial_pair_evidence !== null ? metrics.initial_pair_evidence as Record<string, unknown> : {};
  const robustness = typeof metrics.robustness === "object" && metrics.robustness !== null ? metrics.robustness as Record<string, unknown> : {};
  const selectedPortfolio = portfolios.find((item) => item.id === selectedPortfolioId) ?? portfolios[0];
  const latestPairNav = selectedPortfolio?.nav_history[0];
  const latestPairReview = selectedPortfolio?.reviews[0];
  const activePairBatch = selectedPortfolio?.batches.some((item) => ["queued", "running"].includes(item.status));
  const selectedSchedule = schedules.find((item) => item.payload.pair_portfolio_id === selectedPortfolio?.id);
  const validZscoreThresholds = exitZscore < entryZscore && entryZscore < stopZscore;
  const validHedgeRatios = minHedgeRatio < maxHedgeRatio;

  function config() {
    return {
      formation_window: formationWindow, min_correlation: minCorrelation,
      max_cointegration_pvalue: maxCointegrationPvalue,
      cointegration_recheck_days: cointegrationRecheckDays, entry_zscore: entryZscore,
      exit_zscore: exitZscore, stop_zscore: stopZscore, max_holding_days: maxHoldingDays,
      initial_capital: initialCapital, pair_gross_fraction: pairGrossFraction,
      max_volume_participation: maxVolumeParticipation,
      min_capacity_fill_ratio: minCapacityFillRatio,
      open_cost: openCost, close_cost: closeCost, min_commission: minCommission,
      slippage, annual_borrow_rate: annualBorrowRate, lot_size: lotSize,
      kalman_process_variance: kalmanProcessVariance,
      kalman_observation_variance: kalmanObservationVariance,
      min_hedge_ratio: minHedgeRatio, max_hedge_ratio: maxHedgeRatio,
      max_drawdown: maxDrawdown, min_sharpe_ratio: minSharpeRatio,
      min_closed_trades: minClosedTrades, min_backtest_days: minBacktestDays,
      min_rolling_cointegration_pass_rate: minRollingCointegrationPassRate,
      min_robustness_pass_rate: minRobustnessPassRate,
    };
  }

  async function createStrategy(event: FormEvent) {
    event.preventDefault();
    if (!validZscoreThresholds || !validHedgeRatios) {
      setMessage("参数关系无效：Z 值必须满足离场 < 入场 < 止损，对冲比下限必须小于上限。");
      return;
    }
    const response = await apiFetch(`${api}/api/pair-strategies`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description, leg_y: legY, leg_x: legX, asset_class: assetClass, shorting_mode: "margin_borrow", config: config(), actor: "pair-researcher" }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(errorText(body, "配对策略创建失败")); return; }
    setSelectedVersion(body.versions[0].id);
    setMessage(`配对策略 ${body.name} v1 已创建；审批前必须完成分钟级原生回测。`);
    await load();
  }

  async function createNextVersion() {
    if (!current) return;
    if (!validZscoreThresholds || !validHedgeRatios) {
      setMessage("参数关系无效：Z 值必须满足离场 < 入场 < 止损，对冲比下限必须小于上限。");
      return;
    }
    const response = await apiFetch(`${api}/api/pair-strategies/${current.strategy.id}/versions`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ leg_y: legY, leg_x: legX, asset_class: assetClass, shorting_mode: "margin_borrow", config: config(), actor: "pair-researcher" }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(errorText(body, "配对策略新版本创建失败")); return; }
    setSelectedVersion(body.id);
    setMessage(`配对策略 v${body.version} 已创建，旧回测证据不会继承。`);
    await load();
  }

  async function runBacktest(event: FormEvent) {
    event.preventDefault();
    if (!selectedVersion) { setMessage("请先创建或选择配对策略版本。"); return; }
    if (!snapshot || !minuteDataset || !shortabilityDataset) { setMessage("执行快照必须同时包含分钟行情和独立融券资格数据集。"); return; }
    const response = await apiFetch(`${api}/api/strategy-versions/${selectedVersion}/pair-backtests`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, execution_snapshot: snapshot, minute_dataset: minuteDataset, shortability_dataset: shortabilityDataset, start, end }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(errorText(body, "配对回测创建失败")); return; }
    setMessage(`配对回测 ${body.id.slice(0, 8)} 已进入 Worker，双腿会按原子成交规则执行。`);
    await load();
  }

  async function approve(event: FormEvent) {
    event.preventDefault();
    if (!selectedVersion) return;
    const response = await apiFetch(`${api}/api/strategy-versions/${selectedVersion}/approve`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actor: "pair-risk-reviewer", reason: approvalReason }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(errorText(body, "配对策略审批失败")); return; }
    setApprovalReason("");
    setMessage(`配对策略 v${body.version} 已通过第二人风险审批。`);
    await load();
  }

  async function createPairPortfolio(event: FormEvent) {
    event.preventDefault();
    if (!current || current.version.status !== "approved") { setMessage("只有完成独立审批的配对版本才能创建双腿模拟组合。"); return; }
    const response = await apiFetch(`${api}/api/pair-portfolios`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: portfolioName, strategy_version_id: current.version.id, dataset,
        execution_snapshot: snapshot, minute_dataset: minuteDataset,
        shortability_dataset: shortabilityDataset, initial_cash: initialCapital,
        dataset_roll_policy: autoRollDaily && datasets.find((item) => item.name === dataset)?.lineage_verified ? "latest_compatible" : "pinned",
        execution_roll_policy: autoRollExecution && snapshots.find((item) => item.name === snapshot)?.lineage_id ? "latest_compatible" : "pinned",
        actor: "pair-paper-operator",
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(errorText(body, "双腿模拟组合创建失败")); return; }
    setSelectedPortfolioId(body.id);
    setMessage(`专用双腿价差账本 ${body.name} 已创建，和多头模拟组合完全隔离。`);
    await load();
  }

  async function runPairPaper(event: FormEvent) {
    event.preventDefault();
    if (!selectedPortfolio) return;
    const response = await apiFetch(`${api}/api/pair-portfolios/${selectedPortfolio.id}/rebalance`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ as_of_date: signalDate }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(errorText(body, "配对模拟批次创建失败")); return; }
    setMessage(`配对批次 ${body.id.slice(0, 8)} 已进入 Worker；两腿将原子成交并写入独立账本。`);
    await load();
  }

  async function createPairSchedule() {
    if (!selectedPortfolio) return;
    const response = await apiFetch(`${api}/api/schedules`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: `${selectedPortfolio.name} 日终调度`, kind: "pair_paper_rebalance",
        timezone: "Asia/Shanghai", run_time: `${pairScheduleTime}:00`, trading_days_only: true,
        payload: { pair_portfolio_id: selectedPortfolio.id }, actor: "pair-paper-operator",
        misfire_grace_seconds: pairScheduleMisfireGrace,
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(errorText(body, "配对日终调度创建失败")); return; }
    setMessage(`交易日 ${pairScheduleTime} 的配对模拟调度已创建；风险待平仓时调度不可被暂停。`);
    await load();
  }

  async function changePairStatus(status: "active" | "paused") {
    if (!selectedPortfolio) return;
    const response = await apiFetch(`${api}/api/pair-portfolios/${selectedPortfolio.id}/status`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(errorText(body, "配对组合状态更新失败")); return; }
    setMessage(status === "paused" ? "配对组合已暂停。" : "配对组合已恢复；未解决的 critical 风险会阻止恢复。 ");
    await load();
  }

  async function actOnPairRisk(item: PairRiskEvent, action: "acknowledge" | "resolve") {
    if (!selectedPortfolio) return;
    const response = await apiFetch(`${api}/api/pair-portfolios/${selectedPortfolio.id}/risk-events/${item.id}/${action}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actor: "pair-risk-operator", ...(action === "resolve" ? { reason: riskResolution } : {}) }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(errorText(body, "配对风险事件处置失败")); return; }
    if (action === "resolve") setRiskResolution("");
    setMessage(action === "acknowledge" ? "风险事件已确认并保留审计记录。" : "风险事件已处置；只有双腿平仓且无活动批次时才能完成处置。");
    await load();
  }

  return <>
    {message && <div className="notice">{message}</div>}
    <div className="page-tabs" role="tablist" aria-label="配对交易工作区">
      {[["research", "策略研究与回测"], ["paper", "模拟执行"], ["ledger", "订单与风险"]].map(([value, label]) => <button type="button" role="tab" aria-selected={view === value} className={view === value ? "active" : ""} onClick={() => setView(value)} key={value}>{label}</button>)}
    </div>
    {view === "research" && <>
    <section className="pair-readiness">
      <article className={readiness?.status === "ready" ? "ready" : "blocked"}><span>PAIR RESEARCH READINESS</span><strong>{readiness?.status === "ready" ? "配对研究已就绪" : "配对研究仍有阻断项"}</strong><small>{readiness ? `${readiness.passed}/${readiness.total} 项通过` : "等待正式验收结果"}</small></article>
      <div className="pair-checks">{readiness?.checks.map((item) => <div className={item.status} key={item.id}><b>{item.status === "pass" ? "✓" : "!"}</b><span><strong>{item.title}</strong><small>{item.evidence}</small>{item.remediation && <em>{item.remediation}</em>}</span></div>)}{!readiness?.checks.length && <div className="block"><b>!</b><span><strong>尚无验收证据</strong><small>需要日线 Qlib、分钟快照、融券资格和已审批配对版本。</small></span></div>}</div>
    </section>

    <section className="backtest-hero pair-hero">
      <form className="strategy-builder" onSubmit={createStrategy}>
        <div className="card-heading"><div><span>统计套利 · 不可变版本</span><strong>ETF 配对与 Kalman 价差</strong></div><span className="status-chip">卫星策略</span></div>
        <div className="form-row"><label>策略名称<input value={name} onChange={(event) => setName(event.target.value)} /></label><label>资产类型<select value={assetClass} onChange={(event) => setAssetClass(event.target.value)}><option value="etf">ETF（首选）</option><option value="stock">股票</option><option value="mixed">混合</option></select></label></div>
        <label>策略说明<textarea value={description} onChange={(event) => setDescription(event.target.value)} /></label>
        <div className="form-row"><label>价差 Y 腿<input value={legY} onChange={(event) => setLegY(event.target.value.toUpperCase())} /></label><label>对冲 X 腿<input value={legX} onChange={(event) => setLegX(event.target.value.toUpperCase())} /></label></div>
        <div className="risk-grid pair-risk-grid">
          <label>形成期（日）<input type="number" min="20" max="252" value={formationWindow} onChange={(event) => setFormationWindow(Number(event.target.value))} /></label>
          <label>最低相关性<input type="number" min="0" max="1" step="0.01" value={minCorrelation} onChange={(event) => setMinCorrelation(Number(event.target.value))} /></label>
          <label>协整 P 值上限<input type="number" min="0.001" max="1" step="0.01" value={maxCointegrationPvalue} onChange={(event) => setMaxCointegrationPvalue(Number(event.target.value))} /></label>
          <label>协整复检（日）<input type="number" min="1" max="63" step="1" value={cointegrationRecheckDays} onChange={(event) => setCointegrationRecheckDays(Number(event.target.value))} /></label>
          <label>入场 Z<input type="number" min="0.1" step="0.1" value={entryZscore} onChange={(event) => setEntryZscore(Number(event.target.value))} /></label>
          <label>离场 Z<input type="number" min="0" step="0.1" value={exitZscore} onChange={(event) => setExitZscore(Number(event.target.value))} /></label>
          <label>止损 Z<input type="number" min="0.2" step="0.1" value={stopZscore} onChange={(event) => setStopZscore(Number(event.target.value))} /></label>
          <label>最长持有（日）<input type="number" min="1" max="20" value={maxHoldingDays} onChange={(event) => setMaxHoldingDays(Number(event.target.value))} /></label>
          <label>初始资金<input type="number" min="100000" step="100000" value={initialCapital} onChange={(event) => setInitialCapital(Number(event.target.value))} /></label>
          <label>价差总敞口<input type="number" min="0.01" max="1" step="0.01" value={pairGrossFraction} onChange={(event) => setPairGrossFraction(Number(event.target.value))} /></label>
          <label>年化融券成本<input type="number" min="0" max="1" step="0.01" value={annualBorrowRate} onChange={(event) => setAnnualBorrowRate(Number(event.target.value))} /></label>
        </div>
        <details className="advanced-config">
          <summary>容量、成本、Kalman 与回测准入参数</summary>
          <div className="risk-grid pair-risk-grid">
            <label>成交量参与率（%）<input type="number" min="0.01" max="20" step="0.1" value={maxVolumeParticipation * 100} onChange={(event) => setMaxVolumeParticipation(Number(event.target.value) / 100)} /></label>
            <label>最低容量成交率（%）<input type="number" min="0.1" max="100" step="0.5" value={minCapacityFillRatio * 100} onChange={(event) => setMinCapacityFillRatio(Number(event.target.value) / 100)} /></label>
            <label>开仓成本（bp）<input type="number" min="0" max="200" step="1" value={openCost * 10000} onChange={(event) => setOpenCost(Number(event.target.value) / 10000)} /></label>
            <label>平仓成本（bp）<input type="number" min="0" max="200" step="1" value={closeCost * 10000} onChange={(event) => setCloseCost(Number(event.target.value) / 10000)} /></label>
            <label>最低佣金（元）<input type="number" min="0" max="1000" step="1" value={minCommission} onChange={(event) => setMinCommission(Number(event.target.value))} /></label>
            <label>滑点（bp）<input type="number" min="0" max="200" step="1" value={slippage * 10000} onChange={(event) => setSlippage(Number(event.target.value) / 10000)} /></label>
            <label>交易单位<input type="number" min="1" max="10000" step="1" value={lotSize} onChange={(event) => setLotSize(Number(event.target.value))} /></label>
            <label>Kalman 过程方差<input type="number" min="0.00000001" max="1" step="0.000001" value={kalmanProcessVariance} onChange={(event) => setKalmanProcessVariance(Number(event.target.value))} /></label>
            <label>Kalman 观测方差<input type="number" min="0.00000001" max="1" step="0.0001" value={kalmanObservationVariance} onChange={(event) => setKalmanObservationVariance(Number(event.target.value))} /></label>
            <label>对冲比下限<input type="number" min="0.01" max="100" step="0.1" value={minHedgeRatio} onChange={(event) => setMinHedgeRatio(Number(event.target.value))} /></label>
            <label>对冲比上限<input type="number" min="0.01" max="100" step="0.1" value={maxHedgeRatio} onChange={(event) => setMaxHedgeRatio(Number(event.target.value))} /></label>
            <label>最大回撤（%）<input type="number" min="0.1" max="50" step="0.5" value={maxDrawdown * 100} onChange={(event) => setMaxDrawdown(Number(event.target.value) / 100)} /></label>
            <label>最低 Sharpe<input type="number" min="-5" max="10" step="0.1" value={minSharpeRatio} onChange={(event) => setMinSharpeRatio(Number(event.target.value))} /></label>
            <label>最低闭合交易<input type="number" min="1" max="10000" step="1" value={minClosedTrades} onChange={(event) => setMinClosedTrades(Number(event.target.value))} /></label>
            <label>最低回测日数<input type="number" min="60" max="2520" step="1" value={minBacktestDays} onChange={(event) => setMinBacktestDays(Number(event.target.value))} /></label>
            <label>滚动协整通过率（%）<input type="number" min="0" max="100" step="1" value={minRollingCointegrationPassRate * 100} onChange={(event) => setMinRollingCointegrationPassRate(Number(event.target.value) / 100)} /></label>
            <label>稳健性通过率（%）<input type="number" min="0" max="100" step="1" value={minRobustnessPassRate * 100} onChange={(event) => setMinRobustnessPassRate(Number(event.target.value) / 100)} /></label>
          </div>
        </details>
        {(!validZscoreThresholds || !validHedgeRatios) && <small className="danger-text">离场 Z &lt; 入场 Z &lt; 止损 Z；对冲比下限 &lt; 上限。</small>}
        <div className="execution-note"><b>文档规则已经固化</b><span>相关性与协整筛选，Kalman 动态对冲，±1.5σ 入场、±0.5σ 离场、±3σ 止损。</span><span>信号只使用当日收盘前信息，订单在下一交易日允许的分钟窗口执行。</span><span>任一腿停牌、涨跌停、容量不足或无融券资格时，两腿均不成交。</span></div>
        <button className="primary" disabled={!validZscoreThresholds || !validHedgeRatios}>创建配对策略 v1</button>
      </form>

      <form className="backtest-launcher" onSubmit={runBacktest}>
        <div className="card-heading"><div><span>分钟执行 · 原生引擎</span><strong>回测与容量验证</strong></div><span className="status-chip">{active ? "运行中" : "待命"}</span></div>
        <label>配对策略版本<select value={selectedVersion} onChange={(event) => setSelectedVersion(event.target.value)}><option value="">请选择</option>{versions.map(({ strategy, version }) => <option key={version.id} value={version.id}>{strategy.name} · v{version.version} · {version.status}</option>)}</select></label>
        <div className="form-row"><label>日线 Qlib 数据集<select value={dataset} onChange={(event) => { const name = event.target.value; setDataset(name); if (!datasets.find((item) => item.name === name)?.lineage_verified) setAutoRollDaily(false); }}><option value="">请选择</option>{datasets.filter((item) => item.ready && item.reproducible).map((item) => <option key={item.name} value={item.name}>{item.name} · {item.trading_days} 日</option>)}</select></label><label>不可变执行快照<select value={snapshot} onChange={(event) => chooseSnapshot(event.target.value)}><option value="">请选择</option>{snapshots.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select></label></div>
        <label>分钟行情数据集<select value={minuteDataset} onChange={(event) => setMinuteDataset(event.target.value)}><option value="">请选择</option>{snapshotDatasets.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
        <label>融券资格数据集<select value={shortabilityDataset} onChange={(event) => setShortabilityDataset(event.target.value)}><option value="">请选择</option>{snapshotDatasets.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
        <div className="form-row"><label>开始日期<input type="date" value={start} onChange={(event) => setStart(event.target.value)} /></label><label>结束日期<input type="date" value={end} onChange={(event) => setEnd(event.target.value)} /></label></div>
        <div className="execution-note"><b>两个独立证据源</b><span>分钟行情用于 VWAP、涨跌停、停牌与成交量容量。</span><span>融券资格必须逐日逐证券保存，禁止从融资融券成交明细反推。</span></div>
        <button className="primary" disabled={active || !selectedVersion || !dataset || !minuteDataset || !shortabilityDataset}>排队运行配对回测</button>
        <button type="button" disabled={!current || !validZscoreThresholds || !validHedgeRatios} onClick={createNextVersion}>按当前参数创建新版本</button>
      </form>
    </section>

    <section className="data-panel pair-metrics">
      <div className="panel-heading"><div><p className="eyebrow">PAIR EVIDENCE / RISK GATES</p><h2>最新回测证据</h2></div><span>{currentBacktest ? `${currentBacktest.status} · ${currentBacktest.id.slice(0, 8)}` : "尚无回测"}</span></div>
      <div className="metric-strip pair-metric-strip"><div><span>相关系数</span><strong>{decimal(evidence.correlation)}</strong></div><div><span>协整 P 值</span><strong>{decimal(evidence.cointegration_pvalue)}</strong></div><div><span>Sharpe</span><strong>{decimal(metrics.sharpe_ratio)}</strong></div><div><span>最大回撤</span><strong>{pct(metrics.max_drawdown)}</strong></div><div><span>闭合交易</span><strong>{count(metrics.closed_trade_count)}</strong></div><div><span>滚动协整通过率</span><strong>{pct(metrics.rolling_cointegration_pass_rate)}</strong></div><div><span>容量成交率</span><strong>{pct(metrics.capacity_fill_ratio)}</strong></div><div><span>稳健性通过率</span><strong>{pct(robustness.pass_rate ?? metrics.pair_robustness_pass_rate)}</strong></div></div>
      {currentBacktest?.error && <div className="notice danger-notice">{currentBacktest.error}</div>}
      {!currentBacktest && <div className="empty compact">选择策略版本并提交分钟级回测后，这里会显示协整、收益风险、容量和稳健性证据。</div>}
    </section>

    <form className="approval-panel" onSubmit={approve}><div><span>SECOND-PERSON APPROVAL</span><strong>风险审批与不可变证据</strong><small>正式认证开启时，创建人必须退出并由另一名有 strategy:approve 权限的用户审批。</small></div><textarea value={approvalReason} onChange={(event) => setApprovalReason(event.target.value)} placeholder="至少 10 个字符：说明协整稳定性、容量、融券成本和压力场景为何可接受" /><button className="primary" disabled={!selectedVersion || currentBacktest?.status !== "succeeded" || approvalReason.trim().length < 10}>批准该配对版本</button></form>
    </>}
    {view === "paper" && <>
    <div className="notice pair-ledger-notice"><b>专用双腿价差账本已接入。</b> 配对组合与现有多头模拟组合完全隔离；订单、成交、融券成本、净值、风险事件与复盘均在同一 PostgreSQL 事务中原子落账。</div>

    <section className="pair-paper-hero">
      <form className="portfolio-launcher" onSubmit={createPairPortfolio}>
        <label className="policy-toggle"><input type="checkbox" checked={autoRollDaily} disabled={!datasets.find((item) => item.name === dataset)?.lineage_verified} onChange={(event) => setAutoRollDaily(event.target.checked)} /><span>日线自动推进到同血缘、追加式验证的新 Qlib 快照</span></label>
        <label className="policy-toggle"><input type="checkbox" checked={autoRollExecution} disabled={!snapshots.find((item) => item.name === snapshot)?.lineage_id} onChange={(event) => setAutoRollExecution(event.target.checked)} /><span>分钟与融券资格自动推进到同血缘、追加式验证的新执行快照</span></label>
        <div className="card-heading"><div><span>PAIR PAPER LEDGER</span><strong>创建双腿模拟组合</strong></div><span className="status-chip verified">ATOMIC LEGS</span></div>
        <label>账本名称<input value={portfolioName} onChange={(event) => setPortfolioName(event.target.value)} /></label>
        <label>已审批配对版本<select value={selectedVersion} onChange={(event) => setSelectedVersion(event.target.value)}><option value="">请选择</option>{versions.filter(({ version }) => version.status === "approved").map(({ strategy, version }) => <option key={version.id} value={version.id}>{strategy.name} · v{version.version}</option>)}</select></label>
        <div className="pair-evidence-summary"><span>日线：{dataset || "未选择"}</span><span>快照：{snapshot || "未选择"}</span><span>分钟：{minuteDataset || "未选择"}</span><span>融券：{shortabilityDataset || "未选择"}</span></div>
        <label>初始资金<input type="number" min="100000" step="100000" value={initialCapital} onChange={(event) => setInitialCapital(Number(event.target.value))} /></label>
        <button className="primary" disabled={!current || current.version.status !== "approved" || !dataset || !snapshot || !minuteDataset || !shortabilityDataset || portfolioName.trim().length < 3}>创建专用双腿账本</button>
      </form>

      <article className="portfolio-summary">
        {selectedPortfolio && <small className="muted">日线策略：{selectedPortfolio.dataset_roll_policy === "latest_compatible" ? "同血缘自动推进" : "固定快照"} · 执行策略：{selectedPortfolio.execution_roll_policy === "latest_compatible" ? "同血缘自动推进" : "固定快照"}</small>}
        <div className="card-heading"><div><span>受治理价差组合</span><strong>{selectedPortfolio?.name ?? "等待创建账本"}</strong></div>{selectedPortfolio && <span className={`state ${selectedPortfolio.status === "active" ? "ready" : "partial"}`}>{selectedPortfolio.status}</span>}</div>
        <label>当前账本<select value={selectedPortfolio?.id ?? ""} onChange={(event) => setSelectedPortfolioId(event.target.value)}>{portfolios.length ? portfolios.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.status}</option>) : <option value="">尚无双腿账本</option>}</select></label>
        <div className="portfolio-kpis"><div><span>净值</span><strong>{money(selectedPortfolio?.nav)}</strong></div><div><span>现金</span><strong>{money(selectedPortfolio?.cash)}</strong></div><div><span>Y / X 数量</span><strong>{selectedPortfolio ? `${selectedPortfolio.quantity_y.toLocaleString()} / ${selectedPortfolio.quantity_x.toLocaleString()}` : "—"}</strong></div><div><span>回撤</span><strong>{pct(latestPairNav?.drawdown)}</strong></div></div>
        <form className="rebalance-form" onSubmit={runPairPaper}><label>信号日期<input type="date" value={signalDate} onChange={(event) => setSignalDate(event.target.value)} /></label><button className="primary" disabled={!selectedPortfolio || !["active", "liquidation_pending"].includes(selectedPortfolio.status) || !signalDate || activePairBatch}>{selectedPortfolio?.status === "liquidation_pending" ? "执行强制平仓" : "运行配对日终批次"}</button></form>
        <div className="form-row"><label>日终调度时间<input type="time" min="15:10" value={pairScheduleTime} onChange={(event) => setPairScheduleTime(event.target.value)} /></label><label>错过宽限（秒）<input type="number" min="60" max="86400" step="60" value={pairScheduleMisfireGrace} onChange={(event) => setPairScheduleMisfireGrace(Number(event.target.value))} /></label></div>
        <div className="pair-ledger-actions">
          {selectedPortfolio && <button type="button" className="secondary-action" disabled={selectedPortfolio.status === "liquidation_pending"} onClick={() => changePairStatus(selectedPortfolio.status === "active" ? "paused" : "active")}>{selectedPortfolio.status === "liquidation_pending" ? "待完成风险平仓" : selectedPortfolio.status === "active" ? "暂停组合" : "恢复组合"}</button>}
          <button type="button" className="secondary-action" disabled={!selectedPortfolio || Boolean(selectedSchedule)} onClick={createPairSchedule}>{selectedSchedule ? `日终调度：${selectedSchedule.status}` : `创建 ${pairScheduleTime} 日终调度`}</button>
        </div>
        <small className="muted">下一交易日分钟窗口执行 · 两腿原子成交 · 逐日融券资格 · 状态哈希防止过期 Worker 结果覆盖账本</small>
      </article>
    </section>

    <section className="pair-paper-readiness">
      <div><span>PAIR PAPER READINESS</span><strong>{paperReadiness?.status === "ready" ? "配对模拟盘已就绪" : "配对模拟盘仍有阻断项"}</strong><small>{paperReadiness ? `${paperReadiness.passed}/${paperReadiness.total} 项通过` : "等待后端验收"}</small></div>
      <div className="pair-paper-checks">{paperReadiness?.checks.slice(-4).map((item) => <article className={item.status} key={item.id}><b>{item.status === "pass" ? "✓" : "!"}</b><span><strong>{item.title}</strong><small>{item.evidence}</small></span></article>)}</div>
    </section>

    <section className="metric-strip pair-ledger-metrics"><div><span>初始资金</span><strong>{money(selectedPortfolio?.initial_cash)}</strong></div><div><span>累计收益</span><strong>{selectedPortfolio ? pct(selectedPortfolio.nav / selectedPortfolio.initial_cash - 1) : "—"}</strong></div><div><span>总敞口</span><strong>{pct(latestPairNav?.gross_exposure)}</strong></div><div><span>净敞口</span><strong>{pct(latestPairNav?.net_exposure)}</strong></div><div><span>融券成本</span><strong>{money(latestPairNav?.borrow_cost)}</strong></div><div><span>最近批次</span><strong>{selectedPortfolio?.batches[0]?.status ?? "—"}</strong></div></section>
    </>}

    {view === "ledger" && <>
    {latestPairReview && <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">PAIR POST-TRADE REVIEW</p><h2>双腿盘后复盘 · {latestPairReview.trade_date}</h2></div><span className="state ready">{latestPairReview.status}</span></div><div className="metric-strip pair-ledger-metrics"><div><span>动作</span><strong>{latestPairReview.summary.action}</strong></div><div><span>成交腿数</span><strong>{latestPairReview.summary.fills} / {latestPairReview.summary.orders}</strong></div><div><span>当日收益</span><strong>{pct(latestPairReview.summary.daily_return)}</strong></div><div><span>费用</span><strong>{money(latestPairReview.summary.fees)}</strong></div><div><span>融券成本</span><strong>{money(latestPairReview.summary.borrow_cost)}</strong></div><div><span>Z 值</span><strong>{decimal(latestPairReview.summary.zscore)}</strong></div></div>{latestPairReview.summary.rejection && <div className="notice danger-notice">原子拒单：{latestPairReview.summary.rejection}</div>}</section>}

    <section className="portfolio-lower pair-ledger-lower">
      <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">PAIR ORDER LEDGER</p><h2>双腿订单与批次</h2></div><span>{selectedPortfolio?.orders.length ?? 0} 笔订单</span></div><div className="table-wrap"><table className="portfolio-table"><thead><tr><th>腿</th><th>证券</th><th>方向</th><th>申请数量</th><th>目标数量</th><th>状态</th><th>原因</th></tr></thead><tbody>{selectedPortfolio?.orders.slice(0, 20).map((item) => <tr key={item.id}><td>{item.leg.toUpperCase()}</td><td><code>{item.instrument}</code></td><td>{item.side === "buy" ? "买入" : "卖出"}</td><td>{item.requested_quantity.toLocaleString()}</td><td>{item.target_quantity.toLocaleString()}</td><td><span className={`state ${item.status === "filled" ? "ready" : "failed"}`}>{item.status}</span></td><td>{item.reason ?? "—"}</td></tr>)}</tbody></table>{!selectedPortfolio?.orders.length && <div className="empty compact">尚无双腿订单；运行第一个信号批次后会写入两腿原子订单。</div>}</div></section>
      <section className="data-panel pair-risk-panel"><div className="panel-heading"><div><p className="eyebrow">PAIR RISK EVENTS</p><h2>风险事件</h2></div><span>{selectedPortfolio?.risk_events.length ?? 0} 条</span></div><label>处置结论<textarea value={riskResolution} onChange={(event) => setRiskResolution(event.target.value)} placeholder="说明平仓、敞口复核和恢复依据（至少 10 字）" /></label><div className="risk-list">{selectedPortfolio?.risk_events.map((item) => <article key={item.id}><span className={`job-state ${item.severity === "critical" ? "failed" : "running"}`} /><div><strong>{item.rule}</strong><small>{item.event_type} · 观察值 {decimal(item.observed)} / 阈值 {decimal(item.limit_value)}{item.resolution_reason ? ` · ${item.resolution_reason}` : ""}</small></div><span>{item.severity} · {item.status}</span><div>{item.status === "open" && <button type="button" className="inline-action" onClick={() => actOnPairRisk(item, "acknowledge")}>确认</button>}{item.status === "acknowledged" && <button type="button" className="inline-action" disabled={riskResolution.trim().length < 10} onClick={() => actOnPairRisk(item, "resolve")}>完成处置</button>}</div></article>)}{!selectedPortfolio?.risk_events.length && <div className="empty compact">当前没有风险事件；critical 事件会阻止组合恢复，待平仓状态不可暂停日终调度。</div>}</div></section>
    </section>
    </>}
  </>;
}
