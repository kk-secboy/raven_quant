"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";

type PairDefinition = {
  leg_y: string;
  leg_x: string;
  asset_class: "etf" | "stock" | "mixed";
};

type StrategyVersion = {
  id: string;
  version: number;
  status: string;
  strategy_type: string;
  pair?: PairDefinition | null;
  config: Record<string, unknown>;
};

type Strategy = {
  id: string;
  name: string;
  description: string;
  versions: StrategyVersion[];
};

type Backtest = {
  id: string;
  strategy_version_id: string;
  status: string;
  dataset: string;
  execution_dataset?: string | null;
  metrics?: Record<string, unknown> | null;
  error?: string | null;
};

type Dataset = {
  name: string;
  frequency: string;
  ready: boolean;
  reproducible: boolean;
  start_date?: string | null;
  end_date?: string | null;
};

type Snapshot = {
  name: string;
  frequency?: string;
  datasets?: Record<string, unknown>;
};

type Simulation = {
  id: string;
  name: string;
  status: string;
  source_type?: string;
  source_id?: string;
  execution_adapter?: string;
  execution_frequency?: string;
  cash: number;
  nav: number;
};

const pct = (value: unknown) => typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
const decimal = (value: unknown) => typeof value === "number" ? value.toFixed(3) : "—";
const PAIR_MINUTE_DATASETS = new Set<string>(["etf_1m", "liquid_stocks_1m"]);

export function PairSatellitePanel({ api }: { api: string }) {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [backtests, setBacktests] = useState<Backtest[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [simulations, setSimulations] = useState<Simulation[]>([]);
  const [selectedVersion, setSelectedVersion] = useState("");
  const [name, setName] = useState("沪深300 ETF 配对卫星");
  const [description, setDescription] = useState("使用协整、Kalman 对冲比、容量和融券资格约束的配对卫星策略。");
  const [legY, setLegY] = useState("SH510300");
  const [legX, setLegX] = useState("SZ159919");
  const [assetClass, setAssetClass] = useState<"etf" | "stock" | "mixed">("etf");
  const [dailyDataset, setDailyDataset] = useState("");
  const [executionSnapshot, setExecutionSnapshot] = useState("");
  const [minuteDataset, setMinuteDataset] = useState("");
  const [shortabilityDataset, setShortabilityDataset] = useState("");
  const [start, setStart] = useState("2024-01-01");
  const [end, setEnd] = useState(new Date().toISOString().slice(0, 10));
  const [approvalReason, setApprovalReason] = useState("");
  const [simulationName, setSimulationName] = useState("配对卫星模拟账户");
  const [simulationDataset, setSimulationDataset] = useState("");
  const [initialCash, setInitialCash] = useState(5_000_000);
  const [replaySimulationId, setReplaySimulationId] = useState("");
  const [replayBacktestId, setReplayBacktestId] = useState("");
  const [replayTradeDate, setReplayTradeDate] = useState("");
  const [message, setMessage] = useState("");

  async function load() {
    try {
      const responses = await Promise.all([
        apiFetch(`${api}/api/strategies`, { cache: "no-store" }),
        apiFetch(`${api}/api/backtests`, { cache: "no-store" }),
        apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" }),
        apiFetch(`${api}/api/snapshots`, { cache: "no-store" }),
        apiFetch(`${api}/api/simulation-portfolios`, { cache: "no-store" }),
      ]);
      if (responses.slice(0, 4).some((response) => !response.ok)) {
        throw new Error("pair governance APIs are unavailable");
      }
      const nextStrategies = await responses[0].json() as Strategy[];
      const nextBacktests = await responses[1].json() as Backtest[];
      const nextDatasets = await responses[2].json() as Dataset[];
      const nextSnapshots = await responses[3].json() as Snapshot[];
      const nextSimulations = responses[4].ok
        ? await responses[4].json() as Simulation[]
        : [];
      const pairStrategies = nextStrategies.filter((strategy) =>
        strategy.versions.some((version) => version.strategy_type === "pair"),
      );
      const pairVersions = pairStrategies.flatMap((strategy) => strategy.versions
        .filter((version) => version.strategy_type === "pair")
        .map((version) => ({ strategy, version })));
      const daily = nextDatasets.filter((item) =>
        item.ready && item.reproducible && item.frequency === "day",
      );
      const minute = nextDatasets.filter((item) =>
        item.ready && item.reproducible && item.frequency === "1min",
      );
      const executionSnapshots = nextSnapshots.filter((item) => {
        const keys = Object.keys(item.datasets ?? {});
        return item.frequency === "1min"
          && keys.includes("margin_eligibility")
          && keys.some((key) => PAIR_MINUTE_DATASETS.has(key));
      });
      setStrategies(pairStrategies);
      setBacktests(nextBacktests);
      setDatasets(nextDatasets);
      setSnapshots(executionSnapshots);
      const pairSimulations = nextSimulations.filter(
        (item) => item.execution_adapter === "pair",
      );
      setSimulations(pairSimulations);
      setReplaySimulationId((current) => current || pairSimulations[0]?.id || "");
      setSelectedVersion((current) => current || pairVersions[0]?.version.id || "");
      setDailyDataset((current) => current || daily[0]?.name || "");
      setSimulationDataset((current) => current || minute[0]?.name || "");
      setExecutionSnapshot((current) => current || executionSnapshots[0]?.name || "");
    } catch {
      setMessage("无法读取配对研究控制面，请确认后端已启动。");
    }
  }

  useEffect(() => {
    const initial = window.setTimeout(load, 0);
    const timer = window.setInterval(load, 8000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const versions = useMemo(() => strategies.flatMap((strategy) => strategy.versions
    .filter((version) => version.strategy_type === "pair")
    .map((version) => ({ strategy, version }))), [strategies]);
  const current = versions.find((item) => item.version.id === selectedVersion);
  const currentBacktest = backtests.find((item) => item.strategy_version_id === selectedVersion);
  const replayBacktests = backtests.filter(
    (item) =>
      item.strategy_version_id === selectedVersion
      && item.status === "succeeded",
  );
  const replaySimulations = simulations.filter(
    (item) => item.status === "active" && item.source_id === selectedVersion,
  );
  const activeReplaySimulationId = replaySimulations.some(
    (item) => item.id === replaySimulationId,
  ) ? replaySimulationId : "";
  const activeReplayBacktestId = replayBacktests.some(
    (item) => item.id === replayBacktestId,
  ) ? replayBacktestId : "";
  const selectedSnapshot = snapshots.find((item) => item.name === executionSnapshot);
  const snapshotDatasets = Object.keys(selectedSnapshot?.datasets ?? {});
  const minuteOptions = snapshotDatasets.filter((key) => PAIR_MINUTE_DATASETS.has(key));
  const shortabilityOptions: string[] = snapshotDatasets.filter(
    (key) => key === "margin_eligibility",
  );
  const activeMinuteDataset = minuteOptions.includes(minuteDataset)
    ? minuteDataset
    : minuteOptions[0] ?? "";
  const activeShortabilityDataset = shortabilityOptions.includes(shortabilityDataset)
    ? shortabilityDataset
    : shortabilityOptions[0] ?? "";
  const dailyDatasets = datasets.filter((item) => item.ready && item.reproducible && item.frequency === "day");
  const minuteQlibDatasets = datasets.filter((item) => item.ready && item.reproducible && item.frequency === "1min");
  const metrics = currentBacktest?.metrics ?? {};

  async function createPair(event: FormEvent) {
    event.preventDefault();
    const response = await apiFetch(`${api}/api/pair-strategies`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name, description, leg_y: legY.trim().toUpperCase(), leg_x: legX.trim().toUpperCase(),
        asset_class: assetClass, shorting_mode: "margin_borrow", actor: "pair-researcher",
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "配对策略创建失败"); return; }
    setSelectedVersion(body.versions[0].id);
    setMessage("配对研究版本已创建；完成日线信号、下一日分钟执行和融券证据回测后才能审批。");
    await load();
  }

  async function runBacktest(event: FormEvent) {
    event.preventDefault();
    if (!selectedVersion) return;
    const response = await apiFetch(`${api}/api/strategy-versions/${selectedVersion}/pair-backtests`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: dailyDataset,
        execution_snapshot: executionSnapshot,
        minute_dataset: activeMinuteDataset,
        shortability_dataset: activeShortabilityDataset,
        start,
        end,
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "配对回测创建失败"); return; }
    setMessage(`配对回测 ${body.id.slice(0, 8)} 已进入研究队列。`);
    await load();
  }

  async function approve(event: FormEvent) {
    event.preventDefault();
    if (!selectedVersion) return;
    const response = await apiFetch(`${api}/api/strategy-versions/${selectedVersion}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actor: "pair-risk-approver", reason: approvalReason }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "配对策略审批失败"); return; }
    setApprovalReason("");
    setMessage("配对卫星版本已审批；只有统一模拟盘可以消费该版本。");
    await load();
  }

  async function createSimulation(event: FormEvent) {
    event.preventDefault();
    if (!current || current.version.status !== "approved") return;
    const response = await apiFetch(`${api}/api/simulation-portfolios`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: simulationName,
        source_type: "strategy_version",
        source_id: current.version.id,
        execution_dataset: simulationDataset,
        execution_frequency: "1min",
        execution_adapter: "pair",
        initial_cash: initialCash,
        execution_algorithm: "vwap",
        slice_minutes: 5,
        max_slices: 24,
        max_participation: 0.01,
        actor: "simulation-operator",
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "配对模拟账户创建失败"); return; }
    setMessage("配对模拟账户已创建；缺少当日融券资格或任一腿证据时会整组拒绝。");
    await load();
  }

  async function setSimulationStatus(simulation: Simulation, status: "active" | "pause") {
    const response = await apiFetch(
      `${api}/api/simulation-portfolios/${simulation.id}/${status === "active" ? "activate" : "pause"}`,
      { method: "POST" },
    );
    const body = await response.json();
    setMessage(response.ok ? `模拟账户已${status === "active" ? "激活" : "暂停"}。` : body.detail ?? "状态更新失败");
    await load();
  }

  async function runSimulationReplay(event: FormEvent) {
    event.preventDefault();
    if (!activeReplaySimulationId || !activeReplayBacktestId || !replayTradeDate) return;
    const response = await apiFetch(
      `${api}/api/simulation-portfolios/${activeReplaySimulationId}/pair-replays`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          backtest_id: activeReplayBacktestId,
          trade_date: replayTradeDate,
          actor: "simulation-operator",
        }),
      },
    );
    const body = await response.json();
    if (!response.ok) {
      setMessage(body.detail ?? "配对受控回放创建失败");
      return;
    }
    setMessage(
      `配对回放 ${String(body.id).slice(0, 8)} 已排队；双腿、数量和借券率均由已审批制品推导。`,
    );
    await load();
  }

  return <div className="portfolio-page">
    {message && <div className="notice">{message}</div>}
    <section className="factor-intro">
      <div><p className="eyebrow">PAIR SATELLITE / GOVERNED ONLY</p><h2>配对卫星只有研究、审批和统一模拟盘</h2><p>日线形成信号，下一交易日用 1 分钟证据执行。缺少 Tushare 分钟数据、当日融券资格或任一腿成交条件时失败关闭；本页不连接券商，也没有独立模拟账本。</p></div>
      <div className="factor-flow"><span>协整研究</span><i>→</i><span>成本与容量</span><i>→</i><span>第二人审批</span><i>→</i><span>原子双腿模拟</span></div>
    </section>

    <section className="portfolio-hero">
      <form className="portfolio-launcher" onSubmit={createPair}>
        <div className="card-heading"><div><span>RESEARCH</span><strong>创建不可变配对版本</strong></div><span className="status-chip">卫星</span></div>
        <label>策略名称<input value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label>研究说明<textarea value={description} onChange={(event) => setDescription(event.target.value)} /></label>
        <div className="form-row"><label>Y 腿<input value={legY} onChange={(event) => setLegY(event.target.value)} /></label><label>X 腿<input value={legX} onChange={(event) => setLegX(event.target.value)} /></label></div>
        <label>资产类型<select value={assetClass} onChange={(event) => setAssetClass(event.target.value as typeof assetClass)}><option value="etf">ETF</option><option value="stock">股票</option><option value="mixed">混合</option></select></label>
        <button className="primary" disabled={name.length < 3 || description.length < 10 || legY === legX}>创建研究版本</button>
      </form>

      <form className="portfolio-summary" onSubmit={runBacktest}>
        <div className="card-heading"><div><span>BACKTEST</span><strong>配对证据与鲁棒性验证</strong></div><span className={`state ${currentBacktest?.status === "succeeded" ? "ready" : "partial"}`}>{currentBacktest?.status ?? "待运行"}</span></div>
        <label>配对版本<select value={selectedVersion} onChange={(event) => setSelectedVersion(event.target.value)}><option value="">尚无版本</option>{versions.map(({ strategy, version }) => <option key={version.id} value={version.id}>{strategy.name} · v{version.version} · {version.status}</option>)}</select></label>
        <label>日线 Qlib 数据<select value={dailyDataset} onChange={(event) => setDailyDataset(event.target.value)}>{dailyDatasets.map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
        <label>1 分钟执行快照<select value={executionSnapshot} onChange={(event) => setExecutionSnapshot(event.target.value)}><option value="">无完整快照</option>{snapshots.map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
        <div className="form-row"><label>分钟行情<select value={activeMinuteDataset} onChange={(event) => setMinuteDataset(event.target.value)}>{minuteOptions.map((item) => <option key={item}>{item}</option>)}</select></label><label>融券资格<select value={activeShortabilityDataset} onChange={(event) => setShortabilityDataset(event.target.value)}>{shortabilityOptions.map((item) => <option key={item}>{item}</option>)}</select></label></div>
        <div className="form-row"><label>开始日期<input type="date" value={start} onChange={(event) => setStart(event.target.value)} /></label><label>结束日期<input type="date" value={end} onChange={(event) => setEnd(event.target.value)} /></label></div>
        <button className="primary" disabled={!selectedVersion || !dailyDataset || !activeMinuteDataset || !activeShortabilityDataset || end <= start}>运行配对回测</button>
      </form>
    </section>

    <section className="metric-strip portfolio-metrics">
      <div><span>相关性</span><strong>{decimal((metrics.initial_pair_evidence as Record<string, unknown> | undefined)?.correlation)}</strong></div>
      <div><span>协整 p 值</span><strong>{decimal((metrics.initial_pair_evidence as Record<string, unknown> | undefined)?.cointegration_pvalue)}</strong></div>
      <div><span>Sharpe</span><strong>{decimal(metrics.sharpe_ratio)}</strong></div>
      <div><span>最大回撤</span><strong>{pct(metrics.max_drawdown)}</strong></div>
      <div><span>容量成交率</span><strong>{pct(metrics.capacity_fill_ratio)}</strong></div>
      <div><span>鲁棒性通过率</span><strong>{pct(metrics.pair_robustness_pass_rate)}</strong></div>
    </section>

    <section className="data-panel">
      <div className="panel-heading"><div><p className="eyebrow">APPROVAL</p><h2>第二人审批</h2></div><span>{current?.version.status ?? "尚未选择"}</span></div>
      <form className="approval-panel" onSubmit={approve}><textarea value={approvalReason} onChange={(event) => setApprovalReason(event.target.value)} placeholder="记录协整稳定性、双倍成本、容量、融券与双腿原子成交复核依据（至少 10 字）" /><button className="primary" disabled={currentBacktest?.status !== "succeeded" || approvalReason.trim().length < 10 || current?.version.status === "approved"}>批准配对卫星版本</button></form>
      {currentBacktest?.error && <div className="notice">{currentBacktest.error}</div>}
    </section>

    <section className="data-panel">
      <div className="panel-heading"><div><p className="eyebrow">UNIFIED SIMULATION</p><h2>原子双腿模拟账户</h2></div><span>{simulations.length} 个账户</span></div>
      <div className="portfolio-hero">
        <form className="portfolio-launcher" onSubmit={createSimulation}>
          <label>账户名称<input value={simulationName} onChange={(event) => setSimulationName(event.target.value)} /></label>
          <label>1 分钟 Qlib 执行数据<select value={simulationDataset} onChange={(event) => setSimulationDataset(event.target.value)}><option value="">无可用数据</option>{minuteQlibDatasets.map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
          <label>初始现金<input type="number" min="100000" step="100000" value={initialCash} onChange={(event) => setInitialCash(Number(event.target.value))} /></label>
          <button className="primary" disabled={current?.version.status !== "approved" || !simulationDataset || simulationName.length < 3}>创建统一模拟账户</button>
        </form>
        <div className="portfolio-summary">
          {simulations.map((simulation) => <article className="action-row" key={simulation.id}><span className={`task-status ${simulation.status}`}>{simulation.status}</span><div><strong>{simulation.name}</strong><small>1 分钟 · 原子双腿 · 现金 {Number(simulation.cash ?? 0).toFixed(2)} · NAV {Number(simulation.nav ?? 0).toFixed(2)}</small></div><button type="button" className="inline-action" onClick={() => setSimulationStatus(simulation, simulation.status === "active" ? "pause" : "active")}>{simulation.status === "active" ? "暂停" : "激活"}</button></article>)}
          {!simulations.length && <div className="empty compact">尚无配对模拟账户。不会回退到旧配对模拟路径。</div>}
        </div>
      </div>
      <form className="approval-panel" onSubmit={runSimulationReplay}>
        <label>已激活配对模拟账户
          <select
            value={activeReplaySimulationId}
            onChange={(event) => setReplaySimulationId(event.target.value)}
          >
            <option value="">请选择账户</option>
            {replaySimulations.map(
              (item) => <option key={item.id} value={item.id}>{item.name}</option>,
            )}
          </select>
        </label>
        <label>已成功正式回测制品
          <select
            value={activeReplayBacktestId}
            onChange={(event) => setReplayBacktestId(event.target.value)}
          >
            <option value="">请选择不可变制品</option>
            {replayBacktests.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id.slice(0, 8)} · {item.dataset}
              </option>
            ))}
          </select>
        </label>
        <label>制品内交易日
          <input
            type="date"
            value={replayTradeDate}
            onChange={(event) => setReplayTradeDate(event.target.value)}
          />
        </label>
        <p>
          客户端不能编辑双腿、方向、数量或借券率；后端只读取该回测制品和同源
          Tushare 快照中的 1 分钟与当日融券资格。
        </p>
        <button
          className="primary"
          disabled={!activeReplaySimulationId || !activeReplayBacktestId || !replayTradeDate}
        >
          排队受控原子双腿回放
        </button>
      </form>
    </section>
  </div>;
}
