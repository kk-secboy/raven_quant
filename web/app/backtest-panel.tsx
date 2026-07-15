"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch } from "./api-client";
import { ParameterExperimentPanel } from "./parameter-experiment-panel";

type Factor = { id: string; name: string; description: string; status: string };
type Dataset = { name: string; ready: boolean; reproducible: boolean; start_date: string; end_date: string; trading_days: number };
type StrategyFactor = { factor_candidate_id: string; name: string; weight: number; direction: number };
type StrategyVersion = {
  id: string; version: number; status: string; benchmark: string; universe: string;
  config: Record<string, number | string>; factors: StrategyFactor[];
};
type Strategy = { id: string; name: string; description: string; status: string; versions: StrategyVersion[] };
type StrategyRecipe = {
  id: string; version: string; name: string; category: string; description: string;
  benchmark: string; universe: string; rdagent_objective: string;
  factor_guidance: string[]; config_overrides: Record<string, number | string>;
};
type Backtest = {
  id: string; strategy_version_id: string; dataset: string; status: string;
  periods: { start: string; end: string }; metrics?: Record<string, unknown> | null;
  error?: string | null;
};
type RobustnessScenario = {
  name: string; status: string; overrides: Record<string, number>;
  metrics?: Record<string, unknown>; error?: string;
};
type RobustnessReport = {
  scenario_count: number; passed_count: number; pass_rate: number; passed: boolean;
  scenarios: RobustnessScenario[];
};
type RollingReport = {
  window_count: number; passed_count: number; pass_rate: number; passed: boolean;
  windows: Array<{ start: string; end: string; status: string; metrics?: Record<string, unknown>; error?: string }>;
};
type EventStressReport = {
  event_count: number; passed_count: number; pass_rate: number; passed: boolean;
  events: Array<{ start: string; end: string; status: string; strategy_return?: number; benchmark_return?: number; underperformance?: number; error?: string }>;
};

const pct = (value: unknown) => typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
const decimal = (value: unknown) => typeof value === "number" ? value.toFixed(3) : "—";

export function BacktestPanel({ api }: { api: string }) {
  const [view, setView] = useState("create");
  const [factors, setFactors] = useState<Factor[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [backtests, setBacktests] = useState<Backtest[]>([]);
  const [recipes, setRecipes] = useState<StrategyRecipe[]>([]);
  const [recipeId, setRecipeId] = useState("custom");
  const [selectedFactors, setSelectedFactors] = useState<Record<string, number>>({});
  const [name, setName] = useState("沪深300 多因子增强");
  const [description, setDescription] = useState("使用已晋级因子构建下一交易日开盘执行的受约束 Top-K 策略。");
  const [topk, setTopk] = useState(50);
  const [nDrop, setNDrop] = useState(5);
  const [maxPositionWeight, setMaxPositionWeight] = useState(0.02);
  const [maxDailyTurnover, setMaxDailyTurnover] = useState(0.20);
  const [maxDailyLoss, setMaxDailyLoss] = useState(0.03);
  const [maxTrackingError, setMaxTrackingError] = useState(0.12);
  const [maxDrawdown, setMaxDrawdown] = useState(0.25);
  const [maxTurnover, setMaxTurnover] = useState(0.60);
  const [minSharpe, setMinSharpe] = useState(0);
  const [minSortino, setMinSortino] = useState(0);
  const [minRobustness, setMinRobustness] = useState(0.75);
  const [capacityNotional, setCapacityNotional] = useState(5_000_000);
  const [stopLoss, setStopLoss] = useState(0.07);
  const [takeProfitPartial, setTakeProfitPartial] = useState(0.12);
  const [takeProfitPartialFraction, setTakeProfitPartialFraction] = useState(0.50);
  const [takeProfit, setTakeProfit] = useState(0.20);
  const [maxDrawdownReduce, setMaxDrawdownReduce] = useState(0.10);
  const [maxDrawdownLiquidate, setMaxDrawdownLiquidate] = useState(0.15);
  const [drawdownReductionExposure, setDrawdownReductionExposure] = useState(0.50);
  const [maxIndustryWeight, setMaxIndustryWeight] = useState(0.30);
  const [maxIndustryDeviation, setMaxIndustryDeviation] = useState(0.05);
  const [maxSizeDeviation, setMaxSizeDeviation] = useState(0.30);
  const [portfolioConstruction, setPortfolioConstruction] = useState("topk_equal_weight");
  const [optimizerAlphaWeight, setOptimizerAlphaWeight] = useState(0.05);
  const [optimizerTrackingPenalty, setOptimizerTrackingPenalty] = useState(1.0);
  const [optimizerTurnoverPenalty, setOptimizerTurnoverPenalty] = useState(0.10);
  const [maxParticipation, setMaxParticipation] = useState(0.01);
  const [minCapacityFill, setMinCapacityFill] = useState(0.95);
  const [selectedVersion, setSelectedVersion] = useState("");
  const [dataset, setDataset] = useState("");
  const [start, setStart] = useState("2024-01-01");
  const [end, setEnd] = useState(new Date().toISOString().slice(0, 10));
  const [approvalReason, setApprovalReason] = useState("");
  const [message, setMessage] = useState("");
  const [serverDefaults, setServerDefaults] = useState<Record<string, number | string>>({});
  const defaultsApplied = useRef(false);

  function applyVisibleConfig(config: Record<string, number | string>) {
    setTopk(Number(config.topk));
    setNDrop(Number(config.n_drop));
    setMaxPositionWeight(Number(config.max_position_weight));
    setMaxDailyTurnover(Number(config.max_daily_turnover));
    setMaxDailyLoss(Number(config.max_daily_loss));
    setMaxTrackingError(Number(config.max_tracking_error));
    setMaxDrawdown(Number(config.max_drawdown));
    setMaxTurnover(Number(config.max_turnover));
    setMinSharpe(Number(config.min_sharpe_ratio));
    setMinSortino(Number(config.min_sortino_ratio));
    setMinRobustness(Number(config.min_robustness_pass_rate));
    setCapacityNotional(Number(config.capacity_notional));
    setStopLoss(Number(config.stop_loss));
    setTakeProfitPartial(Number(config.take_profit_partial));
    setTakeProfitPartialFraction(Number(config.take_profit_partial_fraction));
    setTakeProfit(Number(config.take_profit));
    setMaxDrawdownReduce(Number(config.max_drawdown_reduce));
    setMaxDrawdownLiquidate(Number(config.max_drawdown_liquidate));
    setDrawdownReductionExposure(Number(config.drawdown_reduction_exposure));
    setMaxIndustryWeight(Number(config.max_industry_weight));
    setMaxIndustryDeviation(Number(config.max_industry_deviation ?? 0.05));
    setMaxSizeDeviation(Number(config.max_size_deviation ?? 0.30));
    setPortfolioConstruction(String(config.portfolio_construction ?? "topk_equal_weight"));
    setOptimizerAlphaWeight(Number(config.optimizer_alpha_weight ?? 0.05));
    setOptimizerTrackingPenalty(Number(config.optimizer_tracking_penalty ?? 1.0));
    setOptimizerTurnoverPenalty(Number(config.optimizer_turnover_penalty ?? 0.10));
    setMaxParticipation(Number(config.max_volume_participation));
    setMinCapacityFill(Number(config.min_capacity_fill_ratio));
  }

  async function load() {
    try {
      const responses = await Promise.all([
        apiFetch(`${api}/api/factors?status=promoted`, { cache: "no-store" }),
        apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" }),
        apiFetch(`${api}/api/strategies`, { cache: "no-store" }),
        apiFetch(`${api}/api/backtests`, { cache: "no-store" }),
        apiFetch(`${api}/api/settings/strategy-defaults`, { cache: "no-store" }),
        apiFetch(`${api}/api/strategy-recipes`, { cache: "no-store" }),
      ]);
      const nextFactors = await responses[0].json();
      const nextDatasets = await responses[1].json();
      const nextStrategies = await responses[2].json();
      setFactors(nextFactors);
      setDatasets(nextDatasets);
      setStrategies(nextStrategies);
      setBacktests(await responses[3].json());
      const defaults = await responses[4].json();
      if (!responses[4].ok) throw new Error(defaults.detail ?? "strategy defaults unavailable");
      setServerDefaults(defaults.config);
      const recipeBody: { recipes: StrategyRecipe[] } = await responses[5].json();
      setRecipes(recipeBody.recipes.filter((item) => item.category === "multifactor"));
      if (!defaultsApplied.current) {
        applyVisibleConfig(defaults.config as Record<string, number | string>);
        defaultsApplied.current = true;
      }
      if (!Object.keys(selectedFactors).length && nextFactors.length) {
        setSelectedFactors({ [nextFactors[0].id]: 1 });
      }
      if (!dataset && nextDatasets.length) {
        const eligible = nextDatasets.find((item: Dataset) => item.ready && item.reproducible);
        if (eligible) {
          setDataset(eligible.name);
          setStart(eligible.start_date);
          setEnd(eligible.end_date);
        }
      }
      if (!selectedVersion && nextStrategies.length) {
        setSelectedVersion(nextStrategies[0].versions[0]?.id ?? "");
      }
    } catch {
      setMessage("无法读取策略控制面，请确认 Python 后端正在运行。");
    }
  }

  useEffect(() => {
    const initial = window.setTimeout(load, 0);
    const timer = window.setInterval(load, 8000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const versions = strategies.flatMap((strategy) => strategy.versions.map((version) => ({ strategy, version })));
  const current = versions.find((item) => item.version.id === selectedVersion);
  const currentBacktest = backtests.find((item) => item.strategy_version_id === selectedVersion);
  const active = useMemo(() => backtests.some((item) => ["queued", "running"].includes(item.status)), [backtests]);
  const riskConfigurationValid = takeProfitPartial < takeProfit
    && maxDrawdownReduce < maxDrawdownLiquidate
    && maxIndustryWeight >= maxPositionWeight
    && maxIndustryDeviation >= 0
    && maxSizeDeviation >= 0
    && (portfolioConstruction !== "benchmark_relative_qp" || topk * maxPositionWeight >= 1)
    && optimizerAlphaWeight + optimizerTrackingPenalty + optimizerTurnoverPenalty > 0;

  function toggleFactor(factorId: string) {
    const next = { ...selectedFactors };
    if (factorId in next) delete next[factorId]; else next[factorId] = 1;
    setSelectedFactors(next);
  }

  function selectRecipe(value: string) {
    setRecipeId(value);
    if (value === "custom") {
      applyVisibleConfig(serverDefaults as Record<string, number | string>);
      return;
    }
    const recipe = recipes.find((item) => item.id === value);
    if (!recipe) return;
    const merged = { ...serverDefaults, ...recipe.config_overrides } as Record<string, number | string>;
    applyVisibleConfig(merged);
    setName(recipe.name);
    setDescription(recipe.description);
  }

  function strategyDefinition() {
    const recipe = recipes.find((item) => item.id === recipeId);
    return {
      benchmark: recipe?.benchmark ?? "SH000300", universe: recipe?.universe ?? "cn_all",
      factors: Object.entries(selectedFactors).map(([candidate_id, weight]) => ({ candidate_id, weight })),
      config: {
        ...serverDefaults,
        ...(recipe?.config_overrides ?? {}),
        recipe_id: recipe?.id ?? "custom",
        recipe_version: recipe?.version ?? "custom",
        topk, n_drop: nDrop, max_position_weight: maxPositionWeight,
        max_daily_turnover: maxDailyTurnover,
        max_daily_loss: maxDailyLoss, stop_loss: stopLoss, take_profit: takeProfit,
        take_profit_partial: takeProfitPartial, take_profit_partial_fraction: takeProfitPartialFraction,
        max_drawdown_reduce: maxDrawdownReduce, max_drawdown_liquidate: maxDrawdownLiquidate,
        drawdown_reduction_exposure: drawdownReductionExposure,
        max_industry_weight: maxIndustryWeight,
        max_industry_deviation: maxIndustryDeviation,
        max_size_deviation: maxSizeDeviation,
        portfolio_construction: portfolioConstruction,
        optimizer_alpha_weight: optimizerAlphaWeight,
        optimizer_tracking_penalty: optimizerTrackingPenalty,
        optimizer_turnover_penalty: optimizerTurnoverPenalty,
        max_tracking_error: maxTrackingError, max_drawdown: maxDrawdown,
        max_turnover: maxTurnover,
        min_sharpe_ratio: minSharpe, min_sortino_ratio: minSortino,
        min_robustness_pass_rate: minRobustness,
        capacity_notional: capacityNotional,
        max_volume_participation: maxParticipation, min_capacity_fill_ratio: minCapacityFill,
      },
    };
  }

  async function createStrategy(event: FormEvent) {
    event.preventDefault();
    const response = await apiFetch(`${api}/api/strategies`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name, description, ...strategyDefinition(),
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "策略创建失败"); return; }
    setMessage(`策略 ${body.name} v1 已创建，必须先完成回测才能审批。`);
    setSelectedVersion(body.versions[0].id);
    await load();
  }

  async function createNextVersion() {
    if (!current) return;
    const response = await apiFetch(`${api}/api/strategies/${current.strategy.id}/versions`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(strategyDefinition()),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "策略新版本创建失败"); return; }
    setSelectedVersion(body.id);
    setMessage(`策略 v${body.version} 已创建，必须重新完成 Qlib 回测和审批。`);
    await load();
  }

  async function runBacktest(event: FormEvent) {
    event.preventDefault();
    if (!selectedVersion) return;
    const response = await apiFetch(`${api}/api/strategy-versions/${selectedVersion}/backtests`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, start, end }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "回测创建失败"); return; }
    setMessage(`回测 ${body.id.slice(0, 8)} 已进入 Qlib Worker。`);
    await load();
  }

  async function approve(event: FormEvent) {
    event.preventDefault();
    if (!selectedVersion) return;
    const response = await apiFetch(`${api}/api/strategy-versions/${selectedVersion}/approve`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actor: "local-operator", reason: approvalReason }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "审批失败"); return; }
    setMessage(`策略版本 v${body.version} 已通过风险审批。`);
    setApprovalReason("");
    await load();
  }

  const metrics = currentBacktest?.metrics ?? {};
  const isQlibNative = metrics.backtest_engine === "qlib" && metrics.qlib_native_backtest === true;
  const robustness = typeof metrics.robustness === "object" && metrics.robustness !== null
    ? metrics.robustness as RobustnessReport : null;
  const rolling = typeof metrics.rolling === "object" && metrics.rolling !== null
    ? metrics.rolling as RollingReport : null;
  const eventStress = typeof metrics.event_stress === "object" && metrics.event_stress !== null
    ? metrics.event_stress as EventStressReport : null;
  const selectedRecipe = recipes.find((item) => item.id === recipeId);
  return <>
    {message && <div className="notice">{message}</div>}
    <div className="page-tabs" role="tablist" aria-label="回测工作区">
      {[["create", "新建回测"], ["results", "结果与压力测试"], ["experiments", "参数实验"], ["history", "运行记录"]].map(([value, label]) => <button type="button" role="tab" aria-selected={view === value} className={view === value ? "active" : ""} onClick={() => setView(value)} key={value}>{label}</button>)}
    </div>
    {view === "experiments" && <ParameterExperimentPanel api={api} />}
    {view === "create" && <section className="backtest-hero">
      <form className="strategy-builder" onSubmit={createStrategy}>
        <div className="card-heading"><div><span>不可变策略版本</span><strong>用已晋级因子建模</strong></div><span className="status-chip">研究草稿</span></div>
        <label>文档策略配方<select value={recipeId} onChange={(event) => selectRecipe(event.target.value)}><option value="custom">自定义（使用系统默认模板）</option>{recipes.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.version}</option>)}</select></label>
        {selectedRecipe && <div className="execution-note"><b>不可变配方基线 · {selectedRecipe.name}</b><span>{selectedRecipe.description}</span>{selectedRecipe.factor_guidance.map((item) => <span key={item}>{item}</span>)}</div>}
        <div className="form-row"><label>策略名称<input value={name} onChange={(event) => setName(event.target.value)} /></label><label>基准<input value="沪深300 · SH000300" disabled /></label></div>
        <label>策略说明<textarea value={description} onChange={(event) => setDescription(event.target.value)} /></label>
        <div className="factor-picker"><span>选择已晋级因子</span>{factors.map((item) => <label key={item.id}><input type="checkbox" checked={item.id in selectedFactors} onChange={() => toggleFactor(item.id)} /><b>{item.name}</b><small>{item.description}</small>{item.id in selectedFactors ? <input type="number" step="0.1" value={selectedFactors[item.id]} onChange={(event) => setSelectedFactors({ ...selectedFactors, [item.id]: Number(event.target.value) })} /> : null}</label>)}{!factors.length && <div className="empty compact">因子库中还没有已晋级因子。先完成 RD-Agent → Qlib → 人工晋级。</div>}</div>
        <details className="advanced-options strategy-risk-options">
          <summary>风控与容量参数（使用系统默认值，可展开修改）</summary>
          <div className="execution-note"><b>默认风控模板</b><span>比例直接按百分数填写，例如 7 = 7%</span><span>已审批版本不会被修改；调整参数会创建新的不可变策略版本</span></div>
          <div className="risk-grid">
          <label>组合构建方式<select value={portfolioConstruction} onChange={(event) => setPortfolioConstruction(event.target.value)}><option value="topk_equal_weight">Top-K 等权</option><option value="benchmark_relative_qp">基准相对优化</option></select></label>
          <label>持仓数量<input type="number" min="5" max="500" value={topk} onChange={(event) => setTopk(Number(event.target.value))} /></label>
          <label>缓冲替换<input type="number" min="0" max={topk} value={nDrop} onChange={(event) => setNDrop(Number(event.target.value))} /></label>
          <label>单票权重上限（%）<input type="number" step="0.1" min="0.1" max="20" value={maxPositionWeight * 100} onChange={(event) => setMaxPositionWeight(Number(event.target.value) / 100)} /></label>
          <label>单日换手上限（%）<input type="number" step="1" min="0.1" max="100" value={maxDailyTurnover * 100} onChange={(event) => setMaxDailyTurnover(Number(event.target.value) / 100)} /></label>
          <label>单日亏损熔断（%）<input type="number" step="0.5" min="0.1" max="20" value={maxDailyLoss * 100} onChange={(event) => setMaxDailyLoss(Number(event.target.value) / 100)} /></label>
          <label>单票止损（%）<input type="number" step="1" min="1" max="50" value={stopLoss * 100} onChange={(event) => setStopLoss(Number(event.target.value) / 100)} /></label>
          <label>首次止盈阈值（%）<input type="number" step="1" min="1" max="200" value={takeProfitPartial * 100} onChange={(event) => setTakeProfitPartial(Number(event.target.value) / 100)} /></label>
          <label>首次止盈减仓（%）<input type="number" step="5" min="5" max="95" value={takeProfitPartialFraction * 100} onChange={(event) => setTakeProfitPartialFraction(Number(event.target.value) / 100)} /></label>
          <label>最终止盈清仓（%）<input type="number" step="1" min="1" max="500" value={takeProfit * 100} onChange={(event) => setTakeProfit(Number(event.target.value) / 100)} /></label>
          <label>组合回撤降仓（%）<input type="number" step="1" min="1" max="50" value={maxDrawdownReduce * 100} onChange={(event) => setMaxDrawdownReduce(Number(event.target.value) / 100)} /></label>
          <label>降仓后目标仓位（%）<input type="number" step="5" min="5" max="95" value={drawdownReductionExposure * 100} onChange={(event) => setDrawdownReductionExposure(Number(event.target.value) / 100)} /></label>
          <label>组合回撤清仓（%）<input type="number" step="1" min="1" max="80" value={maxDrawdownLiquidate * 100} onChange={(event) => setMaxDrawdownLiquidate(Number(event.target.value) / 100)} /></label>
          <label>行业权重上限（%）<input type="number" step="5" min="5" max="100" value={maxIndustryWeight * 100} onChange={(event) => setMaxIndustryWeight(Number(event.target.value) / 100)} /></label>
          <label>相对基准行业偏离（%）<input type="number" step="0.5" min="0" max="30" value={maxIndustryDeviation * 100} onChange={(event) => setMaxIndustryDeviation(Number(event.target.value) / 100)} /></label>
          <label>市值风格偏离（标准差）<input type="number" step="0.05" min="0" max="2" value={maxSizeDeviation} onChange={(event) => setMaxSizeDeviation(Number(event.target.value))} /></label>
          {portfolioConstruction === "benchmark_relative_qp" && <>
          <label>因子收益权重<input type="number" step="0.01" min="0" max="10" value={optimizerAlphaWeight} onChange={(event) => setOptimizerAlphaWeight(Number(event.target.value))} /></label>
          <label>基准跟踪惩罚<input type="number" step="0.1" min="0" max="100" value={optimizerTrackingPenalty} onChange={(event) => setOptimizerTrackingPenalty(Number(event.target.value))} /></label>
          <label>换手惩罚<input type="number" step="0.01" min="0" max="100" value={optimizerTurnoverPenalty} onChange={(event) => setOptimizerTurnoverPenalty(Number(event.target.value))} /></label>
          </>}
          <label>最大跟踪误差<input type="number" step="0.01" value={maxTrackingError} onChange={(event) => setMaxTrackingError(Number(event.target.value))} /></label>
          <label>回测最大回撤<input type="number" step="0.01" value={maxDrawdown} onChange={(event) => setMaxDrawdown(Number(event.target.value))} /></label>
          <label>最大平均换手<input type="number" step="0.05" value={maxTurnover} onChange={(event) => setMaxTurnover(Number(event.target.value))} /></label>
          <label>最小 Sharpe<input type="number" step="0.1" value={minSharpe} onChange={(event) => setMinSharpe(Number(event.target.value))} /></label>
          <label>最小 Sortino<input type="number" step="0.1" value={minSortino} onChange={(event) => setMinSortino(Number(event.target.value))} /></label>
          <label>稳健场景通过率<input type="number" step="0.05" min="0" max="1" value={minRobustness} onChange={(event) => setMinRobustness(Number(event.target.value))} /></label>
          <label>容量测试资金<input type="number" step="100000" min="100000" value={capacityNotional} onChange={(event) => setCapacityNotional(Number(event.target.value))} /></label>
          <label>日成交量参与上限（%）<input type="number" step="0.1" min="0.1" max="20" value={maxParticipation * 100} onChange={(event) => setMaxParticipation(Number(event.target.value) / 100)} /></label>
          <label>容量最小成交率<input type="number" step="0.01" min="0" max="1" value={minCapacityFill} onChange={(event) => setMinCapacityFill(Number(event.target.value))} /></label>
          </div>
        </details>
        {!riskConfigurationValid && <div className="notice">参数无效：请检查止盈/回撤阈值、行业与单票上限；基准相对优化还要求持仓数 × 单票上限不低于 100%，且目标函数至少有一个正权重。</div>}
        <button className="primary" disabled={!riskConfigurationValid || !Object.keys(selectedFactors).length || name.length < 3 || description.length < 10}>创建策略 v1</button>
        <button type="button" onClick={createNextVersion} disabled={!riskConfigurationValid || !current || !Object.keys(selectedFactors).length}>基于当前参数创建 vNext</button>
      </form>

      <form className="backtest-launcher" onSubmit={runBacktest}>
        <div className="card-heading"><div><span>策略回测</span><strong>下一交易日开盘执行</strong></div></div>
        <label>策略版本<select value={selectedVersion} onChange={(event) => setSelectedVersion(event.target.value)}>{versions.length ? versions.map(({ strategy, version }) => <option key={version.id} value={version.id}>{strategy.name} · v{version.version} · {version.status}</option>) : <option value="">无策略版本</option>}</select></label>
        <label>Qlib 快照<select value={dataset} onChange={(event) => { const value = event.target.value; setDataset(value); const item = datasets.find((candidate) => candidate.name === value); if (item) { setStart(item.start_date); setEnd(item.end_date); } }}>{datasets.map((item) => <option key={item.name} value={item.name} disabled={!item.ready || !item.reproducible}>{item.name} · {item.trading_days} 日 · {item.reproducible ? "可复现" : "缺少溯源"}</option>)}</select></label>
        <div className="form-row"><label>开始<input type="date" value={start} onChange={(event) => setStart(event.target.value)} /></label><label>结束<input type="date" value={end} onChange={(event) => setEnd(event.target.value)} /></label></div>
        <div className="execution-note"><b>执行与压力测试</b><span>信号日 t 收盘后生成，t+1 开盘成交</span><span>审批至少需要 504 个交易日，且必须位于全部因子的独立测试窗</span><span>买入 5bp / 卖出 15bp，并额外运行双倍成本</span><span>重跑持仓数量、缓冲区和换手参数扰动</span><span>按执行日成交额限制资金容量</span></div>
        <button className="primary" disabled={!selectedVersion || !dataset || active}>运行策略回测</button>
      </form>
    </section>}

    {view === "results" && <>
    {currentBacktest?.status === "succeeded" && <div className="notice">正式回测引擎：{isQlibNative ? "Qlib 原生回测" : "非 Qlib 验证结果（不可审批）"}</div>}
    {currentBacktest?.status === "succeeded" && <div className="notice">正式结果来自统一Qlib回测与PortfolioPolicy；推荐服务使用相同Policy版本。</div>}
    {currentBacktest?.status === "succeeded" && metrics.portfolio_construction === "benchmark_relative_qp" && <div className="notice">组合优化：基准相对权重已执行 · 平均主动权重 {pct(metrics.optimizer_mean_active_share)} · 优化器预计单边换手 {pct(metrics.optimizer_mean_expected_turnover)} · 最大跟踪代理 {decimal(metrics.optimizer_max_tracking_risk_proxy)} · 最大行业偏离 {pct(metrics.max_industry_deviation)} · 最大市值风格偏离 {decimal(metrics.max_size_deviation)} · 最大迭代 {String(metrics.optimizer_max_iterations ?? "—")}</div>}
    <section className="metric-strip backtest-metrics"><div><span>年化收益</span><strong>{pct(metrics.annualized_return)}</strong></div><div><span>年化超额</span><strong>{pct(metrics.annualized_excess_return)}</strong></div><div><span>跟踪误差</span><strong>{pct(metrics.tracking_error)}</strong></div><div><span>信息比率</span><strong>{decimal(metrics.information_ratio)}</strong></div><div><span>最大回撤</span><strong>{pct(metrics.max_drawdown)}</strong></div><div><span>平均换手</span><strong>{pct(metrics.average_turnover)}</strong></div></section>
    <section className="metric-strip backtest-metrics"><div><span>Sharpe</span><strong>{decimal(metrics.sharpe_ratio)}</strong></div><div><span>Sortino</span><strong>{decimal(metrics.sortino_ratio)}</strong></div><div><span>容量成交率</span><strong>{pct(metrics.capacity_fill_ratio)}</strong></div><div><span>稳健通过率</span><strong>{pct(metrics.robustness_pass_rate)}</strong></div><div><span>最差场景超额</span><strong>{pct(metrics.worst_scenario_excess_return)}</strong></div><div><span>单日最大亏损</span><strong>{pct(metrics.max_daily_loss)}</strong></div></section>

    {robustness && <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">ROBUSTNESS / CAPACITY</p><h2>稳健性压力场景</h2></div><span className={`state ${robustness.passed ? "ready" : "failed"}`}>{robustness.passed_count} / {robustness.scenario_count} 通过</span></div><div className="table-wrap"><table className="robustness-table"><thead><tr><th>场景</th><th>状态</th><th>年化超额</th><th>Sharpe</th><th>最大回撤</th><th>容量成交率</th></tr></thead><tbody>{robustness.scenarios.map((scenario) => <tr key={scenario.name}><td><code>{scenario.name}</code></td><td><span className={`state ${scenario.status === "passed" ? "ready" : "failed"}`}>{scenario.status}</span></td><td>{pct(scenario.metrics?.annualized_excess_return)}</td><td>{decimal(scenario.metrics?.sharpe_ratio)}</td><td>{pct(scenario.metrics?.max_drawdown)}</td><td>{pct(scenario.metrics?.capacity_fill_ratio)}</td></tr>)}</tbody></table></div></section>}

    {rolling && <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">ROLLING OUT-OF-SAMPLE</p><h2>滚动窗口稳定性</h2></div><span className={`state ${rolling.passed ? "ready" : "failed"}`}>{rolling.passed_count} / {rolling.window_count} 通过</span></div><div className="table-wrap"><table><thead><tr><th>窗口</th><th>状态</th><th>年化超额</th><th>Sharpe</th><th>最大回撤</th></tr></thead><tbody>{rolling.windows.map((item) => <tr key={`${item.start}-${item.end}`}><td>{item.start} → {item.end}</td><td><span className={`state ${item.status === "passed" ? "ready" : "failed"}`}>{item.status}</span></td><td>{pct(item.metrics?.annualized_excess_return)}</td><td>{decimal(item.metrics?.sharpe_ratio)}</td><td>{pct(item.metrics?.max_drawdown)}</td></tr>)}</tbody></table></div></section>}

    {eventStress && <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">HISTORICAL EVENT STRESS</p><h2>基准最差历史区间</h2></div><span className={`state ${eventStress.passed ? "ready" : "failed"}`}>{eventStress.passed_count} / {eventStress.event_count} 通过</span></div><div className="table-wrap"><table><thead><tr><th>事件窗口</th><th>状态</th><th>策略收益</th><th>基准收益</th><th>相对落后</th></tr></thead><tbody>{eventStress.events.map((item) => <tr key={`${item.start}-${item.end}`}><td>{item.start} → {item.end}</td><td><span className={`state ${item.status === "passed" ? "ready" : "failed"}`}>{item.status}</span></td><td>{pct(item.strategy_return)}</td><td>{pct(item.benchmark_return)}</td><td>{pct(item.underperformance)}</td></tr>)}</tbody></table></div></section>}

    {current && currentBacktest?.status === "succeeded" && isQlibNative && current.version.status !== "approved" ? <form className="approval-panel" onSubmit={approve}><div><span>风险审批</span><strong>{current.strategy.name} · v{current.version.version}</strong><small>Qlib 原生回测及约束、稳健性、容量证据必须全部满足版本阈值。</small></div><textarea value={approvalReason} onChange={(event) => setApprovalReason(event.target.value)} placeholder="填写审批依据（至少 10 个字符）" /><button className="primary" disabled={approvalReason.length < 10}>提交审批</button></form> : null}
    {!currentBacktest || currentBacktest.status !== "succeeded" ? <div className="workspace-card empty-state-card"><h2>还没有可查看的回测结果</h2><p>请先在“新建回测”中选择策略版本和 Qlib 数据集运行回测。</p></div> : null}
    </>}

    {view === "history" && <section className="data-panel panel-without-top-margin"><div className="panel-heading"><div><p className="eyebrow">GOVERNED BACKTESTS</p><h2>策略回测记录</h2></div><span>{backtests.length} 条记录</span></div><div className="table-wrap"><table><thead><tr><th>回测</th><th>策略版本</th><th>区间</th><th>状态</th><th>年化超额</th><th>Sharpe</th><th>稳健通过率</th><th>容量成交率</th></tr></thead><tbody>{backtests.map((item) => <tr key={item.id}><td><code>{item.id.slice(0, 10)}</code></td><td><code>{item.strategy_version_id.slice(0, 10)}</code></td><td>{item.periods.start} → {item.periods.end}</td><td><span className={`state ${item.status === "succeeded" ? "ready" : item.status === "failed" ? "failed" : "partial"}`}>{item.status}</span></td><td>{pct(item.metrics?.annualized_excess_return)}</td><td>{decimal(item.metrics?.sharpe_ratio)}</td><td>{pct(item.metrics?.robustness_pass_rate)}</td><td>{pct(item.metrics?.capacity_fill_ratio)}</td></tr>)}</tbody></table>{!backtests.length && <div className="empty">尚无策略回测。先用已晋级因子创建不可变策略版本。</div>}</div></section>}
  </>;
}
