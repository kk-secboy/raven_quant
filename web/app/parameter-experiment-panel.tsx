"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";

type Dataset = { name: string; ready: boolean; reproducible: boolean; start_date: string; end_date: string };
type StrategyVersion = { id: string; version: number; status: string };
type Strategy = { id: string; name: string; strategy_type?: string; versions: StrategyVersion[] };
type TrialMetrics = { information_ratio?: number; annualized_excess_return?: number; max_drawdown?: number; average_turnover?: number; robustness_pass_rate?: number };
type LeaderboardRow = { trial_index: number; parameters: Record<string, number>; score: number; warnings: string[]; in_sample: TrialMetrics; out_of_sample: TrialMetrics };
type ExperimentSummary = { trial_count: number; succeeded_count: number; failed_count: number; best_trial_index?: number | null; best_parameters?: Record<string, number> | null; warnings: string[]; leaderboard: LeaderboardRow[] };
type Experiment = {
  id: string; strategy_version_id: string; dataset: string; status: string;
  periods: { in_sample: { start: string; end: string }; out_of_sample: { start: string; end: string } };
  parameter_grid: Record<string, number[]>; summary?: ExperimentSummary | null;
  trial_count?: number; progress?: { completed_count: number; trial_count: number; succeeded_count: number; failed_count: number } | null;
  error?: string | null; created_at: string;
};
type GridRow = { key: string; values: string };

const PARAMETER_LABELS: Record<string, string> = {
  topk: "持仓数量 Top-K", n_drop: "每日换出数量", max_position_weight: "单票权重上限",
  max_daily_turnover: "单日换手上限", max_industry_deviation: "行业偏离上限",
  max_size_deviation: "规模暴露偏离", optimizer_alpha_weight: "优化器 Alpha 权重",
  optimizer_tracking_penalty: "跟踪误差惩罚", optimizer_turnover_penalty: "换手惩罚",
  stop_loss: "单票止损", take_profit_partial: "减半止盈", take_profit: "清仓止盈",
  max_drawdown_reduce: "组合降仓回撤", max_drawdown_liquidate: "组合清仓回撤",
  max_volume_participation: "成交量参与率",
};
const WARNING_LABELS: Record<string, string> = {
  oos_sign_reversal: "样本外超额收益转负", performance_decay: "样本外信息比率衰减超过 50%",
  oos_drawdown_high: "样本外回撤高于 25%", oos_robustness_low: "样本外稳健性通过率偏低",
  multiple_testing_risk: "组合较多，存在多重检验风险", fragile_ranking: "前两名得分接近，排名不稳定",
  boundary_optimum: "最优值落在搜索边界，应扩大范围复核",
};
const pct = (value?: number) => typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
const decimal = (value?: number) => typeof value === "number" ? value.toFixed(3) : "—";

export function ParameterExperimentPanel({ api }: { api: string }) {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedVersion, setSelectedVersion] = useState("");
  const [dataset, setDataset] = useState("");
  const [start, setStart] = useState("2024-01-01");
  const [end, setEnd] = useState(new Date().toISOString().slice(0, 10));
  const [gridRows, setGridRows] = useState<GridRow[]>([
    { key: "max_daily_turnover", values: "0.15, 0.20, 0.25" },
    { key: "optimizer_tracking_penalty", values: "0.50, 1.00, 2.00" },
    { key: "optimizer_turnover_penalty", values: "0.05, 0.10, 0.20" },
  ]);
  const [selectedExperiment, setSelectedExperiment] = useState("");
  const [detail, setDetail] = useState<Experiment | null>(null);
  const [message, setMessage] = useState("");

  async function load() {
    try {
      const responses = await Promise.all([
        apiFetch(`${api}/api/strategies`, { cache: "no-store" }),
        apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" }),
        apiFetch(`${api}/api/parameter-experiments`, { cache: "no-store" }),
      ]);
      if (responses.some((response) => !response.ok)) throw new Error("experiment API unavailable");
      const nextStrategies: Strategy[] = await responses[0].json();
      const nextDatasets: Dataset[] = await responses[1].json();
      const nextExperiments: Experiment[] = await responses[2].json();
      setStrategies(nextStrategies); setDatasets(nextDatasets); setExperiments(nextExperiments);
      const versions = nextStrategies.filter((item) => item.strategy_type !== "pair").flatMap((item) => item.versions);
      if (!selectedVersion && versions.length) setSelectedVersion(versions[0].id);
      if (!dataset) {
        const eligible = nextDatasets.find((item) => item.ready && item.reproducible);
        if (eligible) { setDataset(eligible.name); setStart(eligible.start_date); setEnd(eligible.end_date); }
      }
      const target = selectedExperiment || nextExperiments[0]?.id;
      if (target) {
        setSelectedExperiment(target);
        const response = await apiFetch(`${api}/api/parameter-experiments/${target}`, { cache: "no-store" });
        if (response.ok) setDetail(await response.json());
      }
    } catch { setMessage("无法读取参数实验中心，请确认数据库已升级且 Python 后端正在运行。"); }
  }

  useEffect(() => {
    const initial = window.setTimeout(load, 0); const timer = window.setInterval(load, 8000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedExperiment]);

  const versionOptions = strategies.filter((item) => item.strategy_type !== "pair").flatMap((strategy) => strategy.versions.map((version) => ({ strategy, version })));
  const parsedGrid = useMemo(() => {
    const grid: Record<string, number[]> = {}; let valid = gridRows.length > 0;
    for (const row of gridRows) {
      const values = row.values.split(",").map((item) => Number(item.trim())).filter(Number.isFinite);
      if (!row.key || !values.length || row.key in grid) valid = false;
      grid[row.key] = Array.from(new Set(values));
    }
    const trialCount = Object.values(grid).reduce((total, values) => total * values.length, 1);
    return { grid, trialCount, valid: valid && trialCount <= 27 };
  }, [gridRows]);
  const active = experiments.some((item) => ["queued", "running"].includes(item.status));

  function updateGrid(index: number, patch: Partial<GridRow>) {
    setGridRows(gridRows.map((row, rowIndex) => rowIndex === index ? { ...row, ...patch } : row));
  }

  async function createExperiment(event: FormEvent) {
    event.preventDefault(); if (!selectedVersion || !parsedGrid.valid) return;
    const response = await apiFetch(`${api}/api/strategy-versions/${selectedVersion}/parameter-experiments`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, start, end, parameter_grid: parsedGrid.grid, max_trials: 27 }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "参数实验创建失败"); return; }
    setSelectedExperiment(body.id); setDetail(body);
    setMessage(`实验 ${body.id.slice(0, 8)} 已进入 Qlib Worker；失败后重试会复用已完成试验。`);
    await load();
  }

  const summary = detail?.summary;
  return <div className="experiment-workspace">
    {message && <div className="notice">{message}</div>}
    <section className="data-panel">
      <div className="panel-heading"><div><p className="eyebrow">OUT-OF-SAMPLE PARAMETER GOVERNANCE</p><h2>参数实验中心</h2></div><span className={`state ${active ? "running" : "ready"}`}>{active ? "实验运行中" : "可创建实验"}</span></div>
      <div className="execution-note"><b>实验不会修改策略版本</b><span>每组参数都复用正式 Qlib 回测，前 60% 为样本内、后 40% 为样本外；排名只作为创建新策略版本的研究证据。</span></div>
      <form className="experiment-form" onSubmit={createExperiment}>
        <div className="form-row">
          <label>不可变策略版本<select value={selectedVersion} onChange={(event) => setSelectedVersion(event.target.value)}>{versionOptions.map(({ strategy, version }) => <option key={version.id} value={version.id}>{strategy.name} · v{version.version} · {version.status}</option>)}</select></label>
          <label>Qlib 数据集<select value={dataset} onChange={(event) => setDataset(event.target.value)}>{datasets.filter((item) => item.ready && item.reproducible).map((item) => <option key={item.name}>{item.name}</option>)}</select></label>
          <label>开始日期<input type="date" value={start} onChange={(event) => setStart(event.target.value)} /></label>
          <label>结束日期<input type="date" value={end} onChange={(event) => setEnd(event.target.value)} /></label>
        </div>
        <div className="experiment-grid-editor">{gridRows.map((row, index) => <div className="experiment-grid-row" key={`${index}-${row.key}`}>
          <select value={row.key} onChange={(event) => updateGrid(index, { key: event.target.value })}>{Object.entries(PARAMETER_LABELS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select>
          <input value={row.values} onChange={(event) => updateGrid(index, { values: event.target.value })} placeholder="逗号分隔，例如 0.15, 0.20, 0.25" />
          <button type="button" className="inline-action" disabled={gridRows.length === 1} onClick={() => setGridRows(gridRows.filter((_, rowIndex) => rowIndex !== index))}>移除</button>
        </div>)}</div>
        <div className="form-actions"><button type="button" className="secondary-action" disabled={gridRows.length >= 5} onClick={() => setGridRows([...gridRows, { key: "max_industry_deviation", values: "0.03, 0.05" }])}>添加参数</button><span>将运行 {parsedGrid.trialCount} 组参数 × 2 个时间窗（上限 27 组）</span><button className="primary" disabled={active || !selectedVersion || !dataset || !parsedGrid.valid}>创建参数实验</button></div>
      </form>
    </section>
    <section className="data-panel">
      <div className="panel-heading"><div><p className="eyebrow">EXPERIMENT EVIDENCE</p><h2>结果与过拟合警告</h2></div><select value={selectedExperiment} onChange={(event) => setSelectedExperiment(event.target.value)}><option value="">尚无实验</option>{experiments.map((item) => <option key={item.id} value={item.id}>{item.id.slice(0, 8)} · {item.status} · {item.dataset}</option>)}</select></div>
      {detail ? <>
        <div className="metric-strip"><div><span>状态</span><strong>{detail.status}</strong></div><div><span>完成进度</span><strong>{summary ? `${summary.succeeded_count} 成功 / ${summary.trial_count}` : detail.progress ? `${detail.progress.completed_count} / ${detail.progress.trial_count}` : `0 / ${detail.trial_count ?? 0}`}</strong></div><div><span>样本内</span><strong>{detail.periods.in_sample.start} → {detail.periods.in_sample.end}</strong></div><div><span>样本外</span><strong>{detail.periods.out_of_sample.start} → {detail.periods.out_of_sample.end}</strong></div></div>
        {detail.error && <div className="notice danger-notice">{detail.error}</div>}
        {summary?.warnings.length ? <div className="experiment-warnings">{summary.warnings.map((warning) => <span key={warning}>{WARNING_LABELS[warning] ?? warning}</span>)}</div> : null}
        {summary?.leaderboard.length ? <div className="table-wrap"><table><thead><tr><th>排名</th><th>参数</th><th>综合分</th><th>样本内 IR</th><th>样本外 IR</th><th>样本外超额</th><th>样本外回撤</th><th>警告</th></tr></thead><tbody>{summary.leaderboard.map((row, index) => <tr key={row.trial_index}><td>#{index + 1}</td><td>{Object.entries(row.parameters).map(([key, value]) => `${PARAMETER_LABELS[key] ?? key}=${value}`).join(" · ")}</td><td>{decimal(row.score)}</td><td>{decimal(row.in_sample.information_ratio)}</td><td>{decimal(row.out_of_sample.information_ratio)}</td><td>{pct(row.out_of_sample.annualized_excess_return)}</td><td>{pct(row.out_of_sample.max_drawdown)}</td><td>{row.warnings.map((warning) => WARNING_LABELS[warning] ?? warning).join("；") || "—"}</td></tr>)}</tbody></table></div> : <div className="empty compact">实验完成后，这里会显示样本外优先的排名、参数边界和性能衰减警告。</div>}
      </> : <div className="empty">尚无参数实验。先选择策略版本、数据集和参数范围。</div>}
    </section>
  </div>;
}
