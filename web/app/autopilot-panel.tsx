"use client";

import { useCallback, useMemo, useState } from "react";
import { apiFetch, type PaperTargetProjection } from "./api-client";
import { usePolling } from "./use-polling";

type Branch = { id: string; scenario: string; status: string; error?: string | null };
type Cycle = {
  id: string; dataset: string; status: string; stage: string; updated_at: string; branches: Branch[];
  state?: { blockers?: string[]; [key: string]: unknown };
};
type Trial = {
  id: string; name: string; trial_kind: string; status: string;
  feature_set_id?: string | null; model_family?: string | null;
};
type Tournament = {
  id: string; status: string; stage: string; max_trials: number;
  selected_trial_ids?: string[]; trials?: Trial[];
};
type AutopilotState = {
  config: { enabled: boolean; paper_min_calendar_days: number; [key: string]: unknown };
  revision: number; state: "running" | "idle" | "paused";
  current_cycle?: Cycle | null; current_stage?: string; next_action?: string;
  tournament?: Tournament | null; cycles: Cycle[];
  report_backfill: { counts: Record<string, number>; selected: number; published: number; bytes_downloaded: number; complete: boolean };
  real_trading: { connected: false; automatic: false; minimum_paper_calendar_days: number; decision: "manual_only" };
};
type DataAutomation = {
  coverage: { covered: number; total: number; ready: boolean };
  schedules: { status: string; next_run_at?: string | null }[];
};
type SimulationPortfolio = {
  id: string; name: string; status: string; nav?: number;
  latest_nav?: { trade_date: string; nav: number; daily_return?: number | null; status: string } | null;
  latest_batch?: { signal_date?: string; trade_date?: string; status?: string } | null;
  position_count?: number;
};

async function jsonResponse<T>(request: Promise<Response>): Promise<T> {
  const response = await request;
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

const SCENARIO_LABELS: Record<string, string> = {
  fin_factor: "因子研究", fin_model: "模型研究", fin_factor_report: "研报研究", fin_quant: "联合优化",
  factor_sota: "因子增量验证",
};
const STAGE_LABELS: Record<string, string> = {
  waiting_for_data: "等待数据", parallel_research: "并行研究", feature_screen: "特征筛选",
  model_full: "模型完整验证", model_tournament: "模型竞赛", ensemble: "模型集成",
  joint_optimization: "联合优化", portfolio_selection: "组合方案选择",
  formal_backtest: "正式 OOS", final_oos: "正式 OOS",
  paper: "模拟运行", complete: "日常运行", research_blocked: "研究阻断",
  joint_optimization_blocked: "联合优化阻断", dataset_unavailable: "数据版本不可用",
  capital_gate_blocked: "组合与模拟门禁阻断",
};
const EMPTY_BRANCHES: Branch[] = [];
const EMPTY_TRIALS: Trial[] = [];

function timeText(value?: string | null) {
  return value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "等待调度";
}
function statusText(status?: string) {
  return ({
    planned: "已预注册", preregistered: "已预注册", queued: "排队", running: "运行中",
    evaluating: "独立验证", passed: "通过", selected: "冠军", succeeded: "完成",
    failed: "失败", rejected: "未晋级", blocked: "阻断", skipped: "跳过",
    active: "运行中", complete: "完成",
  } as Record<string, string>)[status ?? ""] ?? "等待";
}
function branchClass(status?: string) {
  if (["succeeded", "passed", "selected", "complete"].includes(status ?? "")) return "ready";
  if (["failed", "blocked"].includes(status ?? "")) return "blocked";
  if (["queued", "running", "evaluating", "planned", "preregistered"].includes(status ?? "")) return "running";
  return "waiting";
}
function pct(value?: number | null) {
  return value == null ? "—" : `${(value * 100).toFixed(2)}%`;
}

export function AutopilotPanel({
  api, onNavigate, onOpenAdvanced,
}: {
  api: string; onNavigate: (index: number) => void; onOpenAdvanced: (index: number) => void;
}) {
  const [autopilot, setAutopilot] = useState<AutopilotState | null>(null);
  const [automation, setAutomation] = useState<DataAutomation | null>(null);
  const [simulations, setSimulations] = useState<SimulationPortfolio[]>([]);
  const [paperTarget, setPaperTarget] = useState<PaperTargetProjection | null>(null);
  const [message, setMessage] = useState("");
  const [loadWarning, setLoadWarning] = useState("");
  const [autopilotLoadState, setAutopilotLoadState] = useState<"loading" | "ready" | "error">("loading");
  const [automationLoadState, setAutomationLoadState] = useState<"loading" | "ready" | "error">("loading");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    const resources = [
      { label: "自动驾驶", request: jsonResponse<AutopilotState>(apiFetch(`${api}/api/autopilot`, { cache: "no-store" })) },
      { label: "数据调度", request: jsonResponse<DataAutomation>(apiFetch(`${api}/api/data-automation`, { cache: "no-store" })) },
      { label: "模拟盘", request: jsonResponse<SimulationPortfolio[]>(apiFetch(`${api}/api/simulation-portfolios`, { cache: "no-store" })) },
      { label: "模拟候选", request: jsonResponse<PaperTargetProjection>(apiFetch(`${api}/api/autopilot/paper-target`, { cache: "no-store" })) },
    ] as const;
    const results = await Promise.allSettled(resources.map((item) => item.request));
    const [autopilotResult, automationResult, simulationResult, paperTargetResult] = results;
    if (autopilotResult.status === "fulfilled") {
      setAutopilot(autopilotResult.value as AutopilotState);
      setAutopilotLoadState("ready");
    } else setAutopilotLoadState("error");
    if (automationResult.status === "fulfilled") {
      setAutomation(automationResult.value as DataAutomation);
      setAutomationLoadState("ready");
    } else setAutomationLoadState("error");
    if (simulationResult.status === "fulfilled") {
      const portfolios = simulationResult.value as SimulationPortfolio[];
      const selected = portfolios.find((item) => item.status === "active") ?? portfolios[0];
      if (!selected) setSimulations(portfolios);
      else {
        const [detailResult, positionsResult, batchesResult] = await Promise.allSettled([
          jsonResponse<SimulationPortfolio>(apiFetch(`${api}/api/simulation-portfolios/${selected.id}`, { cache: "no-store" })),
          jsonResponse<Record<string, unknown>[]>(apiFetch(`${api}/api/simulation-portfolios/${selected.id}/positions?limit=500`, { cache: "no-store" })),
          jsonResponse<{ signal_date?: string; trade_date?: string; status?: string }[]>(apiFetch(`${api}/api/simulation-portfolios/${selected.id}/batches?limit=1`, { cache: "no-store" })),
        ]);
        const detail = detailResult.status === "fulfilled" ? detailResult.value : selected;
        const positions = positionsResult.status === "fulfilled" ? positionsResult.value : [];
        const batches = batchesResult.status === "fulfilled" ? batchesResult.value : [];
        const enriched = {
          ...selected,
          ...detail,
          position_count: positions.filter((item) => Number(item.quantity ?? item.total_quantity ?? 0) > 0).length,
          latest_batch: batches[0] ?? null,
        };
        setSimulations(portfolios.map((item) => item.id === selected.id ? enriched : item));
      }
    }
    if (paperTargetResult.status === "fulfilled") setPaperTarget(paperTargetResult.value as PaperTargetProjection);
    const failed = results.flatMap((result, index) => result.status === "rejected" ? [resources[index].label] : []);
    setLoadWarning(failed.length ? `部分状态暂未更新，已保留上次成功内容：${failed.join("、")}。` : "");
  }, [api]);
  usePolling(load, 5000);

  const cycle = autopilot?.current_cycle ?? null;
  const branches = cycle?.branches ?? EMPTY_BRANCHES;
  const branchByScenario = useMemo(() => Object.fromEntries(branches.map((item) => [item.scenario, item])), [branches]);
  const failures = branches.filter((item) => ["failed", "blocked"].includes(item.status));
  const enabled = autopilot?.config.enabled === true;
  const autopilotUnavailable = !autopilot && autopilotLoadState === "error";
  const cycleBlocked = cycle?.status === "blocked";
  const stage = autopilot?.current_stage ?? cycle?.stage ?? "waiting_for_data";
  const nextAction = autopilot?.next_action ?? "等待系统确定下一步动作";
  const tournament = autopilot?.tournament ?? null;
  const trials = tournament?.trials ?? EMPTY_TRIALS;
  const trialCounts = useMemo(() => trials.reduce<Record<string, number>>((counts, item) => {
    counts[item.status] = (counts[item.status] ?? 0) + 1;
    return counts;
  }, {}), [trials]);
  const completedTrials = trials.filter((item) => ["passed", "selected", "failed", "rejected"].includes(item.status)).length;
  const selectedTrial = trials.find((item) => item.status === "selected") ?? null;
  const latestSimulation = simulations.find((item) => item.status === "active") ?? simulations[0] ?? null;
  const candidateCount = paperTarget?.status === "ready"
    ? paperTarget.targets.filter((item) => item.target_weight > 0).length
    : latestSimulation?.position_count ?? 0;
  const blockers = useMemo(() => [
    ...(cycle?.state?.blockers ?? []),
    ...failures.map((item) => item.error || `${SCENARIO_LABELS[item.scenario] ?? item.scenario}未完成：${statusText(item.status)}`),
  ].filter((item, index, values) => item && values.indexOf(item) === index), [cycle?.state?.blockers, failures]);
  const nextRun = automation?.schedules.filter((item) => item.status === "active" && item.next_run_at)
    .sort((left, right) => String(left.next_run_at).localeCompare(String(right.next_run_at)))[0]?.next_run_at;

  async function setEnabled(value: boolean) {
    if (!autopilot) return;
    setBusy(true);
    try {
      const response = await apiFetch(`${api}/api/autopilot`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          config: { ...autopilot.config, enabled: value },
          reason: value ? "Operator resumed the governed automatic research workflow" : "Operator paused new automatic research without deleting artifacts",
        }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body?.detail?.message ?? body?.detail ?? "自动驾驶更新失败");
      setAutopilot(body);
      setMessage(value ? "自动驾驶已恢复。" : "已暂停创建新研究；数据、历史和模拟证据不会删除。");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "自动驾驶更新失败");
    } finally { setBusy(false); }
  }

  const researchStatuses = ["fin_factor", "fin_model", "fin_factor_report"].map((id) => branchByScenario[id]?.status).filter(Boolean);
  const researchStatus = researchStatuses.some((item) => ["failed", "blocked"].includes(item))
    ? "blocked"
    : researchStatuses.some((item) => ["queued", "running", "evaluating"].includes(item))
      ? "running"
      : researchStatuses.length && researchStatuses.every((item) => ["succeeded", "skipped"].includes(item)) ? "succeeded" : undefined;

  return <div className="autopilot-page">
    {loadWarning ? <div className="notice">{loadWarning}</div> : null}
    {message ? <div className="notice">{message}</div> : null}
    <section className={`autopilot-hero ${autopilot && enabled && !cycleBlocked ? "is-on" : ""}`}>
      <div>
        <span className="status-chip">SINGLE AUTOPILOT CYCLE</span>
        <h2>{autopilotUnavailable ? "自动驾驶状态暂不可用" : !autopilot ? "正在读取自动驾驶状态" : !enabled ? "自动驾驶已暂停" : cycleBlocked ? "本轮研究已阻断" : "唯一主线正在运行"}</h2>
        <p>{!autopilot ? (autopilotUnavailable ? "无法读取当前周期；页面不会把未知状态误报为已暂停。" : "正在读取当前周期和研究分支。") : cycle ? `当前数据版本 ${cycle.dataset}，当前阶段：${STAGE_LABELS[stage] ?? stage}。` : "等待下一份通过血缘校验的 Qlib 日频数据，之后自动开始研究。"}</p>
        <div className="autopilot-next"><span>系统下一步</span><strong>{nextAction}</strong></div>
        <div className="autopilot-actions">
          <button className="action-button action-secondary" disabled={busy || !autopilot} onClick={() => setEnabled(!enabled)}>{enabled ? "暂停新研究" : "恢复自动驾驶"}</button>
          {blockers.length ? <button className="action-button action-danger" onClick={() => onNavigate(9)}>查看异常</button> : null}
          <button className="action-button action-primary" onClick={() => onNavigate(8)}>查看模拟盘</button>
        </div>
      </div>
      <div className="autopilot-state-panel"><span className={autopilot && enabled && !cycleBlocked ? "live" : ""}><i />{STAGE_LABELS[stage] ?? "自动驾驶"}</span><strong>{!autopilot ? (autopilotUnavailable ? "状态未知" : "检查中") : !enabled ? "已暂停" : cycleBlocked ? "已阻断" : statusText(cycle?.status)}</strong><small>权限边界：仅研究与模拟盘<br />不连接真实券商</small></div>
    </section>

    <section className="autopilot-flow autopilot-flow-six" aria-label="唯一自动驾驶流水线">
      <article className={automation?.coverage.ready ? "ready" : automationLoadState === "error" ? "blocked" : "waiting"}><span>01</span><div><strong>每日数据</strong><small>{automation ? `${automation.coverage.covered}/${automation.coverage.total}项 · ${timeText(nextRun)}` : automationLoadState === "error" ? "状态暂不可用" : "读取中"}</small></div></article>
      <article className={branchClass(researchStatus)}><span>02</span><div><strong>并行研究</strong><small>{["fin_factor", "fin_model", "fin_factor_report"].map((id) => `${SCENARIO_LABELS[id]}：${statusText(branchByScenario[id]?.status)}`).join(" · ")}</small></div></article>
      <article className={branchClass(tournament?.status)}><span>03</span><div><strong>因子 / 模型竞赛</strong><small>{tournament ? `${completedTrials}/${tournament.max_trials}项已有结论` : "等待研究候选"}</small></div></article>
      <article className={branchClass(branchByScenario.fin_quant?.status)}><span>04</span><div><strong>联合优化</strong><small>{statusText(branchByScenario.fin_quant?.status)} · 仅冠军变化时触发</small></div></article>
      <article className={["paper", "complete"].includes(stage) ? "ready" : ["portfolio_selection", "formal_backtest", "final_oos"].includes(stage) ? "running" : "waiting"}><span>05</span><div><strong>风险与一次 OOS</strong><small>{["formal_backtest", "final_oos"].includes(stage) ? "唯一冻结冠军正在消费最终样本" : stage === "portfolio_selection" ? "预最终区间比较 TopK 与行业中性 QP" : "前置验证通过后只打开一次"}</small></div></article>
      <article className={latestSimulation ? "ready" : stage === "paper" ? "running" : "waiting"}><span>06</span><div><strong>模拟与每日候选</strong><small>{latestSimulation ? `${latestSimulation.name} · ${statusText(latestSimulation.status)}` : "硬门禁通过后自动创建"}</small></div></article>
    </section>

    <section className="autopilot-summary autopilot-summary-six">
      <article><span>当前阶段</span><strong>{STAGE_LABELS[stage] ?? stage}</strong><small>{timeText(cycle?.updated_at)}</small></article>
      <article><span>竞赛进度</span><strong>{tournament ? `${completedTrials}/${tournament.max_trials}` : "—/—"}</strong><small>{(trialCounts.running ?? 0) + (trialCounts.queued ?? 0)} 项执行中 · {trialCounts.rejected ?? 0} 项未晋级</small></article>
      <article><span>当前冠军</span><strong className="autopilot-summary-name">{selectedTrial ? selectedTrial.model_family ?? selectedTrial.feature_set_id ?? selectedTrial.name : "等待产生"}</strong><small>{selectedTrial?.feature_set_id ?? "预最终验证后冻结"}</small></article>
      <article><span>模拟盘 NAV</span><strong>{latestSimulation?.latest_nav?.nav?.toFixed(4) ?? latestSimulation?.nav?.toFixed(4) ?? "—"}</strong><small>{latestSimulation?.latest_nav ? `${latestSimulation.latest_nav.trade_date} · ${pct(latestSimulation.latest_nav.daily_return)}` : "尚未建立合格账户"}</small></article>
      <article><span>模拟持仓</span><strong>{candidateCount || "—"}</strong><small>{latestSimulation?.latest_batch?.trade_date ?? latestSimulation?.latest_batch?.signal_date ?? "模拟盘建立后生成；不代表实盘建议"}</small></article>
      <article><span>需要你处理</span><strong>{blockers.length}</strong><small>{blockers.length ? "只处理真正异常" : "当前无需操作"}</small></article>
    </section>

    {blockers.length ? <section className="autopilot-blockers"><div><span className="status-chip">BLOCKED</span><h3>当前阻断</h3></div><ul>{blockers.slice(0, 5).map((item) => <li key={item}>{item}</li>)}</ul><button className="action-button action-danger" onClick={() => onNavigate(9)}>打开异常与运行记录</button></section> : null}

    <section className="autopilot-paper-target workspace-card">
      <div className="panel-heading"><div><p className="eyebrow">PAPER TARGETS · READ ONLY</p><h3>今日模拟候选与目标权重</h3><p>{paperTarget?.message ?? "正在读取已冻结的 Qlib 模拟订单计划。"}</p></div><span className={`state ${paperTarget?.status === "ready" ? "ready" : paperTarget?.status?.startsWith("blocked") ? "failed" : "partial"}`}>{paperTarget?.status === "ready" ? "模拟候选" : paperTarget?.status === "blocked_invalid_order_plan" ? "已阻断" : "等待订单计划"}</span></div>
      {paperTarget?.status === "ready" ? <>
        <div className="autopilot-paper-target-meta"><span>信号日 <strong>{paperTarget.signal_date}</strong></span><span>拟执行日 <strong>{paperTarget.trade_date}</strong></span><span>现金目标 <strong>{pct(paperTarget.cash_weight)}</strong></span><span>批次 <code>{paperTarget.batch?.id.slice(0, 12)}</code></span></div>
        <div className="table-wrap"><table className="portfolio-table compact-ledger-table"><thead><tr><th>证券</th><th>目标权重</th><th>上次目标</th><th>调整</th><th>依据</th></tr></thead><tbody>{paperTarget.targets.slice(0, 30).map((item) => <tr key={item.instrument}><td><code>{item.instrument}</code></td><td>{pct(item.target_weight)}</td><td>{pct(item.previous_weight)}</td><td>{({ add: "新增", remove: "移除", increase: "增加", decrease: "减少", hold: "维持" } as Record<string, string>)[item.action]}</td><td>{item.reason}</td></tr>)}</tbody></table></div>
        <small className="autopilot-paper-target-note">这是隔离模拟盘的冻结目标与权重变化，不会创建正式推荐、订单或实盘权限。</small>
      </> : <div className="empty">{paperTarget?.blocker ? `订单计划校验失败：${paperTarget.blocker}` : "没有合格的模拟账户或订单计划时，系统保持等待，不生成示例股票。"}</div>}
    </section>

    <section className="autopilot-help"><div><h3>无需每天点按钮</h3><p>系统自动更新数据、生成模拟候选与目标仓位、执行 T+1 模拟成交、费用和 NAV；交易日研究因子、按新研报触发研报研究、每月滚动模型竞赛。只有 PIT、独立复算、统计、多重检验、成本、风险和最终 OOS 全部通过后才进入模拟盘。运行满 {autopilot?.real_trading.minimum_paper_calendar_days ?? 183} 天后也只提示人工复核。</p></div><div className="autopilot-help-actions"><button className="action-button action-ghost" onClick={() => onOpenAdvanced(2)}>查看竞赛试验</button><button className="action-button action-ghost" onClick={() => onOpenAdvanced(5)}>查看因子库</button></div></section>
  </div>;
}
