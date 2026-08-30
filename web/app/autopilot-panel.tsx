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

type MarketPermissions = {
  main_board: boolean;
  star_market: boolean;
  chi_next: boolean;
  beijing_exchange: boolean;
  etf: boolean;
};
type InvestorProfile = {
  id: string;
  version: number;
  initial_capital: string | number;
  risk_profile: string;
  min_cash_weight: number;
  max_gross_exposure: number;
  market_permissions: MarketPermissions;
};
type InvestorProfileResponse = {
  configured: boolean;
  profile: InvestorProfile | null;
  required_before_simulation?: boolean;
};
type EvidenceCheck = { observed: number; threshold: number; passed: boolean };
type AdviceEvidence = {
  status: string;
  passed: boolean;
  reasons?: string[];
  checks?: Record<string, EvidenceCheck>;
  maturity?: {
    status: "mature" | "accumulating";
    label: string;
    passed: boolean;
    recommendation_blocking: false;
    checks: Record<string, EvidenceCheck>;
  };
};
type AdviceSignal = {
  instrument: string;
  action: "BUY" | "ADD" | "HOLD" | "REDUCE" | "EXIT" | "NO_ACTION";
  account_action?: "BUY" | "ADD" | "HOLD" | "REDUCE" | "EXIT" | "NO_ACTION" | null;
  target_weight?: number | null;
  target_position_quantity?: number | null;
  trade_quantity?: number | null;
  execution_state?: string | null;
  effective_date?: string | null;
  validity_sessions?: number | null;
  review_date_estimate?: string | null;
  review_date_is_exchange_calendar?: boolean;
  holding_age_sessions?: number | null;
  reason?: { summary?: string; signals?: string[] } | string | null;
  risks?: string[];
  invalidation?: string[];
  evidence_state?: string;
  cannot_buy_reasons?: string[];
  execution_constraints?: Record<string, unknown>;
};
type AdviceCard = {
  horizon: "short_1_5d" | "swing_1_6m" | "long_1_3y";
  title: string;
  holding: string;
  research_cadence: string;
  stage: "research" | "backtest" | "simulation_validation" | "verified" | "restricted" | "suspended" | "retired";
  stage_label: string;
  strategy?: { id: string; name: string; version: number } | null;
  health: string;
  evidence: AdviceEvidence;
  is_investment_advice: boolean;
  data_cutoff?: string | null;
  signals: AdviceSignal[];
  action: AdviceSignal["action"];
  simulation_action?: AdviceSignal["action"] | null;
  safe_mode_risk_exits?: AdviceSignal[];
  veto_reasons?: string[];
};
type UnifiedAccount = {
  status: string;
  action: string;
  reason?: string;
  decision_date?: string;
  inputs_as_of?: string;
  cash_weight?: number;
  verified_horizons?: string[];
  targets?: Array<Record<string, unknown>>;
  trades?: Array<Record<string, unknown>>;
  accounting_rule?: string;
  safe_mode_risk_reductions?: Array<Record<string, unknown>>;
};
type PlatformGate = {
  status: "open" | "safe_mode";
  active: boolean;
  reason?: string | null;
  source?: string | null;
  triggered_at?: string | null;
};
type AdviceToday = {
  generated_at: string;
  data_cutoff?: string | null;
  onboarding_required: boolean;
  investor_profile?: InvestorProfile | null;
  cards: AdviceCard[];
  unified_account: UnifiedAccount;
  platform_gate?: PlatformGate;
  advice_available: boolean;
  execution_contract: {
    signal: string;
    earliest_fill: string;
    intraday_claims: false;
    real_broker_orders: false;
  };
  disclaimer: string;
};

const EMPTY_PERMISSIONS: MarketPermissions = {
  main_board: false,
  star_market: false,
  chi_next: false,
  beijing_exchange: false,
  etf: false,
};
const ACTION_LABELS: Record<string, string> = {
  BUY: "买入",
  ADD: "加仓",
  HOLD: "持有",
  REDUCE: "减仓",
  EXIT: "退出",
  NO_ACTION: "不操作",
  REBALANCE: "按净额调仓",
};
const HORIZON_WEIGHTS: Record<AdviceCard["horizon"], string> = {
  short_1_5d: "20% 预算",
  swing_1_6m: "40% 预算",
  long_1_3y: "40% 预算",
};
const EVIDENCE_LABELS: Record<string, string> = {
  forward_trading_days: "前向交易日",
  decision_batches: "有效决策",
  completed_cycles: "完整交易闭环",
  review_events: "定期复核",
  financial_report_reviews: "财报复核",
  data_completeness: "数据完整度",
  reconciliation_rate: "账实一致率",
};

function money(value: string | number | null | undefined) {
  const amount = Number(value);
  if (!Number.isFinite(amount)) return "—";
  return new Intl.NumberFormat("zh-CN", { style: "currency", currency: "CNY", maximumFractionDigits: 0 }).format(amount);
}

function reasonText(reason: AdviceSignal["reason"]) {
  if (typeof reason === "string") return reason;
  return reason?.summary || reason?.signals?.[0] || "策略规则满足，但暂时没有更详细的解释。";
}

function evidenceProgress(evidence: AdviceEvidence) {
  if (evidence.passed) return 100;
  const checks = Object.entries(evidence.checks ?? {}).filter(([name, check]) =>
    name in EVIDENCE_LABELS && Number(check.threshold) > 0,
  );
  if (!checks.length) return 0;
  return Math.round(Math.min(...checks.map(([, check]) =>
    Math.min(1, Math.max(0, Number(check.observed) / Number(check.threshold))),
  )) * 100);
}

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

function EvidenceSummary({ evidence }: { evidence: AdviceEvidence }) {
  const checks = Object.entries(evidence.checks ?? {}).filter(([name]) => name in EVIDENCE_LABELS);
  const progress = evidenceProgress(evidence);
  return <div className="novice-evidence">
    <div className="novice-evidence-head">
      <span>证据成熟度</span>
      <strong>{evidence.passed ? "门槛已通过" : `${progress}%`}</strong>
    </div>
    <div className="novice-evidence-track" aria-label={`证据成熟度 ${progress}%`}><i style={{ width: `${progress}%` }} /></div>
    {checks.length ? <div className="novice-evidence-checks">
      {checks.slice(0, 5).map(([name, check]) => <span className={check.passed ? "passed" : ""} key={name}>
        {EVIDENCE_LABELS[name]} {Number(check.observed).toLocaleString("zh-CN")}/{Number(check.threshold).toLocaleString("zh-CN")}
      </span>)}
    </div> : <small>{evidence.reasons?.[0] ?? "等待策略进入隔离模拟盘后开始累计。"}</small>}
    {evidence.maturity ? <small className={evidence.maturity.passed ? "maturity mature" : "maturity"}>
      {evidence.maturity.label} · 仅表示真实前向运行资历，不阻断已通过12个月门槛的长线推荐
    </small> : null}
  </div>;
}

function SignalRow({ signal, simulationOnly }: { signal: AdviceSignal; simulationOnly: boolean }) {
  const accountAction = signal.account_action ?? signal.action;
  const displayedAction = simulationOnly ? signal.action : accountAction;
  const executionWaiting = signal.execution_state === "WAIT" || signal.execution_state === "BLOCKED";
  const constraintAction = accountAction === "NO_ACTION" ? signal.action : accountAction;
  const constraintHeading = ["REDUCE", "EXIT"].includes(constraintAction)
    ? "为什么现在不能卖"
    : "为什么现在不能买";
  const tradeQuantityLabel = executionWaiting
    ? "账户当前不可执行"
    : accountAction === "BUY" || accountAction === "ADD"
    ? "账户可买数量"
    : accountAction === "REDUCE" || accountAction === "EXIT"
      ? "账户可卖数量"
      : "账户本次交易";
  return <article className="novice-signal">
    <div className="novice-signal-main">
      <div>
        <code>{signal.instrument}</code>
        <span className={`novice-action action-${displayedAction.toLowerCase()}`}>{simulationOnly ? "模拟" : ""}{ACTION_LABELS[displayedAction]}</span>
      </div>
      <strong>{pct(signal.target_weight)}</strong>
      <small>目标仓位</small>
    </div>
    <p>{reasonText(signal.reason)}</p>
    <div className="novice-signal-facts">
      <span>账户目标持仓 <strong>{signal.target_position_quantity == null ? "等待账户换算" : `${signal.target_position_quantity} 股`}</strong></span>
      <span>{tradeQuantityLabel} <strong>{executionWaiting ? "0 股（等待条件）" : signal.trade_quantity == null ? "等待执行计划" : `${signal.trade_quantity} 股`}</strong></span>
      <span>执行日 <strong>{signal.effective_date ?? "等待下一交易日"}</strong></span>
      <span>有效期 <strong>{signal.validity_sessions ? `${signal.validity_sessions} 个交易日` : "按策略复核"}</strong></span>
      <span>复核日 <strong>{signal.review_date_estimate ?? "等待交易日历"}{signal.review_date_is_exchange_calendar === false ? "（估算）" : ""}</strong></span>
    </div>
    <div className="novice-signal-guardrails">
      <span><b>主要风险</b>{signal.risks?.[0] ?? "市场变化可能使信号失效"}</span>
      <span><b>失效条件</b>{signal.invalidation?.[0] ?? "策略规则或交易资格失效"}</span>
    </div>
    {signal.cannot_buy_reasons?.length ? <div className="novice-cannot-buy">
      <b>{constraintHeading}</b>
      {signal.cannot_buy_reasons.map((reason) => <span key={reason}>{reason}</span>)}
    </div> : null}
    {simulationOnly ? <div className="simulation-boundary">仅供隔离模拟验证，不是荐股，也不会进入统一账户。</div> : null}
  </article>;
}

function HorizonCard({ card }: { card: AdviceCard }) {
  const simulationOnly = !card.is_investment_advice;
  const visibleAction = card.is_investment_advice ? card.action : "NO_ACTION";
  const simulatedAction = card.stage === "simulation_validation" ? card.simulation_action : null;
  const safeModeRiskExits = card.safe_mode_risk_exits ?? [];
  return <section className={`novice-horizon-card stage-${card.stage}`}>
    <header>
      <div>
        <p className="eyebrow">{card.horizon.replaceAll("_", " · ")}</p>
        <h3>{card.title}<span>{card.holding}</span></h3>
      </div>
      <span className={`novice-stage stage-${card.stage}`}>{card.stage_label}</span>
    </header>
    <div className="novice-card-decision">
      <div>
        <span>今天的正式动作</span>
        <strong className={`decision-${visibleAction.toLowerCase()}`}>{ACTION_LABELS[visibleAction]}</strong>
      </div>
      <div><span>资金预算</span><strong>{HORIZON_WEIGHTS[card.horizon]}</strong></div>
      <div><span>策略</span><strong>{card.strategy ? `${card.strategy.name} · v${card.strategy.version}` : "尚未产生"}</strong></div>
    </div>
    {simulatedAction && simulatedAction !== "NO_ACTION" ? <div className="novice-simulation-callout">
      模拟盘正在观察：<strong>{ACTION_LABELS[simulatedAction]}</strong>。这不是已验证荐股。
    </div> : null}
    {safeModeRiskExits.length ? <div className="novice-simulation-callout">
      安全模式风险信息：{safeModeRiskExits.slice(0, 5).map((signal) => `${signal.instrument} ${ACTION_LABELS[signal.action]}`).join("、")}。仅保留提醒，当前不可执行。
    </div> : null}
    <EvidenceSummary evidence={card.evidence} />
    <div className="novice-card-meta">
      <span>数据截止 <strong>{card.data_cutoff ?? "等待完整收盘数据"}</strong></span>
      <span>研究频率 <strong>{card.research_cadence}</strong></span>
    </div>
    {card.signals.length ? <div className="novice-signals">
      <div className="novice-signals-title">
        <strong>{card.is_investment_advice ? "当前股票与操作" : "模拟验证中的观察对象"}</strong>
        <span>{card.signals.length} 只</span>
      </div>
      {card.signals.slice(0, 5).map((signal, index) => <SignalRow key={`${signal.instrument}-${index}`} signal={signal} simulationOnly={simulationOnly} />)}
      {card.signals.length > 5 ? <small className="novice-more">其余 {card.signals.length - 5} 只请到模拟账本查看。</small> : null}
    </div> : <div className="novice-no-action">
      <strong>当前没有需要操作的股票</strong>
      <span>{card.veto_reasons?.[0] ?? "没有股票同时满足收益、成本、风险和交易资格门槛，资金保留为现金。"}</span>
    </div>}
  </section>;
}

function recordText(record: Record<string, unknown>, keys: string[], fallback = "—") {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" || typeof value === "number") return String(value);
  }
  return fallback;
}

function UnifiedAccountCard({ account }: { account: UnifiedAccount }) {
  const trades = account.trades ?? [];
  const targets = account.targets ?? [];
  const ready = account.status === "ready";
  return <section className={`novice-account-card ${ready ? "ready" : "waiting"}`}>
    <div className="novice-account-heading">
      <div><p className="eyebrow">ONE ACCOUNT · THREE HORIZONS</p><h3>统一账户建议</h3></div>
      <span className={`novice-action account-action-${account.action.toLowerCase()}`}>{ACTION_LABELS[account.action] ?? account.action}</span>
    </div>
    <p>{account.reason ?? account.accounting_rule ?? "三周期先独立形成目标，再在账户层合并同一股票的买卖。"}</p>
    <div className="novice-account-metrics">
      <span>短 / 中 / 长预算<strong>20% / 40% / 40%</strong></span>
      <span>最低现金<strong>{account.cash_weight == null ? "10%" : pct(account.cash_weight)}</strong></span>
      <span>数据截止<strong>{account.inputs_as_of ?? "等待三周期验证"}</strong></span>
      <span>执行日<strong>{account.decision_date ?? "尚未生成"}</strong></span>
    </div>
    {trades.length ? <div className="novice-account-trades">
      {trades.slice(0, 8).map((trade, index) => {
        const action = recordText(trade, ["action", "side"], "REBALANCE").toUpperCase();
        const quantity = recordText(trade, ["trade_quantity", "quantity", "delta_quantity"], "0");
        return <div key={`${recordText(trade, ["instrument"])}-${index}`}>
          <code>{recordText(trade, ["instrument"])}</code>
          <strong>{ACTION_LABELS[action] ?? action}</strong>
          <span>{quantity} 股</span>
        </div>;
      })}
    </div> : targets.length ? <div className="novice-account-trades">
      {targets.slice(0, 8).map((target, index) => <div key={`${recordText(target, ["instrument"])}-${index}`}>
        <code>{recordText(target, ["instrument"])}</code><strong>目标</strong><span>{recordText(target, ["target_weight", "weight", "target_position_quantity"])}</span>
      </div>)}
    </div> : <div className="novice-account-empty">没有完整的三周期净额计划时，系统保持现金，不会拼凑一份建议。</div>}
    <small>同一股票会在这里合并为一次账户动作；某个周期暂停时，其预算留在现金中，不挪给其他周期。</small>
  </section>;
}

function InvestorOnboarding({
  api,
  profile,
  initialOpen,
  onSaved,
}: {
  api: string;
  profile: InvestorProfile | null;
  initialOpen: boolean;
  onSaved: () => Promise<void>;
}) {
  const [open, setOpen] = useState(initialOpen);
  const [capital, setCapital] = useState(profile ? String(profile.initial_capital) : "");
  const [permissions, setPermissions] = useState<MarketPermissions>(profile?.market_permissions ?? EMPTY_PERMISSIONS);
  const [confirmed, setConfirmed] = useState(Boolean(profile));
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  async function save() {
    const amount = Number(capital);
    if (!Number.isFinite(amount) || amount <= 0) {
      setMessage("请填写大于 0 的模拟本金；系统没有 50 万元起步限制。");
      return;
    }
    if (!confirmed) {
      setMessage("请逐项确认证券权限。未勾选的市场会明确记录为无权限。");
      return;
    }
    setBusy(true);
    setMessage("");
    try {
      const response = await apiFetch(`${api}/api/investor-profile`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          initial_capital: amount,
          risk_profile: "balanced",
          min_cash_weight: 0.10,
          max_gross_exposure: 0.90,
          market_permissions: permissions,
          actor: "web-investor-onboarding",
        }),
      });
      const body = await response.json() as { detail?: string | { message?: string } };
      if (!response.ok) {
        const detail = typeof body.detail === "string" ? body.detail : body.detail?.message;
        throw new Error(detail ?? "模拟账户设置保存失败");
      }
      await onSaved();
      setOpen(false);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "模拟账户设置保存失败");
    } finally {
      setBusy(false);
    }
  }

  if (!open && profile) return <section className="novice-profile-summary">
    <div><span>模拟本金</span><strong>{money(profile.initial_capital)}</strong></div>
    <div><span>风险档位</span><strong>平衡型 · 至少 10% 现金</strong></div>
    <div><span>已允许市场</span><strong>{Object.values(profile.market_permissions).filter(Boolean).length} / 5</strong></div>
    <button type="button" onClick={() => { setCapital(String(profile.initial_capital)); setPermissions(profile.market_permissions); setConfirmed(true); setOpen(true); }}>修改模拟设置</button>
  </section>;

  const permissionOptions: Array<[keyof MarketPermissions, string, string]> = [
    ["main_board", "沪深主板", "普通 A 股权限"],
    ["star_market", "科创板", "需要券商另行开通"],
    ["chi_next", "创业板", "需要券商另行开通"],
    ["beijing_exchange", "北交所", "需要券商另行开通"],
    ["etf", "境内 ETF", "仅使用规则验证白名单"],
  ];
  return <section className="novice-onboarding">
    <div className="novice-onboarding-copy">
      <span className="status-chip">首次使用 · 必填</span>
      <h2>先建立你的隔离模拟账户</h2>
      <p>只填写模拟本金和你真实拥有的证券权限。这里不会连接券商，也不会下真实订单；资金不足买 100 股时会保留现金并说明原因。</p>
      <div><span>默认分配</span><strong>短线 20% · 中线 40% · 长线 40%</strong></div>
      <div><span>平衡型边界</span><strong>不融资、不做空 · 最大总暴露 90%</strong></div>
    </div>
    <div className="novice-onboarding-form">
      <label>模拟本金（人民币）<input inputMode="decimal" min="0.01" step="0.01" type="number" placeholder="例如 100000" value={capital} onChange={(event) => setCapital(event.target.value)} /></label>
      <fieldset>
        <legend>你已经开通哪些证券权限？</legend>
        {permissionOptions.map(([key, label, hint]) => <label className="permission-choice" key={key}>
          <input type="checkbox" checked={permissions[key]} onChange={(event) => setPermissions({ ...permissions, [key]: event.target.checked })} />
          <span><strong>{label}</strong><small>{hint}</small></span>
        </label>)}
      </fieldset>
      <label className="permissions-confirm"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /><span>我已逐项确认；未勾选的市场表示无权限。</span></label>
      {message ? <div className="novice-form-error">{message}</div> : null}
      <div className="novice-onboarding-actions">
        {profile ? <button type="button" className="action-button action-ghost" onClick={() => setOpen(false)}>取消</button> : null}
        <button type="button" className="action-button action-primary" disabled={busy} onClick={() => void save()}>{busy ? "正在保存…" : "保存并开始模拟"}</button>
      </div>
    </div>
  </section>;
}

function NoviceAdvicePanel({ api, onNavigate }: { api: string; onNavigate: (index: number) => void }) {
  const [advice, setAdvice] = useState<AdviceToday | null>(null);
  const [profileState, setProfileState] = useState<InvestorProfileResponse | null>(null);
  const [loadState, setLoadState] = useState<"loading" | "ready" | "error">("loading");
  const [warning, setWarning] = useState("");

  const load = useCallback(async () => {
    const [adviceResult, profileResult] = await Promise.allSettled([
      jsonResponse<AdviceToday>(apiFetch(`${api}/api/advice/today`, { cache: "no-store" })),
      jsonResponse<InvestorProfileResponse>(apiFetch(`${api}/api/investor-profile`, { cache: "no-store" })),
    ]);
    if (adviceResult.status === "fulfilled") setAdvice(adviceResult.value);
    if (profileResult.status === "fulfilled") setProfileState(profileResult.value);
    if (adviceResult.status === "rejected" && profileResult.status === "rejected") {
      setLoadState("error");
      setWarning("暂时无法读取今日结果。系统不会用旧信号冒充今天的荐股。");
    } else {
      setLoadState("ready");
      setWarning(adviceResult.status === "rejected" ? "今日策略结果暂不可用；已停止展示任何正式动作。" : "");
    }
  }, [api]);
  usePolling(load, 15_000);

  const profile = profileState ? profileState.profile : advice?.investor_profile ?? null;
  const onboardingRequired = profileState ? !profileState.configured : advice?.onboarding_required ?? false;

  if (loadState === "loading" && !advice && !profileState) return <div className="novice-loading"><i /><strong>正在核对今天的数据、策略证据和模拟账户</strong><span>在核对完成前不会显示买入或卖出动作。</span></div>;

  return <div className="novice-advice-page">
    {warning ? <div className="notice novice-warning">{warning}</div> : null}
    {advice?.platform_gate?.active ? <div className="notice novice-warning">平台安全模式已开启：{advice.platform_gate.reason ?? "正式荐股和模拟订单已暂停"}。已有减仓或退出信号只作为风险信息展示。</div> : null}
    <InvestorOnboarding api={api} profile={profile} initialOpen={onboardingRequired} onSaved={load} />
    <section className={`novice-hero ${advice?.advice_available ? "verified" : "waiting"}`}>
      <div>
        <span className="status-chip">TODAY · AFTER CLOSE</span>
        <h2>{onboardingRequired ? "完成设置后开始三周期模拟" : advice?.advice_available ? "今天的账户动作已经生成" : "今天没有正式荐股，系统继续验证"}</h2>
        <p>{advice?.advice_available ? "只有通过严格前向门槛的周期才会进入下方统一账户。" : "研究、回测或模拟中的信号不会冒充荐股；没有合格机会时，正确答案就是持有现金。"}</p>
      </div>
      <div className="novice-hero-facts">
        <span>数据截止<strong>{advice?.data_cutoff ?? "等待最新完整交易日"}</strong></span>
        <span>信号时间<strong>{advice?.execution_contract.signal ?? "D 日收盘后"}</strong></span>
        <span>最早执行<strong>{advice?.execution_contract.earliest_fill ?? "D+1 开盘"}</strong></span>
      </div>
    </section>

    <div className="novice-horizon-grid">
      {advice?.cards?.map((card) => <HorizonCard card={card} key={card.horizon} />)}
      {!advice?.cards?.length ? <div className="novice-results-unavailable">今日结果不可用，正式动作统一为“不操作”。</div> : null}
    </div>

    <UnifiedAccountCard account={advice?.unified_account ?? { status: "unavailable", action: "NO_ACTION", reason: "今日结果不可用，账户保持现金和原持仓。" }} />

    <section className="novice-boundary-note">
      <div><strong>你只需要看两件事</strong><span>每个周期是否“已验证”，以及统一账户最终让你买、卖还是不操作。</span></div>
      <div><strong>短线也不是盘中追涨</strong><span>目前只使用完整收盘数据，D 日盘后计算，最早 D+1 执行。</span></div>
      <button className="action-button action-secondary" type="button" onClick={() => onNavigate(8)}>打开模拟账本</button>
    </section>
    <p className="novice-disclaimer">{advice?.disclaimer ?? "系统仅运行模拟盘；历史和模拟表现不保证未来收益。"}</p>
  </div>;
}

export function AutopilotPanel({
  api, onNavigate, onOpenAdvanced, advancedMode = false,
}: {
  api: string; onNavigate: (index: number) => void; onOpenAdvanced: (index: number) => void; advancedMode?: boolean;
}) {
  return advancedMode
    ? <AdvancedAutopilotPanel api={api} onNavigate={onNavigate} onOpenAdvanced={onOpenAdvanced} />
    : <NoviceAdvicePanel api={api} onNavigate={onNavigate} />;
}

function AdvancedAutopilotPanel({
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
