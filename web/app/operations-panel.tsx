"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";
import type { AuthUser } from "./auth-panel";

type Portfolio = { id: string; name: string; status: string };
type Schedule = {
  id: string; name: string; kind: string; status: string; timezone: string;
  desired_status: string; suspension_reason?: string | null;
  run_time: string; next_run_at: string; payload: Record<string, unknown>;
};
type ScheduleRun = {
  id: string; schedule_name: string; kind: string; scheduled_for: string;
  status: string; attempts: number; message?: string | null; job_id?: string | null;
};
type Alert = {
  id: string; severity: string; category: string; title: string; message: string;
  status: string; delivery_status: string; created_at: string;
};
type Job = {
  id: string; kind: string; status: string; created_at: string; started_at?: string | null;
  finished_at?: string | null; error?: string | null;
};
type SchedulerStatus = {
  status: string; last_tick?: string | null; last_error?: string | null;
  stats?: Record<string, number>;
};
type HealthComponent = { status: string; message: string; age_days?: number; queued?: number; running?: number; stale_running?: number };
type HealthSnapshot = { id: number; status: string; recorded_at: string; age_seconds?: number; components: Record<string, HealthComponent>; summary: { ok_count: number; problem_count: number; bootstrap_count: number } };
type HealthResponse = { latest: HealthSnapshot | null; history: HealthSnapshot[] };
type ReadinessCheck = {
  id: string; title: string; status: "pass" | "block"; evidence: string;
  remediation?: string | null;
};
type ReadinessProfile = {
  id: string; title: string; status: "ready" | "blocked"; passed: number; total: number;
  blocker_count: number; checks: ReadinessCheck[];
};
type ReadinessResponse = {
  generated_at?: string; highest_ready_profile?: string | null; live_trading_supported: boolean;
  profiles: ReadinessProfile[];
};
type BrokerDestination = {
  id: string; name: string; account_ref: string; portfolio_id: string; environment: string; status: string;
  activation_requested_by?: string | null; activated_by?: string | null; updated_at: string;
};
type BrokerOutbox = {
  id: string; destination_id: string; batch_id: string; status: string; attempts: number;
  broker_order_id?: string | null; created_by: string; approved_by?: string | null; updated_at: string;
  payload: { instrument?: string; side?: string; quantity?: number };
};
type BrokerResponse = {
  readiness: {
    status: string; mode: string; live_supported: boolean; gateway_configured: boolean;
    destination_counts: Record<string, number>; outbox_counts: Record<string, number>;
    max_order_notional: number;
  };
  destinations: BrokerDestination[];
  outbox: BrokerOutbox[];
  reconciliations: Array<{
    id: string; destination_id: string; status: string; broker_as_of: string; created_at: string;
    differences: Array<{ type: string; instrument?: string; difference?: number }>;
  }>;
};
type ManagedUser = AuthUser & {
  id: string; active: boolean; failed_login_attempts: number;
  locked_until?: string | null; last_login_at?: string | null; created_at: string;
};
type AuditEvent = {
  id: number; username: string; action: string; method: string; path: string;
  status_code: number; created_at: string;
};

const timeText = (value?: string | null) => value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "—";
const stateClass = (status: string) => ["succeeded", "enqueued", "active", "ok", "acknowledged", "resolved"].includes(status) ? "ready" : ["failed", "missed", "open", "unavailable", "degraded"].includes(status) ? "failed" : "partial";
const scheduleKindText: Record<string, string> = {
  incremental_sync: "数据增量同步",
  data_pipeline: "全数据恢复流水线",
  ashare_5m_sync: "全 A 股 5 分钟增量",
  rdagent_research: "RD-Agent 因子研究",
  paper_rebalance: "模拟组合再平衡",
  pair_paper_rebalance: "配对模拟再平衡",
  broker_reconcile: "券商账户对账",
};

export function OperationsPanel({ api, currentUser }: { api: string; currentUser: AuthUser }) {
  const [view, setView] = useState("status");
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [runs, setRuns] = useState<ScheduleRun[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [scheduler, setScheduler] = useState<SchedulerStatus>({ status: "连接中" });
  const [health, setHealth] = useState<HealthResponse>({ latest: null, history: [] });
  const [readiness, setReadiness] = useState<ReadinessResponse>({
    highest_ready_profile: null, live_trading_supported: false, profiles: [],
  });
  const [broker, setBroker] = useState<BrokerResponse>({
    readiness: { status: "disabled", mode: "disabled", live_supported: false, gateway_configured: false, destination_counts: {}, outbox_counts: {}, max_order_notional: 0 },
    destinations: [], outbox: [], reconciliations: [],
  });
  const [syncName, setSyncName] = useState("每日数据增量同步");
  const [syncTime, setSyncTime] = useState("18:00");
  const [syncKind, setSyncKind] = useState<"incremental_sync" | "data_pipeline" | "ashare_5m_sync">("incremental_sync");
  const [profile, setProfile] = useState("full");
  const [syncLookbackDays, setSyncLookbackDays] = useState(7);
  const [syncBuildQlib, setSyncBuildQlib] = useState(true);
  const [syncSnapshotStart, setSyncSnapshotStart] = useState("2024-01-01");
  const [syncMisfireGrace, setSyncMisfireGrace] = useState(3600);
  const [portfolioName, setPortfolioName] = useState("每日模拟组合再平衡");
  const [portfolioId, setPortfolioId] = useState("");
  const [portfolioTime, setPortfolioTime] = useState("15:30");
  const [portfolioSlippage, setPortfolioSlippage] = useState(0.0005);
  const [portfolioMisfireGrace, setPortfolioMisfireGrace] = useState(1800);
  const [message, setMessage] = useState("");
  const [managedUsers, setManagedUsers] = useState<ManagedUser[]>([]);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [newUsername, setNewUsername] = useState("");
  const [newDisplayName, setNewDisplayName] = useState("");
  const [newRole, setNewRole] = useState("viewer");
  const [newUserPassword, setNewUserPassword] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [replacementPassword, setReplacementPassword] = useState("");
  const [brokerName, setBrokerName] = useState("QMT 集成沙箱");
  const [brokerAccount, setBrokerAccount] = useState("SIM-500W");
  const [brokerPortfolioId, setBrokerPortfolioId] = useState("");
  const [brokerAlgorithm, setBrokerAlgorithm] = useState("twap");
  const [brokerSliceMinutes, setBrokerSliceMinutes] = useState(20);
  const [brokerMaxSlices, setBrokerMaxSlices] = useState(24);
  const [brokerMaxParticipation, setBrokerMaxParticipation] = useState(0.01);
  const [brokerDestinationId, setBrokerDestinationId] = useState("");
  const [brokerBatchId, setBrokerBatchId] = useState("");
  const [brokerScheduleName, setBrokerScheduleName] = useState("每日券商沙箱对账");
  const [brokerScheduleTime, setBrokerScheduleTime] = useState("15:45");
  const [brokerScheduleMisfireGrace, setBrokerScheduleMisfireGrace] = useState(900);

  async function load() {
    try {
      const responses = await Promise.all([
        apiFetch(`${api}/api/portfolios`, { cache: "no-store" }),
        apiFetch(`${api}/api/schedules`, { cache: "no-store" }),
        apiFetch(`${api}/api/schedule-runs`, { cache: "no-store" }),
        apiFetch(`${api}/api/alerts`, { cache: "no-store" }),
        apiFetch(`${api}/api/jobs?limit=100`, { cache: "no-store" }),
        apiFetch(`${api}/api/scheduler/status`, { cache: "no-store" }),
        apiFetch(`${api}/api/operations/health?limit=24`, { cache: "no-store" }),
        apiFetch(`${api}/api/operations/readiness`, { cache: "no-store" }),
      ]);
      if (responses.some((response) => !response.ok)) throw new Error("API unavailable");
      const nextPortfolios: Portfolio[] = await responses[0].json();
      setPortfolios(nextPortfolios);
      setSchedules(await responses[1].json());
      setRuns(await responses[2].json());
      setAlerts(await responses[3].json());
      setJobs(await responses[4].json());
      setScheduler(await responses[5].json());
      setHealth(await responses[6].json());
      setReadiness(await responses[7].json());
      if (!portfolioId && nextPortfolios.length) setPortfolioId(nextPortfolios[0].id);
      if (!brokerPortfolioId && nextPortfolios.length) setBrokerPortfolioId(nextPortfolios[0].id);
      if (currentUser.role === "admin") {
        const [usersResponse, auditResponse, brokerResponse] = await Promise.all([
          apiFetch(`${api}/api/auth/users`, { cache: "no-store" }),
          apiFetch(`${api}/api/audit?limit=100`, { cache: "no-store" }),
          apiFetch(`${api}/api/broker`, { cache: "no-store" }),
        ]);
        if (usersResponse.ok) setManagedUsers(await usersResponse.json());
        if (auditResponse.ok) setAuditEvents(await auditResponse.json());
        if (brokerResponse.ok) {
          const nextBroker: BrokerResponse = await brokerResponse.json();
          setBroker(nextBroker);
          if (!brokerDestinationId && nextBroker.destinations.length) setBrokerDestinationId(nextBroker.destinations[0].id);
        }
      }
    } catch { setMessage("无法读取自动化控制面，请确认 API 与调度服务正在运行。"); }
  }

  useEffect(() => {
    const initial = window.setTimeout(load, 0); const timer = window.setInterval(load, 5000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openAlerts = useMemo(() => alerts.filter((item) => item.status === "open"), [alerts]);
  const activeSchedules = useMemo(() => schedules.filter((item) => item.status === "active"), [schedules]);
  const activeJobs = useMemo(() => jobs.filter((item) => ["queued", "running"].includes(item.status)), [jobs]);
  const healthComponents = Object.entries(health.latest?.components ?? {});
  const can = (permission: string) => currentUser.permissions.includes("*") || currentUser.permissions.includes(permission);

  async function createSchedule(body: Record<string, unknown>) {
    const response = await apiFetch(`${api}/api/schedules`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok) { setMessage(result.detail ?? "自动计划创建失败"); return false; }
    setMessage(`自动计划 ${result.name} 已启用，下一次运行：${timeText(result.next_run_at)}`); await load(); return true;
  }

  async function createSync(event: FormEvent) {
    event.preventDefault();
    await createSchedule({
      name: syncName, kind: syncKind, timezone: "Asia/Shanghai", run_time: syncTime,
      trading_days_only: true, payload: {
        profile, lookback_days: syncLookbackDays, build_qlib: syncBuildQlib,
        snapshot_start: syncSnapshotStart,
        ...(syncKind === "data_pipeline" ? { bundles: ["cn_extended_daily", "cn_funds", "cn_macro", "cn_futures", "cn_options_bonds", "hk_market", "us_market", "global_markets", "cn_institutional", "cn_governance_risk", "cn_capital_flow", "cn_fund_index_enhanced", "cn_derivatives_enhanced", "global_rates_enhanced", "research_corpus"] } : {}),
      },
      misfire_grace_seconds: syncMisfireGrace, actor: "local-operator",
    });
  }

  async function createPortfolioSchedule(event: FormEvent) {
    event.preventDefault();
    await createSchedule({
      name: portfolioName, kind: "paper_rebalance", timezone: "Asia/Shanghai", run_time: portfolioTime,
      trading_days_only: true, payload: { portfolio_id: portfolioId, slippage: portfolioSlippage },
      misfire_grace_seconds: portfolioMisfireGrace, actor: "local-operator",
    });
  }

  async function setScheduleStatus(schedule: Schedule) {
    if (schedule.payload.managed_by === "allocation_schedule_group") {
      setMessage("组合托管的子调度必须在“多策略组合”中统一操作。");
      return;
    }
    const status = schedule.status === "active" ? "paused" : "active";
    const response = await apiFetch(`${api}/api/schedules/${schedule.id}/status`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status }),
    });
    const result = await response.json();
    if (!response.ok) { setMessage(result.detail ?? "计划状态更新失败"); return; }
    setMessage(status === "paused" ? `计划 ${schedule.name} 已暂停。` : `计划 ${schedule.name} 已恢复。`); await load();
  }

  async function actOnAlert(alert: Alert, action: "acknowledge" | "resolve") {
    const response = await apiFetch(`${api}/api/alerts/${alert.id}/${action}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ actor: "local-operator" }),
    });
    const result = await response.json();
    if (!response.ok) { setMessage(result.detail ?? "告警更新失败"); return; }
    setMessage(action === "acknowledge" ? "告警已确认并保留处置痕迹。" : "告警已关闭。"); await load();
  }

  async function createUser(event: FormEvent) {
    event.preventDefault();
    const response = await apiFetch(`${api}/api/auth/users`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: newUsername, display_name: newDisplayName, role: newRole, password: newUserPassword }),
    });
    const result = await response.json();
    if (!response.ok) { setMessage(result.detail ?? "用户创建失败"); return; }
    setNewUsername(""); setNewDisplayName(""); setNewUserPassword("");
    setMessage(`用户 ${result.username} 已创建。`); await load();
  }

  async function setUserActive(user: ManagedUser) {
    const response = await apiFetch(`${api}/api/auth/users/${user.id}/status`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ active: !user.active }),
    });
    const result = await response.json();
    if (!response.ok) { setMessage(result.detail ?? "用户状态更新失败"); return; }
    setMessage(`用户 ${result.username} 已${result.active ? "启用" : "停用"}。`); await load();
  }

  async function changePassword(event: FormEvent) {
    event.preventDefault();
    const response = await apiFetch(`${api}/api/auth/password`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ current_password: currentPassword, new_password: replacementPassword }),
    });
    const result = await response.json();
    if (!response.ok) { setMessage(result.detail ?? "密码修改失败"); return; }
    setCurrentPassword(""); setReplacementPassword("");
    setMessage("密码已更新，其他设备上的会话已撤销。");
  }

  async function createBrokerDestination(event: FormEvent) {
    event.preventDefault();
    const response = await apiFetch(`${api}/api/broker/destinations`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: brokerName, account_ref: brokerAccount, portfolio_id: brokerPortfolioId, config: { execution_algorithm: brokerAlgorithm, slice_minutes: brokerSliceMinutes, max_slices: brokerMaxSlices, max_participation: brokerMaxParticipation } }),
    });
    const result = await response.json();
    if (!response.ok) { setMessage(result.detail ?? "券商沙箱目的地创建失败"); return; }
    setBrokerDestinationId(result.id); setMessage(`沙箱目的地 ${result.name} 已创建，当前保持锁定。`); await load();
  }

  async function brokerDestinationAction(destination: BrokerDestination, action: "request-activation" | "approve-activation" | "disarm") {
    const response = await apiFetch(`${api}/api/broker/destinations/${destination.id}/${action}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}),
    });
    const result = await response.json();
    if (!response.ok) { setMessage(result.detail ?? "券商目的地状态更新失败"); return; }
    setMessage(`沙箱目的地 ${result.name} 已更新为 ${result.status}。`); await load();
  }

  async function brokerBatchAction(action: "stage" | "approve" | "dispatch") {
    if (!brokerDestinationId || !brokerBatchId) return;
    const response = await apiFetch(`${api}/api/broker/destinations/${brokerDestinationId}/${action}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ batch_id: brokerBatchId }),
    });
    const result = await response.json();
    if (!response.ok) { setMessage(result.detail ?? "券商沙箱批次操作失败"); return; }
    setMessage(action === "dispatch" ? `沙箱发送完成：成功 ${result.submitted}，失败 ${result.failed}。` : `沙箱批次已执行 ${action} 操作。`); await load();
  }

  async function createBrokerSchedule(event: FormEvent) {
    event.preventDefault();
    await createSchedule({
      name: brokerScheduleName, kind: "broker_reconcile", timezone: "Asia/Shanghai",
      run_time: brokerScheduleTime, trading_days_only: true,
      payload: { destination_id: brokerDestinationId }, misfire_grace_seconds: brokerScheduleMisfireGrace,
    });
  }

  async function reconcileBrokerDestination(destination: BrokerDestination) {
    const response = await apiFetch(`${api}/api/broker/destinations/${destination.id}/reconcile`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}),
    });
    const result = await response.json();
    if (!response.ok) { setMessage(result.detail ?? "券商沙箱对账失败"); return; }
    setMessage(result.status === "matched" ? "券商沙箱资金、持仓和订单对账一致。" : `发现 ${result.differences.length} 项差异，目的地已锁定并生成严重告警。`); await load();
  }

  return <>
    {message && <div className="notice">{message}</div>}
    <div className="page-tabs" role="tablist" aria-label="系统运行工作区">
      {[["status", "运行状态"], ["automation", "计划与任务"], ["broker", "券商沙箱"], ["access", "账户与审计"]].map(([value, label]) => <button type="button" role="tab" aria-selected={view === value} className={view === value ? "active" : ""} onClick={() => setView(value)} key={value}>{label}{value === "status" && openAlerts.length ? <i>{openAlerts.length}</i> : null}</button>)}
    </div>
    {view === "status" && <>
    <section className="operations-status">
      <article><span className={`pulse ${scheduler.status === "ok" ? "ok" : ""}`} /><div><span>调度服务</span><strong>{scheduler.status}</strong><small>最近心跳 {timeText(scheduler.last_tick)}</small></div></article>
      <article><div><span>启用计划</span><strong>{activeSchedules.length}</strong><small>PostgreSQL 租约调度</small></div></article>
      <article><div><span>运行中任务</span><strong>{activeJobs.length}</strong><small>Worker 队列</small></div></article>
      <article className={openAlerts.length ? "danger-card" : ""}><div><span>待确认告警</span><strong>{openAlerts.length}</strong><small>{openAlerts.length ? "需要人工处置" : "当前正常"}</small></div></article>
    </section>
    <section className="data-panel readiness-panel">
      <div className="panel-heading"><div><p className="eyebrow">DEPLOYMENT GO / NO-GO</p><h2>正式可用验收</h2></div><div className="readiness-summary"><span>当前最高层级</span><strong>{readiness.highest_ready_profile ?? "尚未通过"}</strong><small>实盘交易：{readiness.live_trading_supported ? "已支持" : "明确禁用"}</small></div></div>
      <div className="readiness-profiles">{readiness.profiles.map((profile) => <details key={profile.id} className={profile.status === "ready" ? "readiness-profile ready" : "readiness-profile blocked"} open={profile.id === "research"}><summary><span className={`state ${profile.status === "ready" ? "ready" : "failed"}`}>{profile.status === "ready" ? "通过" : "阻塞"}</span><div><strong>{profile.title}</strong><small>{profile.passed} / {profile.total} 项通过</small></div><b>{profile.blocker_count ? `${profile.blocker_count} 个阻塞项` : "允许进入该阶段"}</b></summary><div className="readiness-checks">{profile.checks.map((check) => <article key={check.id} className={check.status}><span>{check.status === "pass" ? "✓" : "!"}</span><div><strong>{check.title}</strong><small>{check.evidence}</small>{check.remediation && <p>{check.remediation}</p>}</div></article>)}</div></details>)}</div>
      {!readiness.profiles.length && <div className="empty compact">正在生成基于数据库、任务、数据集和运行健康证据的验收结果。</div>}
    </section>
    <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">DURABLE HEALTH HISTORY</p><h2>系统健康与数据新鲜度</h2></div><span className={`state ${health.latest?.status === "ok" ? "ready" : health.latest?.status === "degraded" ? "failed" : "partial"}`}>{health.latest?.status ?? "等待首个快照"}</span></div><div className="operations-status">{healthComponents.map(([name, component]) => <article key={name} className={component.status === "degraded" || component.status === "unavailable" ? "danger-card" : ""}><span className={`pulse ${component.status === "ok" ? "ok" : ""}`} /><div><span>{name}</span><strong>{component.status}</strong><small>{component.message}</small></div></article>)}</div><div className="table-wrap"><table className="operations-table"><thead><tr><th>记录时间</th><th>总体状态</th><th>正常组件</th><th>异常组件</th><th>待初始化</th></tr></thead><tbody>{health.history.slice(0, 12).map((item) => <tr key={item.id}><td>{timeText(item.recorded_at)}</td><td><span className={`state ${stateClass(item.status)}`}>{item.status}</span></td><td>{item.summary.ok_count}</td><td>{item.summary.problem_count}</td><td>{item.summary.bootstrap_count}</td></tr>)}</tbody></table>{!health.history.length && <div className="empty compact">调度器将在首个监控周期写入健康快照。</div>}</div></section>
    </>}

    {view === "broker" && <>
    {currentUser.role === "admin" && <section className="data-panel panel-without-top-margin"><div className="panel-heading"><div><p className="eyebrow">BROKER EXECUTION BOUNDARY</p><h2>券商集成沙箱</h2></div><span className={`state ${broker.readiness.status === "ok" ? "ready" : broker.readiness.status === "unavailable" ? "failed" : "partial"}`}>{broker.readiness.status}</span></div><div className="operations-status"><article><div><span>执行模式</span><strong>{broker.readiness.mode}</strong><small>实盘支持：{broker.readiness.live_supported ? "已启用" : "明确禁用"}</small></div></article><article><div><span>签名网关</span><strong>{broker.readiness.gateway_configured ? "configured" : "locked"}</strong><small>远程网关强制 HTTPS</small></div></article><article><div><span>单笔名义金额上限</span><strong>¥{broker.readiness.max_order_notional.toLocaleString("zh-CN")}</strong><small>超限订单无法进入 outbox</small></div></article><article><div><span>待处理 outbox</span><strong>{Object.entries(broker.readiness.outbox_counts).filter(([status]) => status !== "submitted").reduce((total, [, count]) => total + count, 0)}</strong><small>幂等、可审计、显式重试</small></div></article></div>
      {can("broker:manage") && <section className="operations-forms"><form className="automation-card" onSubmit={createBrokerDestination}><div className="card-heading"><div><span>沙箱目的地</span><strong>创建锁定账户映射</strong></div><span className="status-chip">SANDBOX ONLY</span></div><label>名称<input value={brokerName} onChange={(event) => setBrokerName(event.target.value)} /></label><label>账户引用<input value={brokerAccount} onChange={(event) => setBrokerAccount(event.target.value)} /></label><label>影子组合<select value={brokerPortfolioId} onChange={(event) => setBrokerPortfolioId(event.target.value)}>{portfolios.length ? portfolios.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.status}</option>) : <option value="">尚无模拟组合</option>}</select></label><div className="form-row"><label>执行算法<select value={brokerAlgorithm} onChange={(event) => setBrokerAlgorithm(event.target.value)}><option value="twap">TWAP</option><option value="vwap">VWAP（需成交量曲线）</option></select></label><label>切片分钟<input type="number" min="5" max="30" step="5" value={brokerSliceMinutes} onChange={(event) => setBrokerSliceMinutes(Number(event.target.value))} /></label></div><div className="form-row"><label>最大切片数<input type="number" min="1" max="64" step="1" value={brokerMaxSlices} onChange={(event) => setBrokerMaxSlices(Number(event.target.value))} /></label><label>成交量参与率上限（%）<input type="number" min="0.01" max="20" step="0.1" value={brokerMaxParticipation * 100} onChange={(event) => setBrokerMaxParticipation(Number(event.target.value) / 100)} /></label></div><div className="execution-note"><b>安全边界</b><span>数据库不保存券商密钥</span><span>live 模式在配置解析阶段被拒绝</span><span>激活必须由第二名管理员批准</span></div><button className="primary" disabled={brokerName.length < 3 || brokerAccount.length < 2 || !brokerPortfolioId}>创建沙箱目的地</button></form><article className="automation-card"><div className="card-heading"><div><span>订单演练</span><strong>纸面批次重放到券商沙箱</strong></div><span className="status-chip verified">TWO PERSON</span></div><label>目的地<select value={brokerDestinationId} onChange={(event) => setBrokerDestinationId(event.target.value)}>{broker.destinations.length ? broker.destinations.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.status}</option>) : <option value="">尚无目的地</option>}</select></label><label>成功的模拟批次 ID<input value={brokerBatchId} onChange={(event) => setBrokerBatchId(event.target.value)} /></label><div className="form-row"><button className="inline-action" type="button" disabled={!brokerBatchId || !brokerDestinationId} onClick={() => brokerBatchAction("stage")}>暂存</button><button className="inline-action" type="button" disabled={!brokerBatchId || !brokerDestinationId} onClick={() => brokerBatchAction("approve")}>第二人批准</button><button className="inline-action" type="button" disabled={!brokerBatchId || !brokerDestinationId} onClick={() => brokerBatchAction("dispatch")}>发送沙箱</button></div></article></section>}
      <div className="table-wrap"><table className="operations-table"><thead><tr><th>目的地</th><th>影子组合</th><th>账户引用</th><th>状态</th><th>申请人 / 批准人</th><th>操作</th></tr></thead><tbody>{broker.destinations.map((item) => <tr key={item.id}><td><strong>{item.name}</strong><small>{item.id.slice(0, 10)}</small></td><td><code>{item.portfolio_id?.slice(0, 10)}</code></td><td><code>{item.account_ref}</code></td><td><span className={`state ${stateClass(item.status)}`}>{item.status}</span></td><td>{item.activation_requested_by ?? "—"} / {item.activated_by ?? "—"}</td><td>{can("broker:manage") ? <>{item.status === "disabled" && <button className="inline-action" onClick={() => brokerDestinationAction(item, "request-activation")}>申请激活</button>}{item.status === "pending_activation" && <button className="inline-action" onClick={() => brokerDestinationAction(item, "approve-activation")}>第二人批准</button>}{["pending_activation", "armed", "locked_mismatch"].includes(item.status) && <button className="inline-action" onClick={() => reconcileBrokerDestination(item)}>立即对账</button>}{item.status !== "disabled" && <button className="inline-action" onClick={() => brokerDestinationAction(item, "disarm")}>立即锁定</button>}</> : "—"}</td></tr>)}</tbody></table>{!broker.destinations.length && <div className="empty compact">券商边界默认锁定。只有管理员可创建沙箱映射，当前版本不接受实盘环境。</div>}</div>
      <div className="table-wrap"><table className="operations-table"><thead><tr><th>Outbox</th><th>批次</th><th>订单</th><th>创建 / 批准</th><th>尝试</th><th>状态</th></tr></thead><tbody>{broker.outbox.slice(0, 50).map((item) => <tr key={item.id}><td><code>{item.id.slice(0, 10)}</code></td><td><code>{item.batch_id.slice(0, 10)}</code></td><td>{item.payload.side} {item.payload.instrument} · {item.payload.quantity}</td><td>{item.created_by} / {item.approved_by ?? "—"}</td><td>{item.attempts}</td><td><span className={`state ${stateClass(item.status)}`}>{item.status}</span></td></tr>)}</tbody></table></div>
      <div className="table-wrap"><table className="operations-table"><thead><tr><th>对账时间</th><th>目的地</th><th>券商快照</th><th>状态</th><th>差异</th></tr></thead><tbody>{broker.reconciliations.slice(0, 30).map((item) => <tr key={item.id}><td>{timeText(item.created_at)}</td><td><code>{item.destination_id.slice(0, 10)}</code></td><td>{timeText(item.broker_as_of)}</td><td><span className={`state ${item.status === "matched" ? "ready" : "failed"}`}>{item.status}</span></td><td>{item.differences.length ? item.differences.map((difference) => `${difference.type}${difference.instrument ? `:${difference.instrument}` : ""}`).join(" · ") : "资金、持仓、订单一致"}</td></tr>)}</tbody></table>{!broker.reconciliations.length && <div className="empty compact">尚无券商沙箱对账记录。</div>}</div>
    </section>}

    {currentUser.role === "admin" && broker.destinations.length > 0 && <section className="operations-forms"><form className="automation-card" onSubmit={createBrokerSchedule}><div className="card-heading"><div><span>持续对账</span><strong>每日盘后核对券商状态</strong></div><span className="status-chip verified">FAIL CLOSED</span></div><label>计划名称<input value={brokerScheduleName} onChange={(event) => setBrokerScheduleName(event.target.value)} /></label><label>目的地<select value={brokerDestinationId} onChange={(event) => setBrokerDestinationId(event.target.value)}>{broker.destinations.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.status}</option>)}</select></label><div className="form-row"><label>盘后时间<input type="time" min="15:10" value={brokerScheduleTime} onChange={(event) => setBrokerScheduleTime(event.target.value)} /></label><label>错过宽限（秒）<input type="number" min="60" max="86400" step="60" value={brokerScheduleMisfireGrace} onChange={(event) => setBrokerScheduleMisfireGrace(Number(event.target.value))} /></label></div><div className="execution-note"><b>自动熔断</b><span>核对资金、权益、持仓、委托与成交标识</span><span>未知订单或快照超时立即锁定目的地</span><span>差异进入严重告警，不自动重新激活</span></div><button className="primary" disabled={!brokerDestinationId || brokerScheduleName.length < 3}>启用每日对账</button></form></section>}
    {currentUser.role !== "admin" ? <div className="workspace-card empty-state-card"><h2>券商沙箱仅管理员可见</h2><p>当前版本明确禁用实盘交易。</p></div> : null}
    </>}

    {view === "automation" && <>
    {can("automation:manage") && <section className="operations-forms panel-without-top-margin">
      <form className="automation-card" onSubmit={createSync}>
        <div className="card-heading"><div><span>数据自动化</span><strong>每日增量同步</strong></div><span className="status-chip">可恢复</span></div>
        <label>计划名称<input value={syncName} onChange={(event) => setSyncName(event.target.value)} /></label>
        <div className="form-row"><label>运行时间<input type="time" value={syncTime} onChange={(event) => setSyncTime(event.target.value)} /></label><label>任务模式<select value={syncKind} onChange={(event) => { const kind = event.target.value as "incremental_sync" | "data_pipeline" | "ashare_5m_sync"; setSyncKind(kind); if (kind === "ashare_5m_sync") setSyncLookbackDays((current) => Math.min(current, 30)); }}><option value="incremental_sync">A 股核心增量</option><option value="data_pipeline">全部数据依赖链</option><option value="ashare_5m_sync">全 A 股 5 分钟增量</option></select></label></div>
        <label>数据范围<select value={profile} onChange={(event) => setProfile(event.target.value)}><option value="core">Core</option><option value="research">Research</option><option value="full">Full</option></select></label>
        <div className="form-row"><label>修订回看天数<input type="number" min="1" max={syncKind === "ashare_5m_sync" ? 30 : 90} step="1" value={syncLookbackDays} onChange={(event) => setSyncLookbackDays(Number(event.target.value))} /></label><label>快照起始日<input type="date" value={syncSnapshotStart} onChange={(event) => setSyncSnapshotStart(event.target.value)} /></label></div>
        <div className="form-row"><label>错过宽限（秒）<input type="number" min="60" max="86400" step="60" value={syncMisfireGrace} onChange={(event) => setSyncMisfireGrace(Number(event.target.value))} /></label><label className="policy-toggle"><input type="checkbox" checked={syncBuildQlib} onChange={(event) => setSyncBuildQlib(event.target.checked)} /><span>同步后构建 Qlib 快照</span></label></div>
        <div className="execution-note"><b>自动流程</b><span>{syncKind === "data_pipeline" ? "核心日线与九类扩展数据逐项执行，前项成功才启动后项" : syncKind === "ashare_5m_sync" ? "盘后按股票生命周期补齐最近窗口的全 A 股 5 分钟线" : "按配置的回看窗口吸收 A 股核心修订数据"}</span><span>下载、校验、快照、Qlib 与基线保持独立任务，可分别重试</span><span>错过执行窗口或任一依赖失败时停止后续任务并产生告警</span></div>
        <button className="primary" disabled={syncName.length < 3}>启用数据计划</button>
      </form>
      <form className="automation-card" onSubmit={createPortfolioSchedule}>
        <div className="card-heading"><div><span>组合自动化</span><strong>盘后再平衡</strong></div><span className="status-chip verified">交易日历</span></div>
        <label>计划名称<input value={portfolioName} onChange={(event) => setPortfolioName(event.target.value)} /></label>
        <label>模拟组合<select value={portfolioId} onChange={(event) => setPortfolioId(event.target.value)}>{portfolios.length ? portfolios.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.status}</option>) : <option value="">尚无模拟组合</option>}</select></label>
        <div className="form-row"><label>盘后运行时间<input type="time" min="15:10" value={portfolioTime} onChange={(event) => setPortfolioTime(event.target.value)} /></label><label>模拟滑点（bp）<input type="number" min="0" max="200" step="1" value={portfolioSlippage * 10000} onChange={(event) => setPortfolioSlippage(Number(event.target.value) / 10000)} /></label></div>
        <label>错过宽限（秒）<input type="number" min="60" max="86400" step="60" value={portfolioMisfireGrace} onChange={(event) => setPortfolioMisfireGrace(Number(event.target.value))} /></label>
        <div className="execution-note"><b>执行边界</b><span>只在所选 Qlib 快照的交易日生成信号</span><span>同一计划时点、组合和信号日均有幂等约束</span><span>次日开盘模拟成交由 Qlib Worker 执行</span></div>
        <button className="primary" disabled={!portfolioId || portfolioName.length < 3}>启用组合计划</button>
      </form>
    </section>}

    <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">DURABLE SCHEDULES</p><h2>自动计划</h2></div><span>{schedules.length} 条</span></div><div className="table-wrap"><table className="operations-table"><thead><tr><th>计划</th><th>类型</th><th>时间</th><th>下一次运行</th><th>期望状态</th><th>实际状态 / 原因</th><th>操作</th></tr></thead><tbody>{schedules.map((item) => <tr key={item.id}><td><strong>{item.name}</strong><small>{item.timezone}{item.payload.managed_by === "allocation_schedule_group" ? " · 组合统一托管" : ""}</small></td><td>{scheduleKindText[item.kind] ?? item.kind}</td><td>{item.run_time}</td><td>{timeText(item.next_run_at)}</td><td><span className={`state ${stateClass(item.desired_status)}`}>{item.desired_status}</span></td><td><span className={`state ${stateClass(item.status)}`}>{item.status}</span><small>{item.suspension_reason ?? "无安全暂停"}</small></td><td>{can("automation:manage") ? item.payload.managed_by === "allocation_schedule_group" ? <span className="muted">在多策略组合中操作</span> : item.status === "retired" ? <span className="muted">已退役</span> : <button className="inline-action" onClick={() => setScheduleStatus(item)}>{item.desired_status === "active" ? "暂停" : "恢复"}</button> : "—"}</td></tr>)}</tbody></table>{!schedules.length && <div className="empty">尚无自动计划。创建后，调度时点与每次运行都会持久化。</div>}</div></section>

    <section className="operations-lower">
      <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">ALERT INBOX</p><h2>告警收件箱</h2></div><span>{openAlerts.length} 条待确认</span></div><div className="alert-list">{alerts.slice(0, 30).map((item) => <article key={item.id}><span className={`alert-severity ${item.severity}`} /><div><strong>{item.title}</strong><p>{item.message}</p><small>{timeText(item.created_at)} · 投递 {item.delivery_status}</small></div><span className={`state ${stateClass(item.status)}`}>{item.status}</span><div>{can("alerts:manage") && item.status === "open" && <button className="inline-action" onClick={() => actOnAlert(item, "acknowledge")}>确认</button>}{can("alerts:manage") && item.status !== "resolved" && <button className="inline-action" onClick={() => actOnAlert(item, "resolve")}>关闭</button>}</div></article>)}{!alerts.length && <div className="empty compact">没有告警。任务失败、调度错过和组合风险事件会自动进入这里。</div>}</div></section>
      <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">SCHEDULE RUNS</p><h2>最近调度</h2></div><span>{runs.length} 次</span></div><div className="run-list">{runs.slice(0, 30).map((item) => <article key={item.id}><span className={`job-state ${item.status === "failed" || item.status === "missed" ? "failed" : item.status === "running" ? "running" : "succeeded"}`} /><div><strong>{item.schedule_name}</strong><small>{timeText(item.scheduled_for)} · 尝试 {item.attempts}</small></div><span className={`state ${stateClass(item.status)}`}>{item.status}</span>{item.message && <p>{item.message}</p>}</article>)}{!runs.length && <div className="empty compact">尚无调度运行记录。</div>}</div></section>
    </section>

    <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">WORKER JOBS</p><h2>后台任务</h2></div><span>{jobs.length} 条</span></div><div className="table-wrap"><table className="operations-table"><thead><tr><th>任务</th><th>类型</th><th>创建时间</th><th>开始</th><th>结束</th><th>状态</th></tr></thead><tbody>{jobs.map((item) => <tr key={item.id}><td><code>{item.id.slice(0, 10)}</code></td><td>{item.kind}</td><td>{timeText(item.created_at)}</td><td>{timeText(item.started_at)}</td><td>{timeText(item.finished_at)}</td><td><span className={`state ${stateClass(item.status)}`}>{item.status}</span></td></tr>)}</tbody></table></div></section>
    </>}

    {view === "access" && <>
    <section className="access-grid panel-without-top-margin">
      <form className="automation-card" onSubmit={changePassword}>
        <div className="card-heading"><div><span>账户安全</span><strong>修改我的密码</strong></div><span className="status-chip verified">Argon2</span></div>
        <label>当前密码<input type="password" autoComplete="current-password" value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} /></label>
        <label>新密码<input type="password" autoComplete="new-password" value={replacementPassword} onChange={(event) => setReplacementPassword(event.target.value)} /></label>
        <div className="execution-note"><b>会话保护</b><span>至少 12 个字符，包含字母、数字和符号</span><span>修改后撤销此账户的其他登录会话</span></div>
        <button className="primary" disabled={!currentPassword || replacementPassword.length < 12}>更新密码</button>
      </form>
      {currentUser.role === "admin" && <form className="automation-card" onSubmit={createUser}>
        <div className="card-heading"><div><span>权限管理</span><strong>创建本地用户</strong></div><span className="status-chip">RBAC</span></div>
        <div className="form-row"><label>用户名<input autoComplete="off" value={newUsername} onChange={(event) => setNewUsername(event.target.value)} /></label><label>显示名称<input value={newDisplayName} onChange={(event) => setNewDisplayName(event.target.value)} /></label></div>
        <label>角色<select value={newRole} onChange={(event) => setNewRole(event.target.value)}><option value="viewer">只读查看</option><option value="researcher">研究员</option><option value="operator">运维操作员</option><option value="admin">管理员</option></select></label>
        <label>初始密码<input type="password" autoComplete="new-password" value={newUserPassword} onChange={(event) => setNewUserPassword(event.target.value)} /></label>
        <button className="primary" disabled={newUsername.length < 3 || !newDisplayName || newUserPassword.length < 12}>创建用户</button>
      </form>}
    </section>

    {currentUser.role === "admin" && <>
      <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">ACCESS CONTROL</p><h2>用户与角色</h2></div><span>{managedUsers.length} 个账户</span></div><div className="table-wrap"><table className="operations-table user-table"><thead><tr><th>用户</th><th>角色</th><th>最近登录</th><th>失败次数</th><th>状态</th><th>操作</th></tr></thead><tbody>{managedUsers.map((user) => <tr key={user.id}><td><strong>{user.display_name}</strong><small>{user.username}</small></td><td>{user.role}</td><td>{timeText(user.last_login_at)}</td><td>{user.failed_login_attempts}</td><td><span className={`state ${user.active ? "ready" : "failed"}`}>{user.active ? "active" : "disabled"}</span></td><td><button className="inline-action" onClick={() => setUserActive(user)}>{user.active ? "停用" : "启用"}</button></td></tr>)}</tbody></table></div></section>
      <section className="data-panel"><div className="panel-heading"><div><p className="eyebrow">SECURITY AUDIT</p><h2>操作审计</h2></div><span>最近 {auditEvents.length} 条</span></div><div className="audit-list">{auditEvents.map((item) => <article key={item.id}><div><strong>{item.username}</strong><small>{timeText(item.created_at)}</small></div><code>{item.method} {item.path}</code><span>{item.action}</span><span className={`state ${item.status_code < 400 ? "ready" : "failed"}`}>{item.status_code}</span></article>)}{!auditEvents.length && <div className="empty compact">尚无安全审计记录。</div>}</div></section>
    </>}
    </>}
  </>;
}
