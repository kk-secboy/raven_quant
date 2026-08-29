"use client";

import { CSSProperties, FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";
import { BacktestPanel } from "./backtest-panel";
import { DataTask, DataTaskCenter } from "./data-task-center";
import { phaseLabel, targetText } from "./data-progress";
import { AuthPanel, AuthState } from "./auth-panel";
import { AutopilotPanel } from "./autopilot-panel";
import { FactorLibraryPanel } from "./factor-library-panel";
import { JobRunCenter } from "./job-run-center";
import { MarketOverviewPanel } from "./market-overview-panel";
import { PortfolioPanel } from "./portfolio-panel";
import { QlibPanel } from "./qlib-panel";
import { RDAgentPanel } from "./rdagent-panel";
import { SettingsPanel } from "./settings-panel";
import { StrategyAllocationPanel } from "./strategy-allocation-panel";
import { usePolling } from "./use-polling";

type Overview = {
  mode: string;
  credentials_configured: boolean;
  data_root: string;
  rows: number;
  planned_units: number;
  succeeded_units: number;
  coverage: number;
  snapshots: number;
  qlib_datasets: number;
  active_jobs: number;
  active_bootstrap_jobs: number;
  active_finalize_jobs: number;
  running_work_units: number;
  legacy_download_coverage: number;
  readiness_percent: number;
  ready_tasks: number;
  actionable_tasks: number;
  partial_tasks: number;
  failed_tasks: number;
  running_tasks: number;
  waiting_tasks: number;
  retry_waiting_tasks: number;
  blocked_tasks: number;
  terminal_failed_tasks: number;
  startable_tasks: number;
  components: { name: string; state: string }[];
};

type Dataset = {
  name: string;
  profile: "core" | "research" | "full";
  planned: number;
  succeeded: number;
  failed: number;
  running: number;
  rows: number;
  coverage: number;
  state: "ready" | "partial" | "empty";
};

type RetentionEntry = {
  name: string; created_at: string; bytes: number; locations: string[];
  state: "protected" | "keep_latest" | "keep_young" | "eligible" | "inventory_pending";
  reasons: string[]; inventory_complete?: boolean;
};
type RetentionPlan = {
  total_bytes: number; eligible_bytes: number; keep_latest: number; min_age_days: number;
  entries: RetentionEntry[]; inventory_generated_at?: string; cache_state?: "fresh" | "stale" | "building";
  refreshing?: boolean;
};

const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8765";
const navGroups = [
  { label: "日常运行", items: [{ index: 0, label: "策略运行总览" }, { index: 11, label: "行情总览" }, { index: 8, label: "统一模拟盘" }] },
  { label: "研究治理", items: [{ index: 1, label: "数据快照" }, { index: 3, label: "RD-Agent 研究中心" }, { index: 5, label: "因子库与准入" }, { index: 2, label: "模型竞赛与试验" }, { index: 6, label: "Qlib 回测与审批" }, { index: 12, label: "核心 / 卫星分配" }] },
  { label: "审计与系统", items: [{ index: 9, label: "任务、告警与历史" }, { index: 10, label: "系统设置" }] },
];
const simpleNavGroups = [
  { label: "日常使用", items: [{ index: 0, label: "今日建议" }, { index: 11, label: "行情总览" }, { index: 8, label: "模拟账本" }] },
  { label: "需要处理", items: [{ index: 9, label: "异常" }] },
];
const headings: Record<number, [string, string]> = {
  0: ["QUANTLAB / AUTOPILOT", "自动驾驶"],
  1: ["TUSHARE SNAPSHOTS / QLIB DATASET", "数据快照"],
  2: ["MODEL TOURNAMENT / QLIB", "模型竞赛与试验"],
  3: ["AUTONOMOUS RESEARCH / GOVERNED", "RD-Agent 研究中心"],
  5: ["FACTOR GOVERNANCE / REGISTRY", "因子库与准入"],
  6: ["QLIB BACKTEST / RISK APPROVAL", "Qlib 回测与审批"],
  8: ["APPROVED SOURCES / DURABLE LEDGER", "统一模拟盘"],
  9: ["AUTOMATION / ALERTS / AUDIT", "任务、告警与历史"],
  10: ["SERVER CONFIGURATION / ENCRYPTED", "系统设置"],
  11: ["MARKET INTELLIGENCE / RESEARCH SNAPSHOT", "行情总览"],
  12: ["CORE SATELLITE / RISK BUDGET", "核心 / 卫星分配"],
};

function formatNumber(value: number) {
  return new Intl.NumberFormat("zh-CN", { notation: value > 999999 ? "compact" : "standard" }).format(value);
}

function formatBytes(value: number) {
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
}

function statusLabel(status: string) {
  return ({ queued: "排队", running: "运行中", succeeded: "成功", failed: "失败" } as Record<string, string>)[status] ?? status;
}

function retentionStateLabel(state: RetentionEntry["state"]) {
  if (state === "eligible") return "可清理";
  if (state === "inventory_pending") return "统计中";
  return "受保护";
}

export default function Home() {
  const [auth, setAuth] = useState<AuthState | null>(null);
  const [activeNav, setActiveNav] = useState(0);
  const [advancedMode, setAdvancedMode] = useState(false);
  const [dataView, setDataView] = useState("overview");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataTasks, setDataTasks] = useState<DataTask[]>([]);
  const [retention, setRetention] = useState<RetentionPlan | null>(null);
  const [retentionSelection, setRetentionSelection] = useState<Record<string, boolean>>({});
  const [retentionConfirmation, setRetentionConfirmation] = useState("");
  const [profile, setProfile] = useState("full");
  const [downloadStart, setDownloadStart] = useState("2016-01-01");
  const [snapshotStart, setSnapshotStart] = useState("2008-01-01");
  const [message, setMessage] = useState("");

  function navigateTo(index: number) {
    setMessage("");
    setActiveNav(index);
  }
  const [loading, setLoading] = useState(true);

  const checkAuth = useCallback(async () => {
    try {
      const response = await apiFetch(`${API}/api/auth/state`, { cache: "no-store" });
      if (!response.ok) throw new Error("auth unavailable");
      setAuth(await response.json());
    } catch {
      setAuth({ status: "login_required", user: null });
      setMessage("无法连接认证服务。");
    }
  }, []);

  const refresh = useCallback(async (forceRefresh = false) => {
    const [overviewResult, datasetsResult, dataTasksResult] = await Promise.allSettled([
      apiFetch(`${API}/api/overview`, { cache: "no-store", forceRefresh }).then(async (response) => {
        if (!response.ok) throw new Error("overview unavailable");
        return response.json() as Promise<Overview>;
      }),
      apiFetch(`${API}/api/datasets`, { cache: "no-store", forceRefresh }).then(async (response) => {
        if (!response.ok) throw new Error("datasets unavailable");
        return response.json() as Promise<Dataset[]>;
      }),
      apiFetch(`${API}/api/data-tasks`, { cache: "no-store", forceRefresh }).then(async (response) => {
        if (!response.ok) throw new Error("data tasks unavailable");
        return response.json() as Promise<DataTask[]>;
      }),
    ]);
    let loaded = 0;
    if (overviewResult.status === "fulfilled") { setOverview(overviewResult.value); loaded += 1; }
    if (datasetsResult.status === "fulfilled") { setDatasets(datasetsResult.value); loaded += 1; }
    if (dataTasksResult.status === "fulfilled") { setDataTasks(dataTasksResult.value); loaded += 1; }
    setMessage(loaded === 0 ? "暂时无法连接数据服务。" : loaded < 3 ? "部分数据正在恢复，已先展示可用内容。" : "");
    setLoading(false);
  }, []);

  const loadRetention = useCallback(async (forceRefresh = false) => {
    try {
      const response = await apiFetch(`${API}/api/data-retention`, {
        cache: "no-store",
        forceRefresh,
        timeoutMs: 120_000,
      });
      if (!response.ok) throw new Error("retention unavailable");
      const next = await response.json() as RetentionPlan;
      setRetention(next);
      const selectable = new Set(next.entries.filter((item) => item.state === "eligible").map((item) => item.name));
      setRetentionSelection((current) => Object.fromEntries(
        Object.entries(current).filter(([name, selected]) => selected && selectable.has(name)),
      ));
    } catch {
      setMessage("存储容量统计暂时不可用；其他数据不受影响。");
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(checkAuth, 0);
    return () => window.clearTimeout(initial);
  }, [checkAuth]);

  const dataPollingEnabled = Boolean(
    auth && ["authenticated", "disabled"].includes(auth.status) && [0, 1].includes(activeNav),
  );
  usePolling(refresh, 5000, dataPollingEnabled);

  const storageViewEnabled = dataPollingEnabled && activeNav === 1 && dataView === "storage";
  usePolling(loadRetention, 60 * 60 * 1000, storageViewEnabled);

  async function refreshVisible() {
    if (activeNav !== 1) return;
    await refresh(true);
    if (dataView === "storage") await loadRetention(true);
  }

  async function logout() {
    await apiFetch(`${API}/api/auth/logout`, { method: "POST" });
    await checkAuth();
  }

  const visibleDatasets = useMemo(
    () => datasets.filter((item) => profile === "full" || item.profile === profile || (profile === "research" && item.profile === "core")),
    [datasets, profile],
  );
  const activeDataTasks = useMemo(
    () => {
      const seenJobs = new Set<string>();
      return dataTasks.filter((task) => {
        if (!["queued", "running"].includes(task.status)) return false;
        if (!task.job_id) return true;
        if (seenJobs.has(task.job_id)) return false;
        seenJobs.add(task.job_id);
        return true;
      });
    },
    [dataTasks],
  );
  const activeHeading = !advancedMode && activeNav === 0
    ? ["THREE HORIZONS / ONE ACCOUNT", "今日选股与账户操作"]
    : headings[activeNav];

  function liveRunningUnits(task: DataTask) {
    const scoped = Number(
      task.progress?.checkpoint?.running ?? task.unit_stats.running,
    );
    return activeDataTasks.length === 1 && task.status === "running"
      ? Math.max(scoped, overview?.running_work_units ?? 0)
      : scoped;
  }

  function liveExecutionPhase(task: DataTask) {
    return task.status === "running" && liveRunningUnits(task) > 0
      ? "downloading"
      : task.execution_phase;
  }

  async function startBootstrap(event: FormEvent) {
    event.preventDefault();
    setMessage("正在创建初始化任务…");
    const response = await apiFetch(`${API}/api/jobs/bootstrap`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile,
        start: downloadStart,
        snapshot_start: snapshotStart,
        end: "latest",
        build_qlib: false,
      }),
    });
    const body = await response.json();
    if (!response.ok) {
      setMessage(body.detail ?? "任务创建失败");
      return;
    }
    setMessage(`任务 ${body.id.slice(0, 8)} 已进入队列`);
    await refresh();
  }

  async function finalizeData() {
    setMessage("正在创建数据质量校验任务…");
    const response = await apiFetch(`${API}/api/jobs/finalize-data`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile, start: snapshotStart, end: "latest" }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "数据收口任务创建失败"); return; }
    setMessage(`数据收口流水线 ${body.payload.pipeline_id.slice(0, 8)} 已启动。`);
    await refresh();
  }

  async function applyRetention() {
    const names = Object.entries(retentionSelection).filter(([, selected]) => selected).map(([name]) => name);
    const response = await apiFetch(`${API}/api/data-retention/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        names,
        confirmation: retentionConfirmation,
        keep_latest: retention?.keep_latest ?? 7,
        min_age_days: retention?.min_age_days ?? 14,
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "数据保留策略执行失败"); return; }
    setRetentionSelection({});
    setRetentionConfirmation("");
    setMessage(`已清理 ${body.deleted.length} 个未引用数据集，释放 ${formatBytes(body.reclaimed_bytes)}。`);
    await refresh();
    await loadRetention();
  }

  if (!auth) return <main className="auth-shell"><div className="auth-loading">正在检查安全会话…</div></main>;
  if (!["authenticated", "disabled"].includes(auth.status)) {
    return <AuthPanel api={API} state={auth} onAuthenticated={checkAuth} />;
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">Q</span><span>Quant<span>Lab</span></span></div>
        <nav aria-label="主导航">
          {(advancedMode ? navGroups : simpleNavGroups).map((group) => (
            <div className="nav-group" key={group.label}>
              <span className="nav-group-label">{group.label}</span>
              {group.items.map((item) => (
                <button className={item.index === activeNav ? "nav-item active" : "nav-item"} key={item.label} disabled={item.index === 10 && auth.user?.role !== "admin"} onClick={() => navigateTo(item.index)}>
                  <span className="nav-dot" />{item.label}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <button className="sidebar-mode" type="button" onClick={() => {
          const next = !advancedMode;
          setAdvancedMode(next);
          if (!next) navigateTo(0);
        }}>
          {advancedMode ? "返回日常模式" : "打开高级管理"}
        </button>
        <div className="sidebar-foot">
          <span className={overview?.credentials_configured ? "pulse ok" : "pulse"} />
          <div><strong>受控研究模式</strong><small>{overview?.credentials_configured ? "数据源已配置" : "等待 Tushare 凭据"}</small></div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div><p className="eyebrow">{activeHeading?.[0]}</p><h1>{activeHeading?.[1]}</h1></div>
          <div className="top-actions"><span className="system-live"><i />受控研究工作台</span><span className="account-chip"><b>{auth.user?.display_name}</b><small>{auth.user?.role}</small></span>{activeNav === 1 ? <button onClick={() => void refreshVisible()}>刷新概况</button> : null}{auth.status === "authenticated" && <button onClick={logout}>退出</button>}</div>
        </header>

        {message && <div className="notice">{message}</div>}

        {activeNav === 0 ? <AutopilotPanel api={API} advancedMode={advancedMode} onNavigate={navigateTo} onOpenAdvanced={(index) => { setAdvancedMode(true); navigateTo(index); }} /> : activeNav === 1 ? (
          <div className="data-center-page">
            <div className="page-tabs" role="tablist" aria-label="数据中心页面">
              {[['overview', '运行概况'], ['catalog', '数据目录'], ['create', '新建任务'], ['runs', '运行记录'], ['storage', '存储与版本']].map(([value, label]) => <button role="tab" aria-selected={dataView === value} className={dataView === value ? "active" : ""} onClick={() => setDataView(value)} key={value}>{label}{value === "runs" && overview?.active_jobs ? <i>{overview.active_jobs}</i> : null}</button>)}
            </div>

            {dataView === "overview" ? <>
              <section className="readiness-hero">
                <div className="readiness-copy"><span className="status-chip">{loading ? "正在连接" : overview?.active_jobs ? "下载器正在工作" : (overview?.partial_tasks || overview?.waiting_tasks) ? "仍有数据能力待准备" : "数据能力已就绪"}</span><h2>{overview?.ready_tasks ?? 0} / {overview?.actionable_tasks ?? 0} 项数据能力已完全可用</h2><p>圆环是全部目录覆盖率的平均值，部分完成的目录也会计入；分数是完全可用的目录数量。它们都不代表单个下载任务的执行进度，当前阶段与 checkpoint 在下方单独展示。</p></div>
                <div className="readiness-ring" style={{ "--progress": `${overview?.readiness_percent ?? 0}%` } as CSSProperties}><strong>{overview?.readiness_percent ?? 0}%</strong><span>平均目录覆盖度</span></div>
              </section>
              <section className="status-strip">
                <article><span>正在运行</span><strong>{overview?.running_tasks ?? 0}</strong></article>
                <article><span>冷却 / 待恢复</span><strong>{overview?.retry_waiting_tasks ?? 0}</strong></article>
                <article><span>等待前置数据</span><strong>{overview?.blocked_tasks ?? 0}</strong></article>
                <article><span>终止失败</span><strong className={overview?.terminal_failed_tasks ? "danger" : ""}>{overview?.terminal_failed_tasks ?? 0}</strong></article>
                <article><span>现在可启动</span><strong>{overview?.startable_tasks ?? 0}</strong><small>依赖已满足</small></article>
              </section>
              <section className="overview-grid data-overview-grid">
                <article className="workspace-card live-downloads">
                  <div className="section-heading"><div><h2>当前下载执行</h2><p>展示真实阶段与 checkpoint 状态；工作单元会随自适应拆分动态变化。</p></div><button onClick={() => setDataView("runs")}>运行记录</button></div>
                  {activeDataTasks.slice(0, 3).map((task) => <div className="live-download" key={task.job_id ?? task.task_key}>
                    <div className="live-download-head"><div><span className={`task-status ${task.status}`}>{phaseLabel(liveExecutionPhase(task))}</span><strong>{task.title}</strong><small>{targetText(task.job?.payload, task.progress)}</small></div><span className="live-updated">{task.progress?.updated_at ? `更新于 ${new Date(task.progress.updated_at).toLocaleTimeString("zh-CN")}` : statusLabel(task.status)}</span></div>
                    <p>{task.config.request_strategy}</p>
                    <div className="checkpoint-metrics">
                      <div><span>成功 checkpoint</span><strong>{formatNumber(Number(task.progress?.checkpoint?.succeeded ?? task.unit_stats.succeeded))}</strong></div>
                      <div><span>正在请求</span><strong>{formatNumber(liveRunningUnits(task))}</strong></div>
                      <div><span>等待重试</span><strong>{formatNumber(Number(task.progress?.checkpoint?.retry_waiting ?? task.unit_stats.retry_waiting))}</strong></div>
                      <div><span>终止失败</span><strong className={Number(task.progress?.checkpoint?.terminal_failed ?? task.unit_stats.terminal_failed) ? "danger" : ""}>{formatNumber(Number(task.progress?.checkpoint?.terminal_failed ?? task.unit_stats.terminal_failed))}</strong></div>
                      <div><span>替代审计</span><strong>{formatNumber(Number(task.progress?.checkpoint?.superseded ?? task.unit_stats.superseded))}</strong></div>
                    </div>
                  </div>)}
                  {!activeDataTasks.length ? <div className="empty compact">当前没有运行中的下载；已完成的数据仍会从 checkpoint 直接复用。</div> : null}
                </article>
                <article className="workspace-card next-actions">
                  <div className="section-heading"><div><h2>需要处理</h2><p>区分可启动、等待依赖、可恢复失败和终止失败。</p></div><button onClick={() => setDataView("catalog")}>查看全部</button></div>
                  {dataTasks.filter((task) => task.status !== "succeeded" && !["queued", "running"].includes(task.status) && !["permission_probe", "external_source_required"].includes(task.implementation_status)).slice(0, 6).map((task) => <div className="action-row" key={task.task_key}><span className={`task-status ${task.status}`}>{phaseLabel(task.execution_phase)}</span><div><strong>{task.title}</strong><small>{task.dependencies_satisfied ? task.config.request_strategy : "等待前置数据完成"}</small></div></div>)}
                </article>
              </section>
            </> : null}

            {dataView === "catalog" ? <DataTaskCenter tasks={dataTasks} api={API} mode="catalog" onCreated={refresh} onMessage={setMessage} /> : null}

            {dataView === "create" ? <div className="create-layout">
              <form className="bootstrap-card compact-bootstrap" onSubmit={startBootstrap}>
                <div className="card-heading"><div><span>首次使用</span><strong>初始化 A 股日线基础库</strong></div><span className="status-chip">可恢复</span></div>
                <p>下载行情、复权、财务和交易约束。下载完成后，质量校验、快照和 Qlib 构建独立执行，失败可单独重试。</p>
                <label>数据范围<select value={profile} onChange={(e) => setProfile(e.target.value)}><option value="core">基础 · 价格与交易约束</option><option value="research">研究 · 资金与行业</option><option value="full">完整 · 财务与事件</option></select></label>
                <div className="form-row"><label>主数据开始日期<input type="date" min="2016-01-01" max="2016-01-01" value={downloadStart} onChange={(e) => setDownloadStart(e.target.value)} /></label><label>研究快照开始日期<input type="date" max={downloadStart} value={snapshotStart} onChange={(e) => setSnapshotStart(e.target.value)} /></label></div>
                <button className="primary" disabled={!overview?.credentials_configured || !!overview?.active_bootstrap_jobs}>创建基础下载任务</button>
                <button type="button" onClick={finalizeData} disabled={profile !== "full" || !!overview?.active_finalize_jobs || !overview?.planned_units}>完整数据下载完成，开始质量校验</button>
              </form>
              <DataTaskCenter tasks={dataTasks} api={API} mode="create" onCreated={refresh} onMessage={setMessage} />
            </div> : null}

            {dataView === "runs" ? <JobRunCenter api={API} canControl={auth.user?.role === "admin" || auth.user?.role === "operator"} onChanged={refresh} onMessage={setMessage} /> : null}

            {dataView === "storage" ? <div className="storage-stack">
              <section className="data-panel storage-panel">
                <div className="panel-heading">
                  <div>
                    <h2>不可变快照与 Qlib 数据</h2>
                    <p>
                      被研究、回测和推荐组合引用的数据不会被清理。
                      {retention?.inventory_generated_at
                        ? ` 容量摘要截至 ${new Date(retention.inventory_generated_at).toLocaleString("zh-CN")}；删除前会实时复核。`
                        : " 正在读取容量摘要。"}
                    </p>
                  </div>
                  <span>{retention?.refreshing ? "后台更新中 · " : ""}{formatBytes(retention?.eligible_bytes ?? 0)} 可清理</span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>选择</th><th>数据集</th><th>创建时间</th><th>占用</th><th>状态 / 原因</th></tr></thead>
                    <tbody>{retention?.entries.map((item) => <tr key={item.name}>
                      <td><input aria-label={`选择清理 ${item.name}`} type="checkbox" disabled={item.state !== "eligible" || auth.user?.role !== "admin"} checked={!!retentionSelection[item.name]} onChange={(event) => setRetentionSelection({ ...retentionSelection, [item.name]: event.target.checked })} /></td>
                      <td><code>{item.name}</code></td>
                      <td>{new Date(item.created_at).toLocaleString("zh-CN")}</td>
                      <td>{item.inventory_complete === false ? "统计中" : formatBytes(item.bytes)}</td>
                      <td><span className={`state ${item.state === "eligible" ? "failed" : item.state === "inventory_pending" ? "partial" : "ready"}`}>{retentionStateLabel(item.state)}</span><small>{item.reasons.join("；") || "未被引用且已过保留期"}</small></td>
                    </tr>)}</tbody>
                  </table>
                  {retention === null ? <div className="empty compact">正在读取上一版容量摘要…</div> : !retention.entries.length ? <div className="empty compact">当前没有快照或 Qlib 数据集。</div> : null}
                </div>
                {auth.user?.role === "admin" ? <div className="retention-action"><label>输入 DELETE_UNREFERENCED_DATASETS 确认<input value={retentionConfirmation} onChange={(event) => setRetentionConfirmation(event.target.value)} /></label><button className="primary" onClick={applyRetention} disabled={retentionConfirmation !== "DELETE_UNREFERENCED_DATASETS" || !Object.values(retentionSelection).some(Boolean)}>清理所选数据</button></div> : null}
              </section>
              <section className="data-panel"><div className="panel-heading"><div><h2>基础数据集</h2><p>用于核对旧初始化流水线的工作单元。</p></div><div className="segmented">{["core", "research", "full"].map((value) => <button className={profile === value ? "selected" : ""} onClick={() => setProfile(value)} key={value}>{value}</button>)}</div></div><div className="table-wrap"><table className="dataset-catalog-table"><thead><tr><th>数据集</th><th>层级</th><th>状态</th><th>进度</th><th>数据行</th><th>失败</th></tr></thead><tbody>{visibleDatasets.map((item) => <tr key={item.name}><td><code>{item.name}</code></td><td><span className={`tier ${item.profile}`}>{item.profile}</span></td><td><span className={`state ${item.state}`}>{item.state === "ready" ? "已就绪" : item.state === "partial" ? "部分完成" : "未下载"}</span></td><td><div className="mini-progress"><i style={{ width: `${item.coverage}%` }} /></div><small>{item.coverage}%</small></td><td>{formatNumber(item.rows)}</td><td className={item.failed ? "danger" : "muted"}>{item.failed}</td></tr>)}</tbody></table></div></section>
            </div> : null}
          </div>
        ) : activeNav === 11 ? <MarketOverviewPanel api={API} onOpenData={() => { navigateTo(1); setDataView("overview"); }} /> : activeNav === 2 ? <QlibPanel api={API} /> : activeNav === 3 ? <RDAgentPanel api={API} /> : activeNav === 5 ? <FactorLibraryPanel api={API} /> : activeNav === 6 ? <BacktestPanel api={API} /> : activeNav === 8 ? <PortfolioPanel api={API} /> : activeNav === 12 ? <StrategyAllocationPanel api={API} /> : activeNav === 9 ? <JobRunCenter api={API} canControl={auth.user?.role === "admin" || auth.user?.role === "operator"} onChanged={refresh} onMessage={setMessage} /> : <SettingsPanel api={API} />}
      </section>
    </main>
  );
}
