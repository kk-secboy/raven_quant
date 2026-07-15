"use client";

import { CSSProperties, FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";
import { BacktestPanel } from "./backtest-panel";
import { DataTask, DataTaskCenter } from "./data-task-center";
import { AuthPanel, AuthState } from "./auth-panel";
import { FactorLibraryPanel } from "./factor-library-panel";
import { JobRunCenter } from "./job-run-center";
import { MarketOverviewPanel } from "./market-overview-panel";
import { PortfolioPanel } from "./portfolio-panel";
import { OperationsPanel } from "./operations-panel";
import { PairTradingPanel } from "./pair-trading-panel";
import { QlibPanel } from "./qlib-panel";
import { RDAgentPanel } from "./rdagent-panel";
import { ResearchCampaignPanel } from "./research-campaign-panel";
import { SettingsPanel } from "./settings-panel";

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
  legacy_download_coverage: number;
  readiness_percent: number;
  ready_tasks: number;
  actionable_tasks: number;
  partial_tasks: number;
  failed_tasks: number;
  running_tasks: number;
  waiting_tasks: number;
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

type Job = {
  id: string;
  kind: string;
  status: string;
  payload: { profile?: string; start?: string; end?: string; snapshot_name?: string };
  created_at: string;
  error?: string | null;
};

type RetentionEntry = {
  name: string; created_at: string; bytes: number; locations: string[];
  state: "protected" | "keep_latest" | "keep_young" | "eligible"; reasons: string[];
};
type RetentionPlan = {
  total_bytes: number; eligible_bytes: number; keep_latest: number; min_age_days: number;
  entries: RetentionEntry[];
};

const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8765";
const navGroups = [
  { label: "主页", items: [{ index: 0, label: "总览" }, { index: 11, label: "行情总览" }] },
  { label: "数据与研究", items: [{ index: 1, label: "数据中心" }, { index: 2, label: "Qlib 研究" }, { index: 3, label: "RD-Agent" }, { index: 4, label: "自动研究" }, { index: 5, label: "因子库" }] },
  { label: "策略与交易", items: [{ index: 6, label: "回测" }, { index: 7, label: "配对交易" }, { index: 8, label: "模拟组合" }] },
  { label: "系统", items: [{ index: 9, label: "任务与告警" }, { index: 10, label: "系统设置" }] },
];
const headings: Record<number, [string, string]> = {
  0: ["QUANTLAB / WORKSPACE", "总览"],
  1: ["DATA OPERATIONS / A-SHARE", "数据中心"],
  2: ["MODEL RESEARCH / QLIB", "Qlib 研究"],
  3: ["AUTONOMOUS RESEARCH / GOVERNED", "RD-Agent 研究"],
  4: ["RESEARCH AUTOPILOT / GOVERNED PIPELINE", "自动研究"],
  5: ["FACTOR GOVERNANCE / REGISTRY", "因子库"],
  6: ["STRATEGY RESEARCH / RISK GATES", "策略回测"],
  7: ["STATISTICAL ARBITRAGE / MINUTE EXECUTION", "配对交易"],
  8: ["PAPER EXECUTION / ATOMIC LEDGER", "模拟组合"],
  9: ["AUTOMATION / ALERTS / RECOVERY", "任务与告警"],
  10: ["SERVER CONFIGURATION / ENCRYPTED", "系统设置"],
  11: ["MARKET INTELLIGENCE / RESEARCH SNAPSHOT", "行情总览"],
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

export default function Home() {
  const [auth, setAuth] = useState<AuthState | null>(null);
  const [activeNav, setActiveNav] = useState(0);
  const [dataView, setDataView] = useState("overview");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [dataTasks, setDataTasks] = useState<DataTask[]>([]);
  const [retention, setRetention] = useState<RetentionPlan | null>(null);
  const [retentionSelection, setRetentionSelection] = useState<Record<string, boolean>>({});
  const [retentionConfirmation, setRetentionConfirmation] = useState("");
  const [profile, setProfile] = useState("core");
  const [start, setStart] = useState("2024-01-01");
  const [message, setMessage] = useState("");
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

  const refresh = useCallback(async () => {
    try {
      const [overviewResponse, datasetsResponse, jobsResponse, dataTasksResponse, retentionResponse] = await Promise.all([
        apiFetch(`${API}/api/overview`, { cache: "no-store" }),
        apiFetch(`${API}/api/datasets`, { cache: "no-store" }),
        apiFetch(`${API}/api/jobs`, { cache: "no-store" }),
        apiFetch(`${API}/api/data-tasks`, { cache: "no-store" }),
        apiFetch(`${API}/api/data-retention`, { cache: "no-store" }),
      ]);
      if (!overviewResponse.ok || !datasetsResponse.ok || !jobsResponse.ok || !dataTasksResponse.ok || !retentionResponse.ok) throw new Error("API unavailable");
      setOverview(await overviewResponse.json());
      setDatasets(await datasetsResponse.json());
      setJobs(await jobsResponse.json());
      setDataTasks(await dataTasksResponse.json());
      setRetention(await retentionResponse.json());
      setMessage("");
    } catch {
      setMessage("后端尚未启动。启动 quant-web serve 后，本页会自动连接真实数据。 ");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const initial = window.setTimeout(checkAuth, 0);
    return () => window.clearTimeout(initial);
  }, [checkAuth]);

  useEffect(() => {
    if (!auth || !["authenticated", "disabled"].includes(auth.status)) return;
    const initial = window.setTimeout(refresh, 0);
    const timer = window.setInterval(refresh, 5000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
  }, [auth, refresh]);

  async function logout() {
    await apiFetch(`${API}/api/auth/logout`, { method: "POST" });
    await checkAuth();
  }

  const visibleDatasets = useMemo(
    () => datasets.filter((item) => profile === "full" || item.profile === profile || (profile === "research" && item.profile === "core")),
    [datasets, profile],
  );

  async function startBootstrap(event: FormEvent) {
    event.preventDefault();
    setMessage("正在创建初始化任务…");
    const response = await apiFetch(`${API}/api/jobs/bootstrap`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile, start, end: "latest", build_qlib: false }),
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
      body: JSON.stringify({ profile, start, end: "latest" }),
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
          {navGroups.map((group) => (
            <div className="nav-group" key={group.label}>
              <span className="nav-group-label">{group.label}</span>
              {group.items.map((item) => (
                <button className={item.index === activeNav ? "nav-item active" : "nav-item"} key={item.label} disabled={item.index === 10 && auth.user?.role !== "admin"} onClick={() => setActiveNav(item.index)}>
                  <span className="nav-dot" />{item.label}
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className={overview?.credentials_configured ? "pulse ok" : "pulse"} />
          <div><strong>本地研究模式</strong><small>{overview?.credentials_configured ? "数据源已配置" : "等待 Tushare 凭据"}</small></div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div><p className="eyebrow">{headings[activeNav]?.[0]}</p><h1>{headings[activeNav]?.[1]}</h1></div>
          <div className="top-actions"><span className="system-live"><i />数据每 5 秒更新</span><span className="account-chip"><b>{auth.user?.display_name}</b><small>{auth.user?.role}</small></span><button onClick={refresh}>刷新</button>{auth.status === "authenticated" && <button onClick={logout}>退出</button>}</div>
        </header>

        {message && <div className="notice">{message}</div>}

        {activeNav === 0 ? (
          <div className="overview-page">
            <section className="overview-kpis">
              <article><span>数据就绪度</span><strong>{overview?.readiness_percent ?? 0}%</strong><small>{overview?.ready_tasks ?? 0} / {overview?.actionable_tasks ?? 0} 项已可用</small></article>
              <article><span>运行中的任务</span><strong>{overview?.active_jobs ?? 0}</strong><small>{overview?.active_jobs ? "后台正在处理" : "当前队列空闲"}</small></article>
              <article><span>研究数据</span><strong>{overview?.qlib_datasets ?? 0}</strong><small>Qlib 数据集 · {overview?.snapshots ?? 0} 份快照</small></article>
              <article><span>已存数据</span><strong>{formatNumber(overview?.rows ?? 0)}</strong><small>数据行</small></article>
            </section>
            <section className="overview-grid">
              <article className="workspace-card next-actions">
                <div className="section-heading"><div><h2>接下来要做什么</h2><p>只显示当前需要关注的事项。</p></div><button onClick={() => { setActiveNav(1); setDataView("overview"); }}>打开数据中心</button></div>
                {dataTasks.filter((task) => task.status !== "succeeded" && !["permission_probe", "external_source_required"].includes(task.implementation_status)).slice(0, 5).map((task) => <div className="action-row" key={task.task_key}><span className={`task-status ${task.status}`}>{statusLabel(task.status) === task.status ? ({ planned: "待开始", partial: "需补齐" } as Record<string, string>)[task.status] ?? task.status : statusLabel(task.status)}</span><div><strong>{task.title}</strong><small>{task.status === "running" ? `${task.coverage}% · 正在下载` : task.dependencies_satisfied ? "可以开始" : "等待前置数据"}</small></div></div>)}
                {!dataTasks.some((task) => task.status !== "succeeded" && !["permission_probe", "external_source_required"].includes(task.implementation_status)) ? <div className="empty compact">当前数据计划已全部就绪。</div> : null}
              </article>
              <article className="workspace-card quick-entry">
                <div className="section-heading"><div><h2>工作入口</h2><p>按工作目标进入，不必记系统模块。</p></div></div>
                <button onClick={() => { setActiveNav(1); setDataView("create"); }}><b>补充市场数据</b><span>新建日线、分钟线或海外数据任务</span></button>
                <button onClick={() => setActiveNav(11)}><b>查看研究行情</b><span>指数、市场宽度、行业强弱与自选观察池</span></button>
                <button onClick={() => setActiveNav(2)}><b>训练 Qlib 模型</b><span>管理数据集和基线实验</span></button>
                <button onClick={() => setActiveNav(4)}><b>启动自动研究</b><span>因子、参数实验、回测和模拟盘候选</span></button>
                <button onClick={() => setActiveNav(6)}><b>运行策略回测</b><span>验证收益、风险和交易约束</span></button>
                <button onClick={() => setActiveNav(8)}><b>查看模拟组合</b><span>持仓、订单和风险状态</span></button>
              </article>
            </section>
          </div>
        ) : activeNav === 1 ? (
          <div className="data-center-page">
            <div className="page-tabs" role="tablist" aria-label="数据中心页面">
              {[['overview', '运行概况'], ['catalog', '数据目录'], ['create', '新建任务'], ['runs', '运行记录'], ['storage', '存储与版本']].map(([value, label]) => <button role="tab" aria-selected={dataView === value} className={dataView === value ? "active" : ""} onClick={() => setDataView(value)} key={value}>{label}{value === "runs" && overview?.active_jobs ? <i>{overview.active_jobs}</i> : null}</button>)}
            </div>

            {dataView === "overview" ? <>
              <section className="readiness-hero">
                <div className="readiness-copy"><span className="status-chip">{loading ? "正在连接" : overview?.active_jobs ? "数据准备中" : (overview?.partial_tasks || overview?.waiting_tasks) ? "仍有数据待处理" : "数据已就绪"}</span><h2>{overview?.ready_tasks ?? 0} / {overview?.actionable_tasks ?? 0} 项数据已可用</h2><p>这是按当前数据目录计算的真实就绪度；基金、宏观、衍生品、海外和分钟数据都会计入。</p></div>
                <div className="readiness-ring" style={{ "--progress": `${overview?.readiness_percent ?? 0}%` } as CSSProperties}><strong>{overview?.readiness_percent ?? 0}%</strong><span>当前计划</span></div>
              </section>
              <section className="status-strip">
                <article><span>正在运行</span><strong>{overview?.running_tasks ?? 0}</strong></article>
                <article><span>需要补齐</span><strong>{overview?.partial_tasks ?? 0}</strong></article>
                <article><span>等待开始</span><strong>{overview?.waiting_tasks ?? 0}</strong></article>
                <article><span>失败</span><strong className={overview?.failed_tasks ? "danger" : ""}>{overview?.failed_tasks ?? 0}</strong></article>
                <article><span>基础日线下载</span><strong>{overview?.legacy_download_coverage ?? 0}%</strong><small>旧工作单元，仅供核对</small></article>
              </section>
              <section className="overview-grid data-overview-grid">
                <article className="workspace-card next-actions">
                  <div className="section-heading"><div><h2>需要处理</h2><p>优先显示运行中、失败和可以启动的数据。</p></div><button onClick={() => setDataView("catalog")}>查看全部</button></div>
                  {dataTasks.filter((task) => task.status !== "succeeded" && !["permission_probe", "external_source_required"].includes(task.implementation_status)).slice(0, 6).map((task) => <div className="action-row" key={task.task_key}><span className={`task-status ${task.status}`}>{({ planned: "待开始", queued: "排队中", running: "下载中", partial: "需补齐", failed: "失败" } as Record<string, string>)[task.status] ?? task.status}</span><div><strong>{task.title}</strong><small>{task.status === "running" ? `${task.coverage}% · ${formatNumber(task.rows)} 行` : task.dependencies_satisfied ? "已具备启动条件" : "等待前置数据"}</small></div></div>)}
                </article>
                <article className="workspace-card active-runs">
                  <div className="section-heading"><div><h2>当前队列</h2><p>任务在后台持续运行，关闭浏览器不会中断。</p></div><button onClick={() => setDataView("runs")}>运行记录</button></div>
                  {jobs.filter((job) => ["queued", "running"].includes(job.status)).slice(0, 5).map((job) => <div className="run-row" key={job.id}><span className={`job-state ${job.status}`} /><div><strong>{job.payload.snapshot_name ?? job.payload.profile ?? job.kind}</strong><small>{statusLabel(job.status)} · {new Date(job.created_at).toLocaleString("zh-CN")}</small></div></div>)}
                  {!jobs.some((job) => ["queued", "running"].includes(job.status)) ? <div className="empty compact">当前没有运行中的任务。</div> : null}
                </article>
              </section>
            </> : null}

            {dataView === "catalog" ? <DataTaskCenter tasks={dataTasks} api={API} mode="catalog" onCreated={refresh} onMessage={setMessage} /> : null}

            {dataView === "create" ? <div className="create-layout">
              <form className="bootstrap-card compact-bootstrap" onSubmit={startBootstrap}>
                <div className="card-heading"><div><span>首次使用</span><strong>初始化 A 股日线基础库</strong></div><span className="status-chip">可恢复</span></div>
                <p>下载行情、复权、财务和交易约束。下载完成后，质量校验、快照和 Qlib 构建独立执行，失败可单独重试。</p>
                <label>数据范围<select value={profile} onChange={(e) => setProfile(e.target.value)}><option value="core">基础 · 价格与交易约束</option><option value="research">研究 · 资金与行业</option><option value="full">完整 · 财务与事件</option></select></label>
                <div className="form-row"><label>开始日期<input type="date" value={start} onChange={(e) => setStart(e.target.value)} /></label><label>结束日期<input value="最新交易日" disabled /></label></div>
                <button className="primary" disabled={!overview?.credentials_configured || !!overview?.active_jobs}>创建基础下载任务</button>
                <button type="button" onClick={finalizeData} disabled={!!overview?.active_jobs || !overview?.planned_units || overview.succeeded_units !== overview.planned_units}>基础下载完成，开始质量校验</button>
              </form>
              <DataTaskCenter tasks={dataTasks} api={API} mode="create" onCreated={refresh} onMessage={setMessage} />
            </div> : null}

            {dataView === "runs" ? <JobRunCenter api={API} canControl={auth.user?.role === "admin" || auth.user?.role === "operator"} onChanged={refresh} onMessage={setMessage} /> : null}

            {dataView === "storage" ? <div className="storage-stack">
              <section className="data-panel storage-panel"><div className="panel-heading"><div><h2>不可变快照与 Qlib 数据</h2><p>被研究、回测和模拟组合引用的数据不会被清理。</p></div><span>{formatBytes(retention?.eligible_bytes ?? 0)} 可清理</span></div><div className="table-wrap"><table><thead><tr><th>选择</th><th>数据集</th><th>创建时间</th><th>占用</th><th>状态 / 原因</th></tr></thead><tbody>{retention?.entries.map((item) => <tr key={item.name}><td><input aria-label={`选择清理 ${item.name}`} type="checkbox" disabled={item.state !== "eligible" || auth.user?.role !== "admin"} checked={!!retentionSelection[item.name]} onChange={(event) => setRetentionSelection({ ...retentionSelection, [item.name]: event.target.checked })} /></td><td><code>{item.name}</code></td><td>{new Date(item.created_at).toLocaleString("zh-CN")}</td><td>{formatBytes(item.bytes)}</td><td><span className={`state ${item.state === "eligible" ? "failed" : "ready"}`}>{item.state === "eligible" ? "可清理" : "受保护"}</span><small>{item.reasons.join("；") || "未被引用且已过保留期"}</small></td></tr>)}</tbody></table>{!retention?.entries.length ? <div className="empty compact">尚无快照或 Qlib 数据集。</div> : null}</div>{auth.user?.role === "admin" ? <div className="retention-action"><label>输入 DELETE_UNREFERENCED_DATASETS 确认<input value={retentionConfirmation} onChange={(event) => setRetentionConfirmation(event.target.value)} /></label><button className="primary" onClick={applyRetention} disabled={retentionConfirmation !== "DELETE_UNREFERENCED_DATASETS" || !Object.values(retentionSelection).some(Boolean)}>清理所选数据</button></div> : null}</section>
              <section className="data-panel"><div className="panel-heading"><div><h2>基础数据集</h2><p>用于核对旧初始化流水线的工作单元。</p></div><div className="segmented">{["core", "research", "full"].map((value) => <button className={profile === value ? "selected" : ""} onClick={() => setProfile(value)} key={value}>{value}</button>)}</div></div><div className="table-wrap"><table><thead><tr><th>数据集</th><th>层级</th><th>状态</th><th>进度</th><th>数据行</th><th>失败</th></tr></thead><tbody>{visibleDatasets.map((item) => <tr key={item.name}><td><code>{item.name}</code></td><td><span className={`tier ${item.profile}`}>{item.profile}</span></td><td><span className={`state ${item.state}`}>{item.state === "ready" ? "已就绪" : item.state === "partial" ? "部分完成" : "未下载"}</span></td><td><div className="mini-progress"><i style={{ width: `${item.coverage}%` }} /></div><small>{item.coverage}%</small></td><td>{formatNumber(item.rows)}</td><td className={item.failed ? "danger" : "muted"}>{item.failed}</td></tr>)}</tbody></table></div></section>
            </div> : null}
          </div>
        ) : activeNav === 11 ? <MarketOverviewPanel api={API} onOpenData={() => { setActiveNav(1); setDataView("overview"); }} /> : activeNav === 2 ? <QlibPanel api={API} /> : activeNav === 3 ? <RDAgentPanel api={API} /> : activeNav === 4 ? <ResearchCampaignPanel api={API} /> : activeNav === 5 ? <FactorLibraryPanel api={API} /> : activeNav === 6 ? <BacktestPanel api={API} /> : activeNav === 7 ? <PairTradingPanel api={API} /> : activeNav === 8 ? <PortfolioPanel api={API} /> : activeNav === 9 ? <OperationsPanel api={API} currentUser={auth.user!} /> : <SettingsPanel api={API} />}
      </section>
    </main>
  );
}
