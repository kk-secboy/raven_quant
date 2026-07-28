"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "./api-client";
import { AuthPanel, AuthState } from "./auth-panel";
import { BacktestPanel } from "./backtest-panel";
import { DataTaskCenter } from "./data-task-center";
import { FactorLibraryPanel } from "./factor-library-panel";
import { JobRunCenter } from "./job-run-center";
import { MarketOverviewPanel } from "./market-overview-panel";
import { PairSatellitePanel } from "./pair-satellite-panel";
import { PortfolioPanel } from "./portfolio-panel";
import { QlibPanel } from "./qlib-panel";
import { RDAgentPanel } from "./rdagent-panel";
import { ResearchCampaignPanel } from "./research-campaign-panel";
import { StrategyAllocationPanel } from "./strategy-allocation-panel";

const API = process.env.NEXT_PUBLIC_API_BASE ?? "";

const sections = [
  "总览",
  "数据快照",
  "RD-Agent 研究",
  "因子准入",
  "Qlib 回测与审批",
  "核心 / 卫星分配",
  "统一模拟盘",
  "任务与告警",
] as const;

type Overview = {
  mode?: string;
  rows?: number;
  snapshots?: number;
  qlib_datasets?: number;
  active_jobs?: number;
  readiness_percent?: number;
};

export default function Home() {
  const [auth, setAuth] = useState<AuthState | null>(null);
  const [active, setActive] = useState(0);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [message, setMessage] = useState("");

  const checkAuth = useCallback(async () => {
    try {
      const response = await apiFetch(`${API}/api/auth/state`, { cache: "no-store" });
      if (!response.ok) throw new Error("auth unavailable");
      setAuth(await response.json());
    } catch {
      setAuth({ status: "login_required", user: null });
      setMessage("后端尚未启动或认证服务不可用。");
    }
  }, []);

  const refreshOverview = useCallback(async () => {
    try {
      const response = await apiFetch(`${API}/api/overview`, { cache: "no-store" });
      if (!response.ok) throw new Error("overview unavailable");
      setOverview(await response.json());
      setMessage("");
    } catch {
      setMessage("无法读取运行状态；所有写操作保持关闭。");
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void checkAuth(), 0);
    return () => window.clearTimeout(timer);
  }, [checkAuth]);

  useEffect(() => {
    if (auth && ["authenticated", "disabled"].includes(auth.status)) {
      const timer = window.setTimeout(() => void refreshOverview(), 0);
      return () => window.clearTimeout(timer);
    }
  }, [auth, refreshOverview]);

  if (!auth) {
    return (
      <main className="loading-shell">
        <section className="loading-card">
          <div className="brand"><span className="brand-mark">Q</span><span>QuantLab</span></div>
          <p className="eyebrow">TUSHARE / QLIB / RD-AGENT</p>
          <h1>正在检查安全会话</h1>
          <p className="muted">基于 Tushare、Qlib 与 RD-Agent 的本地量化研究平台</p>
        </section>
      </main>
    );
  }

  if (!["authenticated", "disabled"].includes(auth.status)) {
    return <AuthPanel api={API} state={auth} onAuthenticated={checkAuth} />;
  }

  const panels = [
    <div className="section-stack" key="overview">
      <OverviewPanel overview={overview} />
      <MarketOverviewPanel api={API} />
    </div>,
    <DataTaskCenter key="data" api={API} />,
    <div className="section-stack" key="rdagent">
      <ResearchCampaignPanel api={API} />
      <RDAgentPanel api={API} />
      <PairSatellitePanel api={API} />
    </div>,
    <FactorLibraryPanel key="factors" api={API} />,
    <div className="section-stack" key="backtests">
      <QlibPanel api={API} />
      <BacktestPanel api={API} />
    </div>,
    <StrategyAllocationPanel key="allocation" api={API} />,
    <PortfolioPanel key="portfolio" api={API} />,
    <JobRunCenter key="jobs" api={API} />,
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">Q</span><span>QuantLab</span></div>
        <nav className="nav" aria-label="主要导航">
          {sections.map((section, index) => (
            <button key={section} className={active === index ? "active" : ""} onClick={() => setActive(index)}>
              {section}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">只读研究与模拟控制台<br />不产生真实券商指令</div>
      </aside>
      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">GOVERNED SINGLE MAINLINE</p>
            <h1>{sections[active]}</h1>
          </div>
          <div className="top-actions">
            <button className="button" onClick={() => void refreshOverview()}>刷新</button>
            <span className="badge good">SAFE / SIMULATION ONLY</span>
          </div>
        </header>
        {message && <div className="message">{message}</div>}
        {panels[active]}
      </main>
    </div>
  );
}

function OverviewPanel({ overview }: { overview: Overview | null }) {
  return (
    <>
      <section className="status-strip">
        <div className="metric"><span>数据行</span><strong>{overview?.rows?.toLocaleString("zh-CN") ?? "—"}</strong></div>
        <div className="metric"><span>不可变快照</span><strong>{overview?.snapshots ?? "—"}</strong></div>
        <div className="metric"><span>Qlib 数据集</span><strong>{overview?.qlib_datasets ?? "—"}</strong></div>
        <div className="metric"><span>目录就绪度</span><strong>{overview?.readiness_percent ?? 0}%</strong></div>
      </section>
      <div className="grid">
        <section className="panel">
          <p className="eyebrow">PRODUCTION CONTRACT</p>
          <h2>唯一研究与模拟主线</h2>
          <p className="muted">数据快照 → 因子准入 → Qlib 正式验证 → 账户分配 → 统一模拟账本。</p>
        </section>
        <section className="panel">
          <p className="eyebrow">RECOVERY</p>
          <h2>成功 checkpoint</h2>
          <p className="muted">失败任务保留证据并从已验证检查点继续，不静默重算或跳过。</p>
        </section>
      </div>
    </>
  );
}
