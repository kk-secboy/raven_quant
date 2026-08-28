"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";

type AutomationConfig = {
  contract_version: string;
  enabled: boolean;
  timezone: string;
  market_daily_time: string;
  market_lookback_days: number;
  research_assets_time: string;
  research_assets_history_start: string;
  information_daily_time: string;
  information_lookback_days: number;
  information_weekly_time: string;
  information_weekday: number;
  ashare_5m_time: string;
  ashare_5m_history_start: string;
  auxiliary_daily_time: string;
  auxiliary_history_start: string;
  max_stocks: number;
  max_options: number;
  download_workers: number;
  requests_per_minute: number;
  strategy_minute_symbols: string[];
};

type CoverageTask = {
  task_key: string;
  title: string;
  frequency: string;
  schedule_group: string;
  schedule_id?: string | null;
  covered: boolean;
  reason: string;
};

type ManagedSchedule = {
  id: string;
  name: string;
  kind: string;
  status: string;
  desired_status: string;
  run_time: string;
  next_run_at?: string | null;
  last_run_at?: string | null;
};

type AutomationState = {
  config: AutomationConfig;
  source: string;
  revision: number;
  updated_by?: string | null;
  updated_at?: string | null;
  blocked: string[];
  coverage: {
    covered: number;
    total: number;
    ready: boolean;
    tasks: CoverageTask[];
  };
  schedules: ManagedSchedule[];
};

const groupLabels: Record<string, string> = {
  market_daily: "日频行情、扩展数据与Qlib",
  research_assets: "研报元数据与PDF入口",
  information_daily: "公告、语料NLP与事件标签",
  information_weekly: "结构化信息因子周更",
  ashare_5m: "全A股5分钟及Qlib",
  auxiliary_daily: "融资资格、核心1分钟与专项分钟",
};

function timeText(value?: string | null) {
  return value ? new Date(value).toLocaleString("zh-CN") : "尚未运行";
}

export function DataAutomationPanel({ api }: { api: string }) {
  const [state, setState] = useState<AutomationState | null>(null);
  const [config, setConfig] = useState<AutomationConfig | null>(null);
  const [symbols, setSymbols] = useState("");
  const [reason, setReason] = useState("通过 Web 更新完整数据自动化策略");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    const response = await apiFetch(`${api}/api/data-automation`, { cache: "no-store" });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail ?? "无法读取数据自动更新配置");
    setState(body);
    setConfig(body.config);
    setSymbols(body.config.strategy_minute_symbols.join("\n"));
  }, [api]);

  useEffect(() => {
    const initial = window.setTimeout(() => load().catch((error) => setMessage(error.message)), 0);
    return () => window.clearTimeout(initial);
  }, [load]);

  const uncovered = useMemo(
    () => state?.coverage.tasks.filter((item) => !item.covered) ?? [],
    [state],
  );

  function update<K extends keyof AutomationConfig>(key: K, value: AutomationConfig[K]) {
    setConfig((current) => current ? { ...current, [key]: value } : current);
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!config) return;
    setBusy(true); setMessage("正在保存并对账6条更新链……");
    try {
      const next = {
        ...config,
        strategy_minute_symbols: symbols.split(/[\s,]+/).map((item) => item.trim()).filter(Boolean),
      };
      const response = await apiFetch(`${api}/api/data-automation`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: next, reason }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "数据自动化配置保存失败");
      setState(body); setConfig(body.config);
      setSymbols(body.config.strategy_minute_symbols.join("\n"));
      setMessage(`已保存第 ${body.revision} 版配置，覆盖 ${body.coverage.covered}/${body.coverage.total} 项。`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "数据自动化配置保存失败");
    } finally { setBusy(false); }
  }

  async function reconcile() {
    setBusy(true); setMessage("正在补齐并校验默认更新计划……");
    try {
      const response = await apiFetch(`${api}/api/data-automation/reconcile`, { method: "POST" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "更新计划对账失败");
      setState(body); setMessage(`计划对账完成：${body.coverage.covered}/${body.coverage.total} 项已覆盖。`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "更新计划对账失败");
    } finally { setBusy(false); }
  }

  async function run(group: string) {
    setBusy(true); setMessage(`正在补跑：${groupLabels[group]}……`);
    try {
      const response = await apiFetch(`${api}/api/data-automation/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ groups: [group] }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "补跑任务创建失败");
      setMessage(`${groupLabels[group]}已进入队列，可在“任务与告警”查看进度。`);
      await load();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "补跑任务创建失败");
    } finally { setBusy(false); }
  }

  if (!config || !state) return <section className="settings-card"><p>正在读取数据自动更新配置……</p></section>;

  return <section className="data-automation-settings">
    {message ? <div className="notice">{message}</div> : null}
    <div className="card-heading">
      <div><span>WEB DATA CONTROL PLANE</span><strong>33项数据自动更新</strong></div>
      <span className={state.coverage.ready ? "status-chip verified" : "status-chip"}>{state.coverage.covered}/{state.coverage.total}</span>
    </div>
    <p>日常更新、补跑、暂停和参数调整统一从这里完成；数据库地址、加密主密钥和容器权限仍由部署环境保护。</p>

    <div className="settings-grid">
      {state.schedules.map((item) => {
        const group = Object.keys(groupLabels).find((key) => state.coverage.tasks.some((task) => task.schedule_group === key && task.schedule_id === item.id));
        return <article className="settings-card" key={item.id}>
          <div className="card-heading"><div><span>{item.kind}</span><strong>{group ? groupLabels[group] : item.name}</strong></div><span className={item.status === "active" ? "status-chip verified" : "status-chip"}>{item.status === "active" ? "已启用" : "已暂停"}</span></div>
          <p>时间 {item.run_time} · 上次 {timeText(item.last_run_at)} · 下次 {timeText(item.next_run_at)}</p>
          {group ? <button type="button" disabled={busy} onClick={() => run(group)}>立即补跑</button> : null}
        </article>;
      })}
    </div>

    {uncovered.length || state.blocked.length ? <div className="notice">
      <strong>仍需处理：</strong>{[...state.blocked, ...uncovered.map((item) => `${item.title}（${item.reason}）`)].join("；")}
    </div> : null}

    <form className="settings-card" onSubmit={save}>
      <div className="card-heading"><div><span>VERSIONED POLICY</span><strong>更新策略</strong></div><span>修订 {state.revision}</span></div>
      <label><input type="checkbox" checked={config.enabled} onChange={(event) => update("enabled", event.target.checked)} /> 默认启用全部更新链</label>
      <div className="automation-form-grid">
        <label>日频与Qlib时间<input type="time" value={config.market_daily_time} onChange={(event) => update("market_daily_time", event.target.value)} /></label>
        <label>日频回看天数<input type="number" min={1} max={90} value={config.market_lookback_days} onChange={(event) => update("market_lookback_days", Number(event.target.value))} /></label>
        <label>研报快照时间<input type="time" value={config.research_assets_time} onChange={(event) => update("research_assets_time", event.target.value)} /></label>
        <label>研报历史起点<input type="date" value={config.research_assets_history_start} onChange={(event) => update("research_assets_history_start", event.target.value)} /></label>
        <label>信息NLP时间<input type="time" value={config.information_daily_time} onChange={(event) => update("information_daily_time", event.target.value)} /></label>
        <label>信息回看天数<input type="number" min={1} max={30} value={config.information_lookback_days} onChange={(event) => update("information_lookback_days", Number(event.target.value))} /></label>
        <label>5分钟更新时间<input type="time" value={config.ashare_5m_time} onChange={(event) => update("ashare_5m_time", event.target.value)} /></label>
        <label>辅助数据时间<input type="time" value={config.auxiliary_daily_time} onChange={(event) => update("auxiliary_daily_time", event.target.value)} /></label>
        <label>核心股票数量<input type="number" min={1} max={500} value={config.max_stocks} onChange={(event) => update("max_stocks", Number(event.target.value))} /></label>
        <label>活跃期权数量<input type="number" min={1} max={500} value={config.max_options} onChange={(event) => update("max_options", Number(event.target.value))} /></label>
        <label>下载并发数<input type="number" min={1} max={16} value={config.download_workers} onChange={(event) => update("download_workers", Number(event.target.value))} /></label>
        <label>接口每分钟上限<input type="number" min={1} max={99} value={config.requests_per_minute} onChange={(event) => update("requests_per_minute", Number(event.target.value))} /></label>
      </div>
      <label>专项分钟标的<textarea rows={8} value={symbols} onChange={(event) => setSymbols(event.target.value)} /></label>
      <label>修改原因<input value={reason} minLength={10} onChange={(event) => setReason(event.target.value)} required /></label>
      <div className="job-run-actions"><button className="primary" disabled={busy} type="submit">保存并立即对账</button><button disabled={busy} type="button" onClick={reconcile}>恢复/补齐默认计划</button></div>
    </form>

    <details><summary>查看33项覆盖明细</summary><div className="job-run-list">{state.coverage.tasks.map((item) => <div key={item.task_key}><strong>{item.title}</strong><small>{item.frequency} · {groupLabels[item.schedule_group]} · {item.covered ? "已覆盖" : item.reason}</small></div>)}</div></details>
  </section>;
}
