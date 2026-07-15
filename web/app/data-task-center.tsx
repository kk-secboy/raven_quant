"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";

export type DataTask = {
  task_key: string;
  phase: number;
  title: string;
  description: string;
  category: string;
  source: string;
  status: string;
  implementation_status: string;
  depends_on: string[];
  dependencies_satisfied: boolean;
  estimated_storage_gb?: number | null;
  job_id?: string | null;
  error?: string | null;
  coverage: number;
  rows: number;
  config: {
    datasets: string[];
    frequency: string;
    range_start: string;
    range_end: string;
  };
};

type Snapshot = { name: string; frequency?: string; end_date?: string };

const statusText: Record<string, string> = {
  planned: "待开始",
  queued: "排队中",
  running: "下载中",
  succeeded: "已可用",
  partial: "需补齐",
  failed: "失败",
};

const implementationText: Record<string, string> = {
  ready: "可直接下载",
  partial_ready: "部分接口可用",
  catalogued: "等待接入",
  permission_probe: "需要检查 Tushare 权限",
  external_source_required: "需要其他数据源",
};

const supplementalBundles = new Set([
  "cn_extended_daily", "cn_funds", "cn_macro", "cn_futures",
  "cn_institutional", "cn_options_bonds", "hk_market", "us_market", "global_markets",
  "cn_governance_risk", "cn_capital_flow", "cn_fund_index_enhanced",
  "cn_derivatives_enhanced", "global_rates_enhanced", "research_corpus",
  "strategy_specialty", "strategy_specialty_minutes",
]);

const groupDefinitions = [
  { key: "cn", label: "A 股基础与研究", description: "行情、复权、财务、因子和 Qlib 研究底座" },
  { key: "fund", label: "基金与宏观", description: "ETF、基金、利率、经济和行业环境" },
  { key: "execution", label: "衍生品与执行", description: "期货、期权、债券、融券资格和分钟数据" },
  { key: "overseas", label: "海外市场", description: "港股、美股及全球主要市场" },
  { key: "advanced", label: "高级与外部数据", description: "权限受限或需要额外数据源的数据" },
];

function groupFor(task: DataTask) {
  if (["cn_funds", "cn_macro", "cn_institutional", "cn_fund_index_enhanced", "global_rates_enhanced"].includes(task.task_key)) return "fund";
  if (["cn_futures", "cn_options_bonds", "cn_derivatives_enhanced", "strategy_specialty_minutes", "cn_margin_eligibility", "liquid_intraday_1m", "liquid_intraday_qlib", "cn_ashare_5m", "cn_ashare_5m_qlib"].includes(task.task_key)) return "execution";
  if (["hk_market", "us_market", "global_markets"].includes(task.task_key)) return "overseas";
  if (task.task_key === "strategy_specialty") return "advanced";
  if (["permission_probe", "external_source_required"].includes(task.implementation_status)) return "advanced";
  return "cn";
}

type DataTaskCenterProps = {
  tasks: DataTask[];
  api: string;
  mode: "catalog" | "create";
  onCreated: () => Promise<void>;
  onMessage: (message: string) => void;
};

export function DataTaskCenter({ tasks, api, mode, onCreated, onMessage }: DataTaskCenterProps) {
  const [start, setStart] = useState("2024-01-01");
  const [end, setEnd] = useState("");
  const [etfs, setEtfs] = useState("510300.SH,159919.SZ");
  const [stocks, setStocks] = useState("");
  const [indices, setIndices] = useState("");
  const [futures, setFutures] = useState("");
  const [options, setOptions] = useState("");
  const [autoSelect, setAutoSelect] = useState(true);
  const [maxStocks, setMaxStocks] = useState(100);
  const [maxOptions, setMaxOptions] = useState(100);
  const [etfCategories, setEtfCategories] = useState(["broad", "industry", "gold", "bond"]);
  const [symbols, setSymbols] = useState("");
  const [selectedBundle, setSelectedBundle] = useState("cn_extended_daily");
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [catalogFilter, setCatalogFilter] = useState("attention");
  const [query, setQuery] = useState("");
  const [createMode, setCreateMode] = useState("supplemental");
  const [minuteSnapshots, setMinuteSnapshots] = useState<Snapshot[]>([]);
  const [minuteSnapshot, setMinuteSnapshot] = useState("");
  const active = tasks.filter((task) => ["queued", "running"].includes(task.status)).length;
  const marginTask = tasks.find((task) => task.task_key === "cn_margin_eligibility");
  const minuteTask = tasks.find((task) => task.task_key === "liquid_intraday_1m");
  const minuteQlibTask = tasks.find((task) => task.task_key === "liquid_intraday_qlib");
  const ashare5mTask = tasks.find((task) => task.task_key === "cn_ashare_5m");
  const supplementalTasks = tasks.filter((task) => supplementalBundles.has(task.task_key));
  const selectedTask = supplementalTasks.find((task) => task.task_key === selectedBundle);

  useEffect(() => {
    let cancelled = false;
    apiFetch(`${api}/api/snapshots`, { cache: "no-store" }).then(async (response) => {
      if (!response.ok || cancelled) return;
      const values = (await response.json() as Snapshot[]).filter((item) => ["1min", "5min"].includes(item.frequency ?? ""));
      setMinuteSnapshots(values);
      setMinuteSnapshot((current) => current || values[0]?.name || "");
    });
    return () => { cancelled = true; };
  }, [api]);

  const visibleTasks = useMemo(() => tasks.filter((task) => {
    if (catalogFilter === "attention" && task.status === "succeeded") return false;
    if (catalogFilter === "ready" && task.status !== "succeeded") return false;
    const haystack = `${task.title} ${task.description} ${task.category} ${task.source}`.toLowerCase();
    return haystack.includes(query.trim().toLowerCase());
  }), [tasks, catalogFilter, query]);

  async function submit(path: string, body: Record<string, unknown>, label: string) {
    setSubmitting(path);
    onMessage(`正在创建“${label}”任务…`);
    try {
      const response = await apiFetch(`${api}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!response.ok) { onMessage(payload.detail ?? `${label}任务创建失败`); return; }
      onMessage(`“${label}”已进入任务队列。`);
      await onCreated();
    } finally {
      setSubmitting(null);
    }
  }

  function startMargin(event: FormEvent) {
    event.preventDefault();
    void submit("/api/jobs/margin-eligibility", { start, end: end || "latest" }, "融券资格历史");
  }

  function startMinutes(event: FormEvent) {
    event.preventDefault();
    const codes = (value: string) => value.split(",").map((item) => item.trim()).filter(Boolean);
    void submit("/api/jobs/core-intraday", {
      start, end: end || "latest", etfs: codes(etfs), stocks: codes(stocks), indices: codes(indices),
      futures: codes(futures), options: codes(options), auto_select: autoSelect,
      max_stocks: maxStocks, max_options: maxOptions, etf_categories: etfCategories,
    }, "核心资产 1 分钟线");
  }

  function startSupplemental(task: DataTask) {
    void submit("/api/jobs/supplemental-download", {
      bundle: task.task_key,
      start,
      end: end || "latest",
      symbols: symbols.split(",").map((item) => item.trim()).filter(Boolean),
    }, task.title);
  }

  function startMinuteQlib(snapshotName = minuteSnapshot) {
    if (!snapshotName) { onMessage("尚无已完成的分钟不可变快照。"); return; }
    void submit("/api/jobs/minute-qlib", { snapshot_name: snapshotName }, "分钟 Qlib 数据集");
  }

  function startAshare5m() {
    void submit(
      "/api/jobs/ashare-5m",
      { start, end: end || "latest" },
      "全 A 股 5 分钟线",
    );
  }

  function retryTask(task: DataTask) {
    if (task.job_id) void submit(`/api/jobs/${task.job_id}/retry`, {}, `${task.title}重试`);
  }

  function toggleEtfCategory(category: string) {
    setEtfCategories((current) => current.includes(category)
      ? current.filter((item) => item !== category)
      : [...current, category]);
  }

  if (mode === "catalog") {
    return (
      <section className="data-task-center catalog-view">
        <div className="catalog-toolbar">
          <div>
            <h2>数据目录</h2>
            <p>按用途查看，不需要理解内部流水线。默认只显示仍需处理的数据。</p>
          </div>
          <div className="catalog-controls">
            <input aria-label="搜索数据" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索行情、基金、期货…" />
            <div className="segmented">
              {[['attention', '待处理'], ['ready', '已可用'], ['all', '全部']].map(([value, label]) => (
                <button type="button" className={catalogFilter === value ? "selected" : ""} onClick={() => setCatalogFilter(value)} key={value}>{label}</button>
              ))}
            </div>
          </div>
        </div>
        <div className="catalog-groups">
          {groupDefinitions.map((group) => {
            const groupTasks = visibleTasks.filter((task) => groupFor(task) === group.key);
            if (!groupTasks.length) return null;
            return (
              <section className="catalog-group" key={group.key}>
                <header><div><h3>{group.label}</h3><p>{group.description}</p></div><span>{groupTasks.length} 项</span></header>
                <div className="catalog-task-list">
                  {groupTasks.map((task) => (
                    <article className={`catalog-task ${task.status}`} key={task.task_key}>
                      <div className="catalog-task-main">
                        <span className={`task-status ${task.status}`}>{statusText[task.status] ?? task.status}</span>
                        <div><strong>{task.title}</strong><p>{task.description}</p></div>
                      </div>
                      <div className="catalog-task-facts">
                        <span>{task.config.frequency}</span>
                        <span>{task.config.range_start} 至最新</span>
                        <span>{task.coverage}%</span>
                      </div>
                      <div className="catalog-task-actions">
                        {supplementalBundles.has(task.task_key) && task.task_key !== "strategy_specialty_minutes" && !["queued", "running"].includes(task.status) ? (
                          <button type="button" disabled={submitting !== null || !task.dependencies_satisfied} onClick={() => startSupplemental(task)}>{task.status === "partial" ? "补齐" : task.status === "succeeded" ? "更新" : "下载"}</button>
                        ) : null}
                        {task.task_key === "strategy_specialty_minutes" && !["queued", "running"].includes(task.status) ? <button type="button" onClick={() => onMessage("请到“新建任务”填写证券代码后启动策略分钟专项数据。")}>配置后下载</button> : null}
                        {["liquid_intraday_qlib", "cn_ashare_5m_qlib"].includes(task.task_key) && !["queued", "running"].includes(task.status) ? (() => { const snapshot = minuteSnapshots.find((item) => item.frequency === task.config.frequency)?.name ?? ""; return <button type="button" disabled={submitting !== null || !task.dependencies_satisfied || !snapshot} onClick={() => startMinuteQlib(snapshot)}>构建 Qlib</button>; })() : null}
                        {task.task_key === "cn_ashare_5m" && !["queued", "running"].includes(task.status) ? <button type="button" disabled={submitting !== null || !task.dependencies_satisfied} onClick={startAshare5m}>下载/更新</button> : null}
                        {task.status === "failed" && task.job_id ? <button className="danger-button" type="button" disabled={submitting !== null} onClick={() => retryTask(task)}>重试</button> : null}
                        <details>
                          <summary>详情</summary>
                          <div className="task-detail-popover">
                            <span>来源：{task.source}</span>
                            <span>数据量：{task.rows.toLocaleString("zh-CN")} 行</span>
                            <span>预计空间：{task.estimated_storage_gb ? `${task.estimated_storage_gb} GB` : "按实际数据量"}</span>
                            <span>{implementationText[task.implementation_status] ?? task.implementation_status}</span>
                            {task.error ? <b>{task.error}</b> : null}
                          </div>
                        </details>
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            );
          })}
          {!visibleTasks.length ? <div className="empty compact">当前筛选下没有数据项。</div> : null}
        </div>
      </section>
    );
  }

  return (
    <section className="data-task-center create-task-view">
      <div className="create-task-head">
        <div><h2>新建下载任务</h2><p>选择要补充的数据，日期和高级参数可随时调整。</p></div>
        <span className="status-chip">{active ? `${active} 项正在运行` : "队列空闲"}</span>
      </div>
      <div className="page-tabs compact-tabs">
        {[['supplemental', '日线与扩展数据'], ['minute', '分钟数据'], ['margin', '融券资格']].map(([value, label]) => (
          <button type="button" className={createMode === value ? "active" : ""} onClick={() => setCreateMode(value)} key={value}>{label}</button>
        ))}
      </div>

      {createMode === "supplemental" && (
        <form className="task-form" onSubmit={(event) => { event.preventDefault(); if (selectedTask) startSupplemental(selectedTask); }}>
          <div className="form-intro"><strong>选择数据范围</strong><span>适合补充基金、宏观、衍生品和海外市场数据。</span></div>
          <label>数据包<select value={selectedBundle} onChange={(event) => { setSelectedBundle(event.target.value); setSymbols(""); }}>{supplementalTasks.map((task) => <option value={task.task_key} key={task.task_key}>{task.title} · {statusText[task.status] ?? task.status}</option>)}</select></label>
          {selectedTask ? <p className="selection-help">{selectedTask.description}</p> : null}
          <div className="form-row">
            <label>开始日期<input type="date" value={start} onChange={(event) => setStart(event.target.value)} /></label>
            <label>结束日期<input type="date" value={end} onChange={(event) => setEnd(event.target.value)} /><small>留空表示最新交易日</small></label>
          </div>
          {["hk_market", "us_market", "strategy_specialty_minutes"].includes(selectedBundle) ? <label>证券代码（{selectedBundle === "strategy_specialty_minutes" ? "必填" : "可选"}）<input required={selectedBundle === "strategy_specialty_minutes"} value={symbols} onChange={(event) => setSymbols(event.target.value)} placeholder={selectedBundle === "hk_market" ? "00700.HK,00941.HK" : selectedBundle === "us_market" ? "AAPL,MSFT,NVDA" : "000001.SZ,600519.SH"} /><small>{selectedBundle === "strategy_specialty_minutes" ? "分钟专项接口必须明确证券范围，避免误拉全市场。" : "留空下载当前配置允许的全市场范围"}</small></label> : null}
          <button className="primary" disabled={!selectedTask || submitting !== null || !selectedTask.dependencies_satisfied || ["queued", "running"].includes(selectedTask.status) || (selectedBundle === "strategy_specialty_minutes" && !symbols.trim())}>创建下载任务</button>
          {selectedTask && !selectedTask.dependencies_satisfied ? <p className="form-warning">前置数据尚未完成，暂不能启动。</p> : null}
        </form>
      )}

      {createMode === "margin" && (
        <form className="task-form" onSubmit={startMargin}>
          <div className="form-intro"><strong>逐日融券资格证据</strong><span>保存每个交易日明确可融券的标的，供策略和回测使用。</span></div>
          <div className="form-row"><label>开始日期<input type="date" value={start} onChange={(event) => setStart(event.target.value)} /></label><label>结束日期<input type="date" value={end} onChange={(event) => setEnd(event.target.value)} /><small>留空表示最新交易日</small></label></div>
          <button className="primary" disabled={submitting !== null || !marginTask?.dependencies_satisfied || ["queued", "running"].includes(marginTask?.status ?? "")}>{marginTask?.status === "succeeded" ? "创建增量更新" : "创建下载任务"}</button>
          {!marginTask?.dependencies_satisfied ? <p className="form-warning">需先完成 Qlib 基线验收。</p> : null}
        </form>
      )}

      {createMode === "minute" && (
        <form className="task-form" onSubmit={startMinutes}>
          <div className="form-intro"><strong>核心资产 1 分钟线</strong><span>默认自动选择主要指数、ETF、股指期货、活跃期权和高流动性股票。</span></div>
          <div className="form-row"><label>开始日期<input type="date" value={start} onChange={(event) => setStart(event.target.value)} /></label><label>结束日期<input type="date" value={end} onChange={(event) => setEnd(event.target.value)} /><small>留空表示最新交易日</small></label></div>
          <label className="checkbox-row"><input type="checkbox" checked={autoSelect} onChange={(event) => setAutoSelect(event.target.checked)} />自动构建核心资产池</label>
          <details className="advanced-options">
            <summary>高级选项</summary>
            <div className="advanced-grid">
              <label>高流动性股票数量<input type="number" min="0" max="500" value={maxStocks} onChange={(event) => setMaxStocks(Number(event.target.value))} /></label>
              <label>活跃期权数量<input type="number" min="0" max="500" value={maxOptions} onChange={(event) => setMaxOptions(Number(event.target.value))} /></label>
              <div className="etf-category-row">{[["broad", "宽基"], ["industry", "行业"], ["gold", "黄金"], ["bond", "债券"]].map(([value, label]) => <label className="checkbox-row" key={value}><input type="checkbox" checked={etfCategories.includes(value)} onChange={() => toggleEtfCategory(value)} />{label} ETF</label>)}</div>
              <label>ETF<input value={etfs} onChange={(event) => setEtfs(event.target.value)} placeholder="510300.SH,159919.SZ" /></label>
              <label>重点股票<input value={stocks} onChange={(event) => setStocks(event.target.value)} placeholder="600519.SH" /></label>
              <label>指数<input value={indices} onChange={(event) => setIndices(event.target.value)} placeholder="000300.SH" /></label>
              <label>期货合约<input value={futures} onChange={(event) => setFutures(event.target.value)} placeholder="IF2607.CFX" /></label>
              <label>期权合约<input value={options} onChange={(event) => setOptions(event.target.value)} placeholder="合约代码" /></label>
            </div>
          </details>
          <button className="primary" disabled={submitting !== null || !minuteTask?.dependencies_satisfied || ["queued", "running"].includes(minuteTask?.status ?? "")}>{minuteTask?.status === "succeeded" ? "创建增量更新" : "创建下载任务"}</button>
          {!minuteTask?.dependencies_satisfied ? <p className="form-warning">需先完成期货、期权和同区间融券资格数据。</p> : null}
          <div className="form-intro"><strong>下载完成后构建分钟 Qlib</strong><span>转换是独立任务；失败只重试构建，不会重新下载分钟行情。</span></div>
          <label>分钟不可变快照<select value={minuteSnapshot} onChange={(event) => setMinuteSnapshot(event.target.value)}><option value="">尚无可用快照</option>{minuteSnapshots.map((item) => <option value={item.name} key={item.name}>{item.name} · {item.frequency}{item.end_date ? ` · 至 ${item.end_date}` : ""}</option>)}</select></label>
          <button type="button" onClick={startMinuteQlib} disabled={submitting !== null || !minuteSnapshot || !minuteQlibTask?.dependencies_satisfied || ["queued", "running"].includes(minuteQlibTask?.status ?? "")}>构建分钟 Qlib 数据集</button>
          <div className="form-intro"><strong>全 A 股 5 分钟线</strong><span>从 stock_basic 自动读取区间内曾上市的沪、深、北股票（含退市股），按股票和月份断点续传。</span></div>
          <button type="button" onClick={startAshare5m} disabled={submitting !== null || !ashare5mTask?.dependencies_satisfied || ["queued", "running"].includes(ashare5mTask?.status ?? "")}>{ashare5mTask?.status === "succeeded" ? "创建增量更新" : "创建全市场 5 分钟下载"}</button>
        </form>
      )}
    </section>
  );
}
