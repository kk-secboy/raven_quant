"use client";

import { FormEvent, useMemo, useState } from "react";
import { apiFetch } from "./api-client";
import { usePolling } from "./use-polling";

type Runtime = { status: string; qlib_version?: string; lightgbm_version?: string; execution_environment?: string };
type QlibDataset = { name: string; ready: boolean; trading_days: number; frequency: string; start_date?: string | null; end_date?: string | null };
type Experiment = {
  id: string; created_at: string; model: string; features: string;
  segments: Record<string, string[]>; metrics: Record<string, number | null>;
};
type Job = {
  id: string; kind: string; status: string;
  payload: Record<string, unknown>; error?: string | null;
};
type AutopilotCycle = { id: string; dataset: string; stage: string; status: string; updated_at: string };
type TournamentTrial = {
  id: string; name: string; trial_kind: string; status: string;
  feature_set_id?: string | null; model_family?: string | null;
  spec?: { round?: string; profiles?: string[]; seeds?: number[] };
  metrics?: Record<string, number | null> | null;
};
type ModelEnsemble = {
  id: string; name: string; status: string; combiner: string; dataset: string;
  components?: { model_family?: string; weight?: number }[];
};

const pct = (value: number | null | undefined) => value == null ? "—" : `${(value * 100).toFixed(2)}%`;
const decimal = (value: number | null | undefined) => value == null ? "—" : value.toFixed(3);
const compactVersion = (value?: string) => !value ? "—" : value.length > 24 ? `${value.slice(0, 21)}…` : value;

async function jsonResponse<T>(request: Promise<Response>): Promise<T> {
  const response = await request;
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export function QlibPanel({ api }: { api: string }) {
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const [runtimeLoadState, setRuntimeLoadState] = useState<"loading" | "ready" | "error">("loading");
  const [datasets, setDatasets] = useState<QlibDataset[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [tournamentCycle, setTournamentCycle] = useState<AutopilotCycle | null>(null);
  const [tournamentTrials, setTournamentTrials] = useState<TournamentTrial[]>([]);
  const [ensembles, setEnsembles] = useState<ModelEnsemble[]>([]);
  const [dataset, setDataset] = useState("");
  const [topk, setTopk] = useState(50);
  const [nDrop, setNDrop] = useState(5);
  const [account, setAccount] = useState(5_000_000);
  const [minuteDataset, setMinuteDataset] = useState("");
  const [minuteStart, setMinuteStart] = useState("2024-01-01");
  const [minuteEnd, setMinuteEnd] = useState(new Date().toISOString().slice(0, 10));
  const [minuteHorizons, setMinuteHorizons] = useState("5,15,30");
  const [minuteCostBps, setMinuteCostBps] = useState(2);
  const [message, setMessage] = useState("");

  async function load() {
    let nextDatasetCount: number | undefined;
    const requests = [
      jsonResponse<Runtime>(apiFetch(`${api}/api/qlib/status`, { cache: "no-store", forceRefresh: true })).then((nextRuntime) => {
        setRuntime(nextRuntime);
        setRuntimeLoadState("ready");
      }),
      jsonResponse<QlibDataset[]>(apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" })).then((nextDatasets) => {
        nextDatasetCount = nextDatasets.length;
        setDatasets(nextDatasets);
      const daily = nextDatasets.filter((item: QlibDataset) => item.frequency === "day");
      const minute = nextDatasets.filter((item: QlibDataset) => item.frequency === "1min");
      if (!dataset && daily.length) setDataset(daily[0].name);
      if (!minuteDataset && minute.length) {
        setMinuteDataset(minute[0].name);
        if (minute[0].start_date) setMinuteStart(String(minute[0].start_date).slice(0, 10));
        if (minute[0].end_date) setMinuteEnd(String(minute[0].end_date).slice(0, 10));
      }
      }),
      jsonResponse<Experiment[]>(apiFetch(`${api}/api/qlib/experiments`, { cache: "no-store" })).then(setExperiments),
      jsonResponse<Job[]>(apiFetch(`${api}/api/jobs?limit=20&kind=qlib_baseline&kind=minute_research`, { cache: "no-store" })).then(setJobs),
      jsonResponse<AutopilotCycle[]>(apiFetch(`${api}/api/autopilot/cycles?limit=1`, { cache: "no-store" })).then(async (cycles) => {
        const cycle = cycles[0] ?? null;
        setTournamentCycle(cycle);
        if (!cycle) { setTournamentTrials([]); return; }
        const response = await apiFetch(`${api}/api/autopilot/cycles/${cycle.id}/trials`, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        setTournamentTrials(await response.json() as TournamentTrial[]);
      }),
      jsonResponse<ModelEnsemble[]>(apiFetch(`${api}/api/model-ensembles?limit=20`, { cache: "no-store" })).then(setEnsembles),
    ];
    const results = await Promise.allSettled(requests);
    const failed = results.filter((item) => item.status === "rejected").length;
    if (results[0].status === "rejected") setRuntimeLoadState("error");
    if (failed === results.length) {
      setMessage("无法连接 Qlib 控制接口，请确认 Python 后端正在运行。 ");
    } else if (failed) {
      setMessage("部分实时状态暂未更新，已保留上次成功数据。 ");
    } else {
      setMessage(nextDatasetCount === 0 ? "还没有可训练的 Qlib 数据集，请先在数据中心完成 Core 初始化。 " : "");
    }
  }

  usePolling(load, 7000);

  async function runBaseline(event: FormEvent) {
    event.preventDefault();
    setMessage("正在创建 Qlib 基线实验…");
    const response = await apiFetch(`${api}/api/jobs/qlib-baseline`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset, market: "cn_all", benchmark: "SH000300", account,
        topk, n_drop: nDrop,
      }),
    });
    const body = await response.json();
    if (!response.ok) {
      setMessage(body.detail ?? "实验创建失败");
      return;
    }
    setMessage(`实验任务 ${body.id.slice(0, 8)} 已进入队列`);
    await load();
  }

  async function runMinuteResearch(event: FormEvent) {
    event.preventDefault();
    const horizons = minuteHorizons.split(",").map((item) => Number(item.trim())).filter(Number.isInteger);
    setMessage("正在创建分钟因子扫描任务…");
    const response = await apiFetch(`${api}/api/jobs/minute-research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: minuteDataset, start: minuteStart, end: minuteEnd, horizons, cost_rate: minuteCostBps / 10000 }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "分钟研究任务创建失败"); return; }
    setMessage(`分钟研究任务 ${body.id.slice(0, 8)} 已进入队列`);
    await load();
  }

  const latest = experiments[0];
  const active = useMemo(
    () => jobs.some((job) => job.kind === "qlib_baseline" && ["queued", "running"].includes(job.status)),
    [jobs],
  );
  const minuteActive = useMemo(() => jobs.some((job) => job.kind === "minute_research" && ["queued", "running"].includes(job.status)), [jobs]);
  const dailyDatasets = useMemo(() => datasets.filter((item) => item.frequency === "day"), [datasets]);
  const minuteDatasets = useMemo(() => datasets.filter((item) => item.frequency === "1min"), [datasets]);

  return <>
    {message && <div className="notice">{message}</div>}
    <section className="research-hero">
      <article className="runtime-card">
        <div className="card-heading"><div><span>研究运行时</span><strong>Qlib 多模型竞赛</strong></div><span className={`status-chip ${runtimeLoadState === "ready" && runtime?.status === "ok" ? "verified" : ""}`}>{runtimeLoadState === "loading" ? "检查中" : runtimeLoadState === "error" ? "读取失败" : runtime?.status === "ok" ? "已验证" : "不可用"}</span></div>
        <div className="runtime-grid"><div><span>Qlib</span><strong title={runtime?.qlib_version}>{compactVersion(runtime?.qlib_version)}</strong></div><div><span>候选模型</span><strong>Ridge / LGBM / GRU / Transformer</strong></div><div><span>受治理特征</span><strong>Alpha158 / 360 / 种子 / SOTA</strong></div><div><span>执行环境</span><strong>{runtimeLoadState === "loading" ? "检查中" : runtime?.execution_environment ?? "状态未知"}</strong></div></div>
        <div className="pipeline"><span>冻结数据</span><i>→</i><span>特征集</span><i>→</i><span>候选模型</span><i>→</i><span>选股策略</span><i>→</i><span>独立验证</span></div>
      </article>
      <form className="experiment-card" onSubmit={runBaseline}>
        <div className="card-heading"><div><span>高级手工对照</span><strong>Alpha158 + LightGBM</strong></div></div>
        <p className="muted">自动驾驶会自行完成多特征、多模型竞赛。这里只保留可复现的手工对照，不代表正式准入。</p>
        <label>Qlib 日频数据集<select value={dataset} onChange={(event) => setDataset(event.target.value)} disabled={!dailyDatasets.length}>{dailyDatasets.length ? dailyDatasets.map((item) => <option key={item.name} value={item.name}>{item.name} · {item.trading_days} 日</option>) : <option>无可用日频数据集</option>}</select></label>
        <div className="form-row"><label>初始资金<input type="number" min="100000" step="100000" value={account} onChange={(event) => setAccount(Number(event.target.value))} /></label><label>股票数量<input type="number" min="1" max="500" value={topk} onChange={(event) => setTopk(Number(event.target.value))} /></label></div>
        <div className="form-row"><label>每日替换<input type="number" min="0" max={topk} value={nDrop} onChange={(event) => setNDrop(Number(event.target.value))} /></label><label>基准<input value="沪深300 · SH000300" disabled /></label></div>
        <button className="primary" disabled={!dailyDatasets.length || runtime?.status !== "ok" || active}>运行 Alpha158 探索基线</button>
      </form>
    </section>
    <section className="data-panel tournament-panel">
      <div className="panel-heading"><div><p className="eyebrow">AUTOPILOT TOURNAMENT</p><h2>预注册模型试验</h2><p>全部尝试先登记后运行，失败项也会保留并计入多重检验；这里没有手工晋级按钮。</p></div><span>{tournamentCycle ? `${tournamentCycle.dataset} · ${tournamentCycle.stage}` : "等待自动周期"}</span></div>
      <div className="table-wrap"><table><thead><tr><th>试验</th><th>特征集</th><th>模型家族</th><th>阶段</th><th>窗口 / 种子</th><th>结论</th></tr></thead><tbody>{tournamentTrials.map((item) => <tr key={item.id}><td><code>{item.name}</code></td><td>{item.feature_set_id ?? "—"}</td><td>{item.model_family ?? "—"}</td><td>{item.spec?.round ?? item.trial_kind}</td><td>{item.spec?.profiles?.join(" / ") ?? "—"} · {item.spec?.seeds?.join(", ") ?? "—"}</td><td><span className={`state ${["selected", "passed"].includes(item.status) ? "ready" : ["failed", "rejected"].includes(item.status) ? "failed" : "partial"}`}>{item.status}</span></td></tr>)}</tbody></table>{!tournamentTrials.length ? <div className="empty">尚无预注册竞赛；新 AutopilotCycle 建立后会自动出现。</div> : null}</div>
    </section>
    <section className="data-panel tournament-panel">
      <div className="panel-heading"><div><p className="eyebrow">EQUAL-RANK ENSEMBLES</p><h2>模型集成候选</h2><p>只组合预测相关性不高于 0.90 的不同模型家族，最多三个成员并等权 Rank 融合，不做 Stacking。</p></div><span>{ensembles.length} 个候选</span></div>
      <div className="table-wrap"><table><thead><tr><th>集成</th><th>数据版本</th><th>模型家族</th><th>组合方式</th><th>状态</th></tr></thead><tbody>{ensembles.map((item) => <tr key={item.id}><td><code>{item.name}</code></td><td>{item.dataset}</td><td>{item.components?.map((component) => component.model_family).filter(Boolean).join(" + ") || "—"}</td><td>{item.combiner === "equal_rank" ? "等权日度 Rank" : item.combiner}</td><td><span className={`state ${item.status === "research_admitted" ? "ready" : item.status === "rejected" ? "failed" : "partial"}`}>{item.status}</span></td></tr>)}</tbody></table>{!ensembles.length ? <div className="empty compact">尚无满足跨家族和相关性约束的集成候选。</div> : null}</div>
    </section>
    <section className="data-panel minute-research-card">
      <div className="panel-heading"><div><p className="eyebrow">INTRADAY FACTOR LAB</p><h2>分钟因子扫描</h2><p>对动量、VWAP 偏离、量能、价格区间和实现波动做含成本横截面检验；结果只进入研究记录，不自动晋级策略。</p></div><span>{minuteDatasets.length} 个分钟数据集</span></div>
      <form className="task-form minute-research-form" onSubmit={runMinuteResearch}>
        <label>分钟 Qlib 数据集<select value={minuteDataset} onChange={(event) => setMinuteDataset(event.target.value)}>{minuteDatasets.length ? minuteDatasets.map((item) => <option key={item.name} value={item.name}>{item.name}</option>) : <option value="">尚无分钟数据集</option>}</select></label>
        <div className="form-row"><label>开始日期<input type="date" value={minuteStart} onChange={(event) => setMinuteStart(event.target.value)} /></label><label>结束日期<input type="date" value={minuteEnd} onChange={(event) => setMinuteEnd(event.target.value)} /></label></div>
        <div className="form-row"><label>预测周期（分钟，逗号分隔）<input value={minuteHorizons} onChange={(event) => setMinuteHorizons(event.target.value)} /></label><label>单边成本（bp）<input type="number" min="0" max="200" step="0.1" value={minuteCostBps} onChange={(event) => setMinuteCostBps(Number(event.target.value))} /></label></div>
        <button className="primary" disabled={!minuteDataset || runtime?.status !== "ok" || minuteActive}>运行分钟因子扫描</button>
      </form>
    </section>
    <section className="metric-strip">
      <div><span>IC</span><strong>{decimal(latest?.metrics.ic)}</strong></div><div><span>Rank IC</span><strong>{decimal(latest?.metrics.rank_ic)}</strong></div><div><span>含成本年化超额</span><strong>{pct(latest?.metrics.annualized_excess_return_with_cost)}</strong></div><div><span>信息比率</span><strong>{decimal(latest?.metrics.information_ratio)}</strong></div><div><span>最大回撤</span><strong>{pct(latest?.metrics.max_drawdown)}</strong></div><div><span>平均换手</span><strong>{pct(latest?.metrics.average_turnover)}</strong></div>
    </section>
    <section className="data-panel">
      <div className="panel-heading"><div><p className="eyebrow">REPRODUCIBLE EXPERIMENTS</p><h2>实验记录</h2></div><span>{experiments.length} 个已完成实验</span></div>
      <div className="table-wrap"><table><thead><tr><th>实验</th><th>测试区间</th><th>模型</th><th>IC</th><th>Rank IC</th><th>年化超额</th><th>最大回撤</th></tr></thead><tbody>{experiments.map((item) => <tr key={item.id}><td><code>{item.id.slice(0, 10)}</code></td><td>{item.segments.test?.join(" → ")}</td><td>{item.model} · {item.features}</td><td>{decimal(item.metrics.ic)}</td><td>{decimal(item.metrics.rank_ic)}</td><td>{pct(item.metrics.annualized_excess_return_with_cost)}</td><td>{pct(item.metrics.max_drawdown)}</td></tr>)}</tbody></table>{!experiments.length && <div className="empty">暂无实验。数据初始化完成后，可运行第一条 Alpha158 + LightGBM 基线。</div>}</div>
    </section>
    <section className="jobs-panel">
      <div className="panel-heading"><div><p className="eyebrow">QLIB JOBS</p><h2>研究任务</h2></div><span>{jobs.length} 条记录</span></div>
      <div className="job-list">{jobs.slice(0, 8).map((job) => <article key={job.id}><span className={`job-state ${job.status}`} /><div><strong>{job.kind === "minute_research" ? "分钟因子扫描" : "Alpha158 · LightGBM"}</strong><small>{String(job.payload.dataset ?? "")}</small></div><code>{job.id.slice(0, 10)}</code><span>{job.status}</span></article>)}{!jobs.length && <div className="empty compact">尚无 Qlib 研究任务。</div>}</div>
    </section>
  </>;
}
