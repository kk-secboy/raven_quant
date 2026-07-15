"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";

type Dataset = { name: string; ready: boolean; reproducible: boolean; start_date?: string | null; end_date?: string | null };
type Recipe = { id: string; name: string; rdagent_objective: string };
type ResearchProgram = {
  id: string;
  name: string;
  status: string;
  recipe_id: string;
  dataset_lineage_id: string;
  min_new_trading_days: number;
  max_active_campaigns: number;
  last_dataset_name?: string | null;
  last_dataset_end_date?: string | null;
  last_message?: string | null;
  last_checked_at?: string | null;
  champion_campaign_id?: string | null;
  champion_strategy_version_id?: string | null;
  champion_score?: number | null;
  decay_status?: "unavailable" | "healthy" | "warning";
  decay_message?: string | null;
};
type Campaign = {
  id: string;
  name: string;
  status: string;
  stage: string;
  dataset: string;
  error?: string | null;
  research_run_id?: string | null;
  strategy_version_id?: string | null;
  parameter_experiment_id?: string | null;
  backtest_id?: string | null;
  paper_portfolio_id?: string | null;
  research_program_id?: string | null;
  updated_at: string;
  state?: { champion?: { decision?: string; baseline_score?: number; challenger_score?: number } };
};

const STAGES = [
  "research",
  "factor_selection",
  "baseline_backtest",
  "parameter_experiment",
  "challenger_backtest",
  "strategy_approval",
  "paper_schedule",
  "complete",
];

const STAGE_LABELS: Record<string, string> = {
  research: "RD-Agent 研究",
  factor_selection: "因子排名与晋级",
  baseline_backtest: "基线回测",
  parameter_experiment: "参数实验",
  challenger_backtest: "挑战者回测",
  strategy_approval: "等待人工审批",
  paper_schedule: "创建模拟盘",
  complete: "自动流水线完成",
};

function today() {
  return new Date().toISOString().slice(0, 10);
}

export function ResearchCampaignPanel({ api }: { api: string }) {
  const [view, setView] = useState<"programs" | "campaigns">("programs");
  const [programs, setPrograms] = useState<ResearchProgram[]>([]);
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [recipes, setRecipes] = useState<Recipe[]>([]);
  const [name, setName] = useState("沪深300 自动研究");
  const [programName, setProgramName] = useState("沪深300 持续研究");
  const [minNewTradingDays, setMinNewTradingDays] = useState(20);
  const [championMinImprovement, setChampionMinImprovement] = useState(0.05);
  const [championDecayFraction, setChampionDecayFraction] = useState(0.25);
  const [dataset, setDataset] = useState("");
  const [recipeId, setRecipeId] = useState("index_enhancement");
  const [objective, setObjective] = useState("");
  const [periods, setPeriods] = useState({
    train_start: "2018-01-01", train_end: "2021-12-31",
    valid_start: "2022-01-01", valid_end: "2023-12-31",
    test_start: "2024-01-01", test_end: today(),
  });
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    const [programResponse, campaignResponse, datasetResponse, recipeResponse] = await Promise.all([
      apiFetch(`${api}/api/research-programs`, { cache: "no-store" }),
      apiFetch(`${api}/api/research-campaigns`, { cache: "no-store" }),
      apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" }),
      apiFetch(`${api}/api/strategy-recipes`, { cache: "no-store" }),
    ]);
    if (!programResponse.ok || !campaignResponse.ok || !datasetResponse.ok || !recipeResponse.ok) return;
    setPrograms(await programResponse.json());
    const nextCampaigns = await campaignResponse.json();
    const nextDatasets = (await datasetResponse.json()) as Dataset[];
    const recipePayload = await recipeResponse.json();
    setCampaigns(nextCampaigns);
    setDatasets(nextDatasets.filter((item) => item.ready && item.reproducible));
    setRecipes(recipePayload.recipes ?? []);
    if (!dataset && nextDatasets.length) {
      const usable = nextDatasets.find((item) => item.ready && item.reproducible);
      if (usable) setDataset(usable.name);
    }
  }, [api, dataset]);

  useEffect(() => {
    const initial = window.setTimeout(refresh, 0);
    const timer = window.setInterval(refresh, 5000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
  }, [refresh]);

  useEffect(() => {
    const recipe = recipes.find((item) => item.id === recipeId);
    if (recipe) setObjective(recipe.rdagent_objective);
  }, [recipeId, recipes]);

  const activeCount = useMemo(
    () => campaigns.filter((item) => ["queued", "running", "awaiting_approval"].includes(item.status)).length,
    [campaigns],
  );

  const activeProgramCount = useMemo(
    () => programs.filter((item) => item.status === "active").length,
    [programs],
  );

  async function createProgram(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setMessage("正在保存持续研究计划…");
    const response = await apiFetch(`${api}/api/research-programs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: programName,
        dataset,
        recipe_id: recipeId,
        objective,
        min_new_trading_days: minNewTradingDays,
        champion_min_score_improvement: championMinImprovement,
        champion_decay_fraction: championDecayFraction,
      }),
    });
    const body = await response.json();
    setBusy(false);
    if (!response.ok) {
      setMessage(typeof body.detail === "string" ? body.detail : body.detail?.message ?? "创建失败");
      return;
    }
    setMessage(`持续计划 ${body.id.slice(0, 8)} 已启用；新数据满足安全窗口后会自动研究。`);
    await refresh();
  }

  async function setProgramStatus(program: ResearchProgram, status: "active" | "paused" | "cancelled") {
    const response = await apiFetch(`${api}/api/research-programs/${program.id}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, actor: "web-operator" }),
    });
    const body = await response.json();
    setMessage(response.ok ? `持续计划已${status === "active" ? "恢复" : status === "paused" ? "暂停" : "取消"}。` : body.detail ?? "操作失败");
    await refresh();
  }

  async function checkProgram(program: ResearchProgram) {
    const response = await apiFetch(`${api}/api/research-programs/${program.id}/check-now`, { method: "POST" });
    const body = await response.json();
    setMessage(response.ok ? "已要求调度器立即检查最新 Qlib 数据。" : body.detail ?? "操作失败");
    await refresh();
  }

  async function createCampaign(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setMessage("正在创建自动研究流水线…");
    const response = await apiFetch(`${api}/api/research-campaigns`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, dataset, recipe_id: recipeId, objective, periods }),
    });
    const body = await response.json();
    setBusy(false);
    if (!response.ok) {
      setMessage(typeof body.detail === "string" ? body.detail : body.detail?.message ?? "创建失败");
      return;
    }
    setMessage(`流水线 ${body.id.slice(0, 8)} 已启动；关闭浏览器不会中断。`);
    await refresh();
  }

  async function setStatus(campaign: Campaign, status: "paused" | "running" | "cancelled") {
    const response = await apiFetch(`${api}/api/research-campaigns/${campaign.id}/status`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, actor: "web-operator" }),
    });
    const body = await response.json();
    setMessage(response.ok ? `流水线已${status === "paused" ? "暂停" : status === "running" ? "恢复" : "取消"}。` : body.detail ?? "操作失败");
    await refresh();
  }

  async function retry(campaign: Campaign) {
    const response = await apiFetch(`${api}/api/research-campaigns/${campaign.id}/retry`, { method: "POST" });
    const body = await response.json();
    setMessage(response.ok ? "失败阶段已按原始参数重新进入队列。" : body.detail ?? "重试失败");
    await refresh();
  }

  return <div className="campaign-page">
    <section className="readiness-hero campaign-hero">
      <div className="readiness-copy"><span className="status-chip">AUTONOMOUS RESEARCH</span><h2>自动研究流水线</h2><p>RD-Agent 提出因子，Qlib 独立评价，系统完成参数实验和冠军/挑战者回测；只有进入模拟盘前保留人工审批。</p></div>
      <div className="campaign-hero-count"><strong>{activeProgramCount}</strong><span>个持续计划 · {activeCount} 条流水线</span></div>
    </section>

    {message ? <div className="notice">{message}</div> : null}

    <div className="campaign-view-tabs">
      <button className={view === "programs" ? "active" : ""} onClick={() => setView("programs")}>持续研究计划</button>
      <button className={view === "campaigns" ? "active" : ""} onClick={() => setView("campaigns")}>单次研究与进度</button>
    </div>

    {view === "programs" ? <section className="campaign-layout">
      <form className="data-panel campaign-create" onSubmit={createProgram}>
        <div className="panel-heading"><div><p className="eyebrow">CONTINUOUS PROGRAM</p><h2>启用持续自动研究</h2><p>同一数据血缘新增足够交易日后，系统自动创建完整研究流水线。</p></div></div>
        <label>计划名称<input value={programName} onChange={(event) => setProgramName(event.target.value)} minLength={3} maxLength={100} /></label>
        <div className="form-row"><label>策略模板<select value={recipeId} onChange={(event) => setRecipeId(event.target.value)}>{recipes.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label><label>数据血缘起点<select value={dataset} onChange={(event) => setDataset(event.target.value)}>{datasets.map((item) => <option value={item.name} key={item.name}>{item.name}</option>)}</select></label></div>
        <label>RD-Agent 研究目标<textarea value={objective} onChange={(event) => setObjective(event.target.value)} minLength={10} maxLength={2000} rows={5} /></label>
        <label>累计新增交易日再研究<input type="number" min={1} max={252} value={minNewTradingDays} onChange={(event) => setMinNewTradingDays(Number(event.target.value))} /></label>
        <details className="campaign-advanced"><summary>安全时间窗口与冠军治理</summary><p>默认使用最近 756 个训练日、252 个验证日和 504 个独立测试日。数据不足时只等待，不会缩短样本窗口。</p><div className="form-row"><label>新冠军最低增分<input type="number" min="0" max="5" step="0.01" value={championMinImprovement} onChange={(event) => setChampionMinImprovement(Number(event.target.value))} /></label><label>衰减告警比例<input type="number" min="0.01" max="1" step="0.05" value={championDecayFraction} onChange={(event) => setChampionDecayFraction(Number(event.target.value))} /></label></div></details>
        <button className="primary" disabled={busy || !dataset || objective.length < 10}>{busy ? "正在保存…" : "启用持续研究"}</button>
        <small>计划只跟随已验证的同一 Qlib 血缘；每个新快照最多创建一次活动，模拟盘前仍需人工审批。</small>
      </form>

      <section className="data-panel campaign-list">
        <div className="panel-heading"><div><p className="eyebrow">RESEARCH SUPERVISOR</p><h2>持续计划</h2><p>调度器自动检查新快照，并限制同时运行的研究数量。</p></div><span>{programs.length} 个</span></div>
        <div className="campaign-cards">{programs.map((item) => <article className={`campaign-card ${item.status}`} key={item.id}>
          <div className="campaign-card-head"><div><strong>{item.name}</strong><small>{item.recipe_id} · {item.id.slice(0, 8)}</small></div><span className={`state ${item.status === "active" ? "ready" : "partial"}`}>{item.status === "active" ? "自动运行" : item.status === "paused" ? "已暂停" : "已取消"}</span></div>
          <div className="campaign-stage"><span>新增 {item.min_new_trading_days} 个交易日触发</span><b>并发 ≤ {item.max_active_campaigns}</b></div>
          {item.last_dataset_name ? <small>最近触发：{item.last_dataset_name} · {item.last_dataset_end_date}</small> : <small>尚未触发研究活动</small>}
          {item.champion_campaign_id ? <div className="campaign-stage"><span>跨轮次冠军 · {item.champion_campaign_id.slice(0, 8)}</span><b>{item.champion_score?.toFixed(4) ?? "—"}</b></div> : null}
          {item.decay_status && item.decay_status !== "unavailable" ? <p className={item.decay_status === "warning" ? "campaign-error" : "campaign-attention"}>{item.decay_status === "warning" ? "性能衰减告警" : "冠军表现稳定"}{item.decay_message ? ` · ${item.decay_message}` : ""}</p> : null}
          {item.last_message ? <p className="campaign-attention">{item.last_message}</p> : null}
          <div className="campaign-actions">
            {item.status === "active" ? <><button onClick={() => checkProgram(item)}>立即检查</button><button onClick={() => setProgramStatus(item, "paused")}>暂停</button></> : null}
            {item.status === "paused" ? <button className="primary" onClick={() => setProgramStatus(item, "active")}>恢复</button> : null}
            {item.status !== "cancelled" ? <button onClick={() => setProgramStatus(item, "cancelled")}>取消</button> : null}
          </div>
        </article>)}{!programs.length ? <div className="empty">尚未启用持续研究计划。</div> : null}</div>
      </section>
    </section> : <section className="campaign-layout">
      <form className="data-panel campaign-create" onSubmit={createCampaign}>
        <div className="panel-heading"><div><p className="eyebrow">NEW CAMPAIGN</p><h2>启动一次完整研究</h2><p>日常只需选择模板和数据集，高级参数已有安全默认值。</p></div></div>
        <label>研究名称<input value={name} onChange={(event) => setName(event.target.value)} minLength={3} maxLength={150} /></label>
        <div className="form-row"><label>策略模板<select value={recipeId} onChange={(event) => setRecipeId(event.target.value)}>{recipes.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label><label>Qlib 数据集<select value={dataset} onChange={(event) => setDataset(event.target.value)}>{datasets.map((item) => <option value={item.name} key={item.name}>{item.name}</option>)}</select></label></div>
        <label>RD-Agent 研究目标<textarea value={objective} onChange={(event) => setObjective(event.target.value)} minLength={10} maxLength={2000} rows={5} /></label>
        <details className="campaign-advanced"><summary>研究时间窗口</summary>
          <div className="form-row"><label>训练开始<input type="date" value={periods.train_start} onChange={(event) => setPeriods({ ...periods, train_start: event.target.value })} /></label><label>训练结束<input type="date" value={periods.train_end} onChange={(event) => setPeriods({ ...periods, train_end: event.target.value })} /></label></div>
          <div className="form-row"><label>验证开始<input type="date" value={periods.valid_start} onChange={(event) => setPeriods({ ...periods, valid_start: event.target.value })} /></label><label>验证结束<input type="date" value={periods.valid_end} onChange={(event) => setPeriods({ ...periods, valid_end: event.target.value })} /></label></div>
          <div className="form-row"><label>样本外开始<input type="date" value={periods.test_start} onChange={(event) => setPeriods({ ...periods, test_start: event.target.value })} /></label><label>样本外结束<input type="date" value={periods.test_end} onChange={(event) => setPeriods({ ...periods, test_end: event.target.value })} /></label></div>
        </details>
        <button className="primary" disabled={busy || !dataset || objective.length < 10}>{busy ? "正在创建…" : "启动自动研究"}</button>
        <small>自动晋级只适用于通过独立 Qlib 门禁的因子；策略进入模拟盘前仍需人工审批。</small>
      </form>

      <section className="data-panel campaign-list">
        <div className="panel-heading"><div><p className="eyebrow">DURABLE PIPELINES</p><h2>研究进度</h2><p>每一步均持久化，服务重启后会从当前阶段继续。</p></div><span>{campaigns.length} 条</span></div>
        <div className="campaign-cards">{campaigns.map((item) => {
          const stageIndex = Math.max(0, STAGES.indexOf(item.stage));
          const progress = Math.round((stageIndex / (STAGES.length - 1)) * 100);
          return <article className={`campaign-card ${item.status}`} key={item.id}>
            <div className="campaign-card-head"><div><strong>{item.name}</strong><small>{item.dataset} · {item.id.slice(0, 8)}</small></div><span className={`state ${item.status === "failed" ? "failed" : item.status === "succeeded" ? "ready" : "partial"}`}>{item.status === "awaiting_approval" ? "待审批" : item.status}</span></div>
            <div className="campaign-stage"><span>{STAGE_LABELS[item.stage] ?? item.stage}</span><b>{progress}%</b></div>
            <div className="mini-progress"><i style={{ width: `${progress}%` }} /></div>
            {item.state?.champion ? <small>冠军选择：{item.state.champion.decision === "challenger" ? "挑战者" : "基线"}</small> : null}
            {item.status === "awaiting_approval" ? <p className="campaign-attention">回测已完成，请到“策略回测”审批冠军版本；审批后系统会自动创建模拟盘和每日任务。</p> : null}
            {item.error ? <p className="campaign-error">{item.error}</p> : null}
            <div className="campaign-actions">
              {["queued", "running", "awaiting_approval"].includes(item.status) ? <button onClick={() => setStatus(item, "paused")}>暂停后续步骤</button> : null}
              {item.status === "paused" ? <button className="primary" onClick={() => setStatus(item, "running")}>继续</button> : null}
              {item.status === "failed" ? <button className="primary" onClick={() => retry(item)}>重试失败阶段</button> : null}
              {!["succeeded", "failed", "cancelled"].includes(item.status) ? <button onClick={() => setStatus(item, "cancelled")}>取消</button> : null}
            </div>
          </article>;
        })}{!campaigns.length ? <div className="empty">尚未创建自动研究流水线。</div> : null}</div>
      </section>
    </section>}
  </div>;
}
