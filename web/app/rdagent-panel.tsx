"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch } from "./api-client";

type Runtime = {
  status: string;
  ready?: boolean;
  version?: string;
  python?: string;
  docker_available?: boolean;
  qlib_data_ready?: boolean;
  llm_credentials_configured?: boolean;
  blockers?: string[];
  limits?: { max_loops: number; max_duration: string };
};

type Dataset = {
  name: string;
  path: string;
  ready: boolean;
  start_date: string | null;
  end_date: string | null;
  trading_days: number;
  instruments: number;
};

type ResearchRun = {
  id: string;
  objective: string;
  dataset: string;
  status: string;
  budget: { loop_n: number; duration: string };
  runtime?: { rounds?: number; candidates?: number } | null;
  created_at: string;
  error?: string | null;
};

type StrategyRecipe = {
  id: string; version: string; name: string; category: string; description: string;
  rdagent_objective: string; factor_guidance: string[]; config_overrides: Record<string, number>;
};

const statusText: Record<string, string> = {
  queued: "排队",
  running: "生成中",
  evaluating: "Qlib 评估中",
  succeeded: "完成",
  failed: "失败",
};

export function RDAgentPanel({ api }: { api: string }) {
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<ResearchRun[]>([]);
  const [recipes, setRecipes] = useState<StrategyRecipe[]>([]);
  const [recipeId, setRecipeId] = useState("index_enhancement");
  const [dataset, setDataset] = useState("");
  const [objective, setObjective] = useState("研究低换手、低拥挤度的质量因子，用于沪深300指数增强。");
  const [loopN, setLoopN] = useState(1);
  const [duration, setDuration] = useState("30m");
  const [periods, setPeriods] = useState({
    train_start: "2018-01-01", train_end: "2021-12-31",
    valid_start: "2022-01-01", valid_end: "2023-12-31",
    test_start: "2024-01-01", test_end: new Date().toISOString().slice(0, 10),
  });
  const [message, setMessage] = useState("正在核对 RD-Agent、Docker、LLM 与 Qlib 数据…");
  const recipeApplied = useRef(false);

  async function load() {
    try {
      const responses = await Promise.all([
        apiFetch(`${api}/api/rdagent/status`, { cache: "no-store" }),
        apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" }),
        apiFetch(`${api}/api/rdagent/runs`, { cache: "no-store" }),
        apiFetch(`${api}/api/strategy-recipes`, { cache: "no-store" }),
      ]);
      const nextRuntime = await responses[0].json();
      const nextDatasets = await responses[1].json();
      setRuntime(nextRuntime);
      setDatasets(nextDatasets);
      setRuns(await responses[2].json());
      const recipeBody: { recipes: StrategyRecipe[] } = await responses[3].json();
      setRecipes(recipeBody.recipes);
      if (!recipeApplied.current) {
        const recipe = recipeBody.recipes.find((item) => item.id === recipeId)
          ?? recipeBody.recipes[0];
        if (recipe) {
          setRecipeId(recipe.id);
          setObjective(recipe.rdagent_objective);
        }
        recipeApplied.current = true;
      }
      if (!dataset && nextDatasets.length) {
        setDataset(nextDatasets[0].name);
        if (nextDatasets[0].end_date) {
          setPeriods((current) => ({ ...current, test_end: nextDatasets[0].end_date }));
        }
      }
      setMessage("");
    } catch {
      setMessage("无法读取 RD-Agent 控制面，请确认 Python 后端正在运行。");
    }
  }

  useEffect(() => {
    const initial = window.setTimeout(load, 0);
    const timer = window.setInterval(load, 8000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedDataset = datasets.find((item) => item.name === dataset);
  const coverageReady = Boolean(
    selectedDataset?.start_date && selectedDataset?.end_date
    && selectedDataset.start_date <= periods.train_start
    && selectedDataset.end_date >= periods.test_end,
  );
  const active = useMemo(
    () => runs.some((item) => ["queued", "running", "evaluating"].includes(item.status)),
    [runs],
  );

  function selectDataset(value: string) {
    setDataset(value);
    const item = datasets.find((candidate) => candidate.name === value);
    if (item?.end_date) setPeriods((current) => ({ ...current, test_end: item.end_date! }));
  }

  function selectRecipe(value: string) {
    setRecipeId(value);
    const recipe = recipes.find((item) => item.id === value);
    if (recipe) setObjective(recipe.rdagent_objective);
  }

  async function startRun(event: FormEvent) {
    event.preventDefault();
    setMessage("正在创建受限 RD-Agent 研究任务…");
    const response = await apiFetch(`${api}/api/rdagent/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ objective, dataset, loop_n: loopN, duration, periods }),
    });
    const body = await response.json();
    if (!response.ok) {
      const detail = body.detail;
      setMessage(typeof detail === "string" ? detail : detail?.blockers?.join("；") ?? "任务创建失败");
      return;
    }
    setMessage(`研究任务 ${body.id.slice(0, 8)} 已进入队列；生成后会自动进入独立 Qlib 评估。`);
    await load();
  }

  const checks = [
    { label: "RD-Agent 运行时", pass: runtime?.status === "ok", value: runtime?.version ?? runtime?.status ?? "检查中" },
    { label: "Docker 沙箱", pass: runtime?.docker_available, value: runtime?.docker_available ? "可用" : "未就绪" },
    { label: "LLM 密钥", pass: runtime?.llm_credentials_configured, value: runtime?.llm_credentials_configured ? "已配置" : "未配置" },
    { label: "研究数据覆盖", pass: coverageReady, value: selectedDataset ? `${selectedDataset.start_date} → ${selectedDataset.end_date}` : "无数据集" },
  ];
  const selectedRecipe = recipes.find((item) => item.id === recipeId);

  return <>
    {message && <div className="notice">{message}</div>}
    <section className="agent-hero">
      <article className="runtime-card">
        <div className="card-heading"><div><span>受控研究运行时</span><strong>RD-Agent · Factor Loop</strong></div><span className={`status-chip ${runtime?.ready ? "verified" : ""}`}>{runtime?.ready ? "可运行" : "尚未就绪"}</span></div>
        <div className="preflight-list">{checks.map((item) => <div key={item.label}><i className={item.pass ? "pass" : "block"} /><span>{item.label}</span><strong>{item.value}</strong></div>)}</div>
        {runtime?.blockers?.length ? <div className="blocker-box"><b>启动前还需处理</b>{runtime.blockers.map((item) => <span key={item}>{item}</span>)}</div> : null}
        <div className="pipeline"><span>研究目标</span><i>→</i><span>因子代码</span><i>→</i><span>隔离执行</span><i>→</i><span>Qlib 样本外</span><i>→</i><span>人工晋级</span></div>
      </article>
      <form className="agent-form" onSubmit={startRun}>
        <div className="card-heading"><div><span>新建因子研究</span><strong>限定预算后启动</strong></div></div>
        <label>文档策略配方<select value={recipeId} onChange={(event) => selectRecipe(event.target.value)}><option value="">自定义研究目标</option>{recipes.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.version}</option>)}</select></label>
        {selectedRecipe && <div className="execution-note"><b>{selectedRecipe.name}</b><span>{selectedRecipe.description}</span>{selectedRecipe.factor_guidance.map((item) => <span key={item}>{item}</span>)}</div>}
        <label>研究目标<textarea value={objective} minLength={10} maxLength={2000} onChange={(event) => setObjective(event.target.value)} /></label>
        <label>Qlib 快照<select value={dataset} onChange={(event) => selectDataset(event.target.value)}>{datasets.length ? datasets.map((item) => <option key={item.name} value={item.name}>{item.name} · {item.instruments} 标的</option>) : <option value="">无可用数据集</option>}</select></label>
        <div className="form-row"><label>循环数<input type="number" min="1" max={runtime?.limits?.max_loops ?? 3} value={loopN} onChange={(event) => setLoopN(Number(event.target.value))} /></label><label>最长运行<select value={duration} onChange={(event) => setDuration(event.target.value)}><option value="30m">30 分钟</option><option value="1h">1 小时</option><option value="2h">2 小时</option></select></label></div>
        <button className="primary" disabled={!runtime?.ready || !coverageReady || active || objective.length < 10}>启动受控研究</button>
      </form>
    </section>

    <section className="period-panel">
      <div className="panel-heading"><div><p className="eyebrow">WALK-FORWARD BOUNDARIES</p><h2>训练 / 验证 / 样本外</h2></div><span className={coverageReady ? "coverage-ok" : "coverage-bad"}>{coverageReady ? "数据覆盖完整" : "当前数据不足"}</span></div>
      <div className="period-grid">{([
        ["训练", "train_start", "train_end"], ["验证", "valid_start", "valid_end"], ["样本外", "test_start", "test_end"],
      ] as const).map(([label, startKey, endKey]) => <div key={label}><strong>{label}</strong><label>开始<input type="date" value={periods[startKey]} onChange={(event) => setPeriods({ ...periods, [startKey]: event.target.value })} /></label><label>结束<input type="date" value={periods[endKey]} onChange={(event) => setPeriods({ ...periods, [endKey]: event.target.value })} /></label></div>)}</div>
      {!coverageReady && selectedDataset ? <p className="period-warning">所选快照覆盖 {selectedDataset.start_date} 至 {selectedDataset.end_date}，无法支撑上面的完整研究区间。仅下载 2024–2026 年数据可以做近期回测，但不足以完成默认训练与验证。</p> : null}
    </section>

    <section className="jobs-panel">
      <div className="panel-heading"><div><p className="eyebrow">RESEARCH RUNS</p><h2>研究运行</h2></div><span>{runs.length} 条记录</span></div>
      <div className="research-run-list">{runs.map((item) => <article key={item.id}><span className={`job-state ${item.status}`} /><div><strong>{item.objective}</strong><small>{item.dataset} · {item.budget.loop_n} 轮 · {item.budget.duration}</small></div><div className="run-count"><strong>{item.runtime?.candidates ?? 0}</strong><small>候选</small></div><code>{item.id.slice(0, 10)}</code><span>{statusText[item.status] ?? item.status}</span></article>)}{!runs.length && <div className="empty compact">尚无 RD-Agent 研究运行。系统不会在缺少数据、Docker 或 LLM 凭据时启动。</div>}</div>
    </section>
  </>;
}
