"use client";

import { FormEvent, useMemo, useRef, useState } from "react";
import { apiFetch } from "./api-client";
import { usePolling } from "./use-polling";

type ScenarioId =
  | "fin_factor"
  | "fin_model"
  | "fin_quant"
  | "fin_factor_report"
  | "general_model"
  | "data_science"
  | "llm_finetune";

type Runtime = {
  status: string;
  ready?: boolean;
  version?: string;
  docker_available?: boolean;
  llm_credentials_configured?: boolean;
  blockers?: string[];
  limits?: { max_loops: number; max_duration: string };
};

type Scenario = {
  id: ScenarioId;
  label: string;
  description: string;
  category: "quant" | "lab";
  ready: boolean;
  blockers: string[];
  requires_dataset: boolean;
  requires_assets: boolean;
  requires_feature_set?: boolean;
  asset_kind?: "pdf" | "dataset" | "finetune" | null;
  auto_select_assets?: boolean;
  asset_count?: { min: number; max: number };
  capital_eligible: boolean;
  gpu_required: boolean;
};

type Dataset = {
  name: string;
  ready: boolean;
  start_date: string | null;
  end_date: string | null;
  trading_days: number;
  instruments: number;
};

type ResearchArtifact = {
  id: string;
  artifact_type: string;
  contract_version: string;
  status: string;
  content_sha256: string;
  size_bytes: number;
  manifest_sha256: string;
  producer: string;
};

type ResearchAsset = {
  id: string;
  asset_key: string;
  asset_type: string;
  asset_kind?: string;
  media_type: string;
  publisher?: string | null;
  source_kind?: string | null;
  title?: string | null;
  content_sha256: string;
  manifest_sha256: string;
  status: string;
  consumption?: {
    status: string;
    scenario: string;
    research_run_id: string;
    reserved_at: string;
  } | null;
};

type FeatureSet = {
  id: string;
  name: string;
  feature_count: number;
  definition_sha256: string;
  contract_version: string;
};

type ResearchAssetAcquisition = {
  id: string;
  kind: string;
  status: string;
  mode: "automatic" | "manual_https" | string;
  attempts: number;
  max_attempts: number;
  created_at: string;
  error?: string | null;
  result?: {
    status?: string;
    published?: number;
    blocked?: number;
    failed?: number;
    tushare_selected?: number;
    arxiv_selected?: number;
  };
};

type ModelCandidate = {
  id: string;
  name: string;
  model_type: string;
  status: string;
  dataset: string;
  code_sha256: string;
  manifest_sha256: string;
  admission_evidence_sha256?: string | null;
};

type ModelEvaluation = {
  id: string;
  model_candidate_id: string;
  profile_id: string;
  seed: number;
  gate_status: string;
  gate_reasons: string[];
  candidate_manifest_sha256: string;
};

type QuantBundle = {
  id: string;
  name: string;
  status: string;
  model_candidate_id: string;
  factor_candidate_ids: string[];
  bundle_manifest_sha256: string;
};

type QuantEvaluation = {
  id: string;
  quant_bundle_candidate_id: string;
  ablation: string;
  profile_id: string;
  seed: number;
  gate_status: string;
  gate_reasons: string[];
  bundle_manifest_sha256: string;
};

type AssetLink = {
  id: string;
  asset_id: string;
  relationship: string;
  factor_candidate_id?: string | null;
  model_candidate_id?: string | null;
  quant_bundle_candidate_id?: string | null;
};

type ResearchRun = {
  id: string;
  kind?: string;
  scenario?: ScenarioId;
  objective: string;
  dataset?: string | null;
  status: string;
  budget: { loop_n?: number; duration?: string };
  runtime?: { rounds?: number; candidates?: number; model_candidates?: number; quant_bundles?: number; lab_status?: string } | null;
  created_at: string;
  error?: string | null;
};

type ResearchRunDetail = ResearchRun & {
  candidates: Array<{ id: string; name: string; status: string }>;
  assets: ResearchAsset[];
  asset_links: AssetLink[];
  asset_consumptions: Array<{
    id: string;
    asset_id: string;
    scenario: string;
    selection_mode: string;
    status: string;
    asset_manifest_sha256: string;
  }>;
  run_artifacts: ResearchArtifact[];
  model_candidates: ModelCandidate[];
  model_evaluations: ModelEvaluation[];
  quant_bundle_candidates: QuantBundle[];
  quant_bundle_evaluations: QuantEvaluation[];
};

type ResearchSchedule = {
  id: string;
  name: string;
  kind: string;
  status: string;
  desired_status: string;
  run_time: string;
  next_run_at: string;
  payload: Record<string, unknown>;
};

type StrategyRecipe = {
  id: string;
  version: string;
  name: string;
  category: string;
  description: string;
  rdagent_objective: string;
};

const fallbackScenarios: Scenario[] = [
  { id: "fin_factor", label: "因子研究", description: "自主提出、实现并迭代因子", category: "quant", ready: false, blockers: ["正在读取运行时状态"], requires_dataset: true, requires_assets: false, capital_eligible: true, gpu_required: false },
  { id: "fin_model", label: "模型研究", description: "固定受治理特征集，研究预测模型", category: "quant", ready: false, blockers: ["正在读取运行时状态"], requires_dataset: true, requires_assets: false, requires_feature_set: true, capital_eligible: true, gpu_required: false },
  { id: "fin_quant", label: "联合研究", description: "把因子集与预测模型作为完整组合迭代", category: "quant", ready: false, blockers: ["正在读取运行时状态"], requires_dataset: true, requires_assets: false, requires_feature_set: true, capital_eligible: true, gpu_required: false },
  { id: "fin_factor_report", label: "研报因子", description: "从已验证研报 PDF 中提取因子", category: "quant", ready: false, blockers: ["正在读取运行时状态"], requires_dataset: true, requires_assets: true, capital_eligible: true, gpu_required: false },
  { id: "general_model", label: "论文模型实现", description: "从 arXiv 或手工 PDF 实现模型代码", category: "lab", ready: false, blockers: ["正在读取运行时状态"], requires_dataset: false, requires_assets: true, capital_eligible: false, gpu_required: false },
  { id: "data_science", label: "数据科学", description: "隔离运行通用数据科学实验", category: "lab", ready: false, blockers: ["正在读取运行时状态"], requires_dataset: false, requires_assets: true, capital_eligible: false, gpu_required: false },
  { id: "llm_finetune", label: "大模型微调", description: "在独立 GPU 队列中训练和评估 LLM", category: "lab", ready: false, blockers: ["正在读取 GPU 能力"], requires_dataset: false, requires_assets: true, capital_eligible: false, gpu_required: true },
];

const statusText: Record<string, string> = {
  queued: "排队",
  running: "运行中",
  exporting: "归档制品",
  evaluating: "独立评估",
  blocked: "能力未满足",
  succeeded: "完成",
  failed: "失败",
};

function normalizeScenarios(value: unknown): Scenario[] {
  const body = value as { scenarios?: unknown };
  const rows = Array.isArray(value) ? value : Array.isArray(body?.scenarios) ? body.scenarios : [];
  if (!rows.length) return fallbackScenarios;
  return rows.map((row) => {
    const item = row as Partial<Scenario> & { scenario?: ScenarioId };
    const id = (item.id ?? item.scenario) as ScenarioId;
    const fallback = fallbackScenarios.find((entry) => entry.id === id) ?? fallbackScenarios[0];
    return { ...fallback, ...item, id, blockers: Array.isArray(item.blockers) ? item.blockers : [] };
  });
}

function shortHash(value?: string | null): string {
  return value ? `${value.slice(0, 10)}…` : "—";
}

function RunAudit({
  detail,
  validationTarget,
  onValidateGeneralModel,
}: {
  detail: ResearchRunDetail;
  validationTarget: string;
  onValidateGeneralModel: (artifactId: string) => void;
}) {
  const generalModelArtifacts = detail.run_artifacts?.filter((artifact) => artifact.artifact_type === "general_model_implementation_code" && artifact.status === "recorded") ?? [];
  return <div className="run-audit">
    <div className="audit-summary">
      {detail.runtime?.lab_status && <span>实验状态 {detail.runtime.lab_status}</span>}
      <span>资料 {detail.assets?.length ?? 0}</span>
      <span>自动消费 {detail.asset_consumptions?.length ?? 0}</span>
      <span>制品 {detail.run_artifacts?.length ?? 0}</span>
      <span>模型评估 {detail.model_evaluations?.length ?? 0}</span>
      <span>联合评估 {detail.quant_bundle_evaluations?.length ?? 0}</span>
    </div>

    {!!detail.assets?.length && <section><h4>输入资料与来源关系</h4>{detail.assets.map((asset) => <div className="audit-row" key={asset.id}>
      <b>{asset.asset_type}</b><span>{asset.asset_key}</span><code>{shortHash(asset.manifest_sha256)}</code><em>{asset.status}</em>
    </div>)}{detail.asset_links?.map((link) => <small key={link.id}>{link.asset_id} · {link.relationship}</small>)}{detail.asset_consumptions?.map((item) => <small key={item.id}>{item.asset_id} · {item.selection_mode} · {item.status} · {shortHash(item.asset_manifest_sha256)}</small>)}</section>}

    {!!detail.run_artifacts?.length && <section><h4>不可变研究制品</h4>{detail.run_artifacts.map((artifact) => <div className="audit-row" key={artifact.id}>
      <b>{artifact.artifact_type}</b><span>{artifact.producer} · {artifact.size_bytes.toLocaleString("zh-CN")} B</span><code>{shortHash(artifact.content_sha256)}</code><em>{artifact.status}</em>
    </div>)}</section>}

    {detail.scenario === "general_model" && detail.status === "succeeded" && !!generalModelArtifacts.length && <section><h4>转入独立模型验证</h4><small>将论文实现绑定到 {validationTarget}；这里只创建研究候选，不会直接进入策略或模拟盘。</small>{generalModelArtifacts.map((artifact) => <div className="audit-row" key={`validate-${artifact.id}`}>
      <b>{artifact.artifact_type}</b><span>先检查 Qlib 接口，再执行 3 窗口 × 3 随机种子</span><code>{shortHash(artifact.content_sha256)}</code><button className="inline-action" type="button" onClick={() => onValidateGeneralModel(artifact.id)}>提交独立验证</button>
    </div>)}</section>}

    {!!detail.model_candidates?.length && <section><h4>模型候选与独立评估</h4>{detail.model_candidates.map((candidate) => <div className="audit-row" key={candidate.id}>
      <b>{candidate.name}</b><span>{candidate.model_type} · {candidate.dataset}</span><code>{shortHash(candidate.manifest_sha256)}</code><em>{candidate.status}</em>
    </div>)}{detail.model_evaluations?.map((evaluation) => <div className="audit-row subordinate" key={evaluation.id}>
      <b>{evaluation.profile_id} / seed {evaluation.seed}</b><span>{evaluation.gate_reasons?.join("；") || "独立证据完整"}</span><code>{shortHash(evaluation.candidate_manifest_sha256)}</code><em>{evaluation.gate_status}</em>
    </div>)}</section>}

    {!!detail.quant_bundle_candidates?.length && <section><h4>因子 + 模型联合候选与 27 格评估</h4>{detail.quant_bundle_candidates.map((bundle) => <div className="audit-row" key={bundle.id}>
      <b>{bundle.name}</b><span>{bundle.factor_candidate_ids.length} 个因子</span><code>{shortHash(bundle.bundle_manifest_sha256)}</code><em>{bundle.status}</em>
    </div>)}{detail.quant_bundle_evaluations?.map((evaluation) => <div className="audit-row subordinate" key={evaluation.id}>
      <b>{evaluation.ablation} · {evaluation.profile_id} · seed {evaluation.seed}</b><span>{evaluation.gate_reasons?.join("；") || "独立证据完整"}</span><code>{shortHash(evaluation.bundle_manifest_sha256)}</code><em>{evaluation.gate_status}</em>
    </div>)}</section>}
  </div>;
}

export function RDAgentPanel({ api }: { api: string }) {
  const [runtime, setRuntime] = useState<Runtime | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>(fallbackScenarios);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<ResearchRun[]>([]);
  const [schedules, setSchedules] = useState<ResearchSchedule[]>([]);
  const [recipes, setRecipes] = useState<StrategyRecipe[]>([]);
  const [featureSets, setFeatureSets] = useState<FeatureSet[]>([]);
  const [researchAssets, setResearchAssets] = useState<ResearchAsset[]>([]);
  const [assetAcquisitions, setAssetAcquisitions] = useState<ResearchAssetAcquisition[]>([]);
  const [scenarioId, setScenarioId] = useState<ScenarioId>("fin_factor");
  const [recipeId, setRecipeId] = useState("index_enhancement");
  const [dataset, setDataset] = useState("");
  const [featureSetId, setFeatureSetId] = useState("governed-baseline");
  const [assetIds, setAssetIds] = useState("");
  const [manualAssetUrl, setManualAssetUrl] = useState("");
  const [manualAssetTitle, setManualAssetTitle] = useState("");
  const [manualAssetKind, setManualAssetKind] = useState<"paper" | "research_report">("paper");
  const [assetRequestPending, setAssetRequestPending] = useState(false);
  const [objective, setObjective] = useState("研究低换手、低拥挤度的稳健信号，并用独立 Qlib 门禁验证。");
  const [loopN, setLoopN] = useState(1);
  const [duration, setDuration] = useState("30m");
  const [scheduleName, setScheduleName] = useState("每日受控 RD-Agent 研究");
  const [scheduleTime, setScheduleTime] = useState("20:30");
  const [message, setMessage] = useState("正在核对各场景运行能力…");
  const [openRunId, setOpenRunId] = useState<string | null>(null);
  const [runDetail, setRunDetail] = useState<ResearchRunDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const recipeApplied = useRef(false);

  async function load() {
    try {
      const responses = await Promise.all([
        apiFetch(`${api}/api/rdagent/status`, { cache: "no-store" }),
        apiFetch(`${api}/api/rdagent/scenarios`, { cache: "no-store" }),
        apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" }),
        apiFetch(`${api}/api/rdagent/runs`, { cache: "no-store" }),
        apiFetch(`${api}/api/strategy-recipes`, { cache: "no-store" }),
        apiFetch(`${api}/api/schedules`, { cache: "no-store" }),
        apiFetch(`${api}/api/rdagent/feature-sets`, { cache: "no-store" }),
        apiFetch(`${api}/api/rdagent/assets`, { cache: "no-store" }),
        apiFetch(`${api}/api/rdagent/assets/acquisitions`, { cache: "no-store" }),
      ]);
      const nextRuntime = await responses[0].json();
      const nextScenarios = responses[1].ok ? normalizeScenarios(await responses[1].json()) : fallbackScenarios;
      const nextDatasets: Dataset[] = await responses[2].json();
      const recipeBody: { recipes: StrategyRecipe[] } = await responses[4].json();
      const nextFeatureSets: FeatureSet[] = responses[6].ok ? await responses[6].json() : [];
      const nextResearchAssets: ResearchAsset[] = responses[7].ok ? await responses[7].json() : [];
      const nextAssetAcquisitions: ResearchAssetAcquisition[] = responses[8].ok ? await responses[8].json() : [];
      setRuntime(nextRuntime);
      setScenarios(nextScenarios);
      setDatasets(nextDatasets);
      setRuns(await responses[3].json());
      setRecipes(recipeBody.recipes);
      setSchedules((await responses[5].json()).filter((item: ResearchSchedule) => item.kind === "rdagent_research"));
      setFeatureSets(nextFeatureSets);
      setResearchAssets(nextResearchAssets);
      setAssetAcquisitions(nextAssetAcquisitions);
      if (!recipeApplied.current) {
        const recipe = recipeBody.recipes.find((item) => item.id === recipeId) ?? recipeBody.recipes[0];
        if (recipe) { setRecipeId(recipe.id); setObjective(recipe.rdagent_objective); }
        recipeApplied.current = true;
      }
      if (!dataset && nextDatasets.length) setDataset(nextDatasets[0].name);
      if (nextFeatureSets.length && !nextFeatureSets.some((item) => item.id === featureSetId)) {
        setFeatureSetId(nextFeatureSets[0].id);
      }
      setMessage("");
    } catch {
      setMessage("无法读取 RD-Agent 研究中心，请确认 QuantLab API 正在运行。");
    }
  }

  usePolling(load, 8000);

  const selectedScenario = scenarios.find((item) => item.id === scenarioId) ?? fallbackScenarios[0];
  const selectedDataset = datasets.find((item) => item.name === dataset);
  const coverageReady = Boolean(!selectedScenario.requires_dataset || (selectedDataset?.ready && selectedDataset.start_date && selectedDataset.end_date && selectedDataset.trading_days >= 3029));
  const activeScenario = useMemo(
    () => runs.some((item) => (item.scenario ?? item.kind) === scenarioId && ["queued", "running", "exporting", "evaluating"].includes(item.status)),
    [runs, scenarioId],
  );
  const selectedRecipe = recipes.find((item) => item.id === recipeId);
  const parsedAssetIds = assetIds.split(/[\s,]+/).map((item) => item.trim()).filter(Boolean);
  const explicitAssetsRequired = scenarioId === "data_science" || scenarioId === "llm_finetune";
  const selectableAssets = researchAssets.filter((item) => {
    if (item.status !== "registered" || item.asset_kind !== selectedScenario.asset_kind) return false;
    if (scenarioId === "fin_factor_report") return item.asset_type === "research_report";
    if (scenarioId === "general_model") return ["arxiv_paper", "manual_paper"].includes(item.asset_type);
    return true;
  });
  const featureSetReady = !selectedScenario.requires_feature_set || featureSets.some((item) => item.id === featureSetId);

  function selectRecipe(value: string) {
    setRecipeId(value);
    const recipe = recipes.find((item) => item.id === value);
    if (recipe) setObjective(recipe.rdagent_objective);
  }

  async function startRun(event: FormEvent) {
    event.preventDefault();
    setMessage(`正在创建 ${selectedScenario.label} 任务…`);
    const response = await apiFetch(`${api}/api/rdagent/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scenario: scenarioId,
        objective,
        dataset: selectedScenario.requires_dataset ? dataset : undefined,
        loop_n: loopN,
        duration,
        asset_ids: parsedAssetIds,
        feature_set_id: selectedScenario.requires_feature_set ? featureSetId : undefined,
      }),
    });
    const body = await response.json();
    if (!response.ok) {
      const detail = body.detail;
      setMessage(typeof detail === "string" ? detail : detail?.blockers?.join("；") ?? detail?.message ?? "任务创建失败");
      return;
    }
    setMessage(`任务 ${body.id.slice(0, 8)} 已进入受控队列；官方结果不会直接进入模拟盘。`);
    await load();
  }

  async function createResearchSchedule(event: FormEvent) {
    event.preventDefault();
    const response = await apiFetch(`${api}/api/schedules`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: scheduleName,
        kind: "rdagent_research",
        timezone: "Asia/Shanghai",
        run_time: scheduleTime,
        trading_days_only: selectedScenario.category === "quant",
        payload: {
          scenario: scenarioId,
          objective,
          dataset: selectedScenario.requires_dataset ? dataset : undefined,
          loop_n: loopN,
          duration,
          asset_ids: parsedAssetIds,
          feature_set_id: selectedScenario.requires_feature_set ? featureSetId : undefined,
          requested_by: "research-scheduler",
        },
        misfire_grace_seconds: 1800,
        actor: "local-operator",
      }),
    });
    const body = await response.json();
    setMessage(response.ok ? `自动研究计划 ${body.name} 已启用。` : body.detail?.message ?? body.detail ?? "计划保存失败");
    if (response.ok) await load();
  }

  async function toggleResearchSchedule(item: ResearchSchedule) {
    const status = item.desired_status === "active" ? "paused" : "active";
    const response = await apiFetch(`${api}/api/schedules/${item.id}/status`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status }),
    });
    setMessage(response.ok ? (status === "paused" ? "自动研究计划已暂停。" : "自动研究计划已恢复。") : "计划状态更新失败。");
    if (response.ok) await load();
  }

  async function runHealthCheck() {
    setMessage("正在执行受限诊断；诊断不会代替生产 readiness。");
    const response = await apiFetch(`${api}/api/rdagent/health-check`, { method: "POST" });
    const body = await response.json();
    setMessage(response.ok ? `诊断完成：${body.status ?? "ok"}` : body.detail?.message ?? body.detail ?? "诊断失败");
    await load();
  }

  async function acquireAutomaticAssets(event: FormEvent) {
    event.preventDefault();
    setAssetRequestPending(true);
    setMessage("正在把今日 Tushare 研报与 arXiv 论文采集任务放入受控队列…");
    try {
      const response = await apiFetch(`${api}/api/rdagent/assets/acquisitions/automatic`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ include_tushare: true, include_arxiv: true }),
      });
      const body = await response.json();
      setMessage(response.ok ? `资料采集任务 ${String(body.id).slice(0, 8)} 已入队；每日最多 20 份研报和 3 篇论文。` : body.detail ?? "资料采集入队失败");
      if (response.ok) await load();
    } finally {
      setAssetRequestPending(false);
    }
  }

  async function acquireManualAsset(event: FormEvent) {
    event.preventDefault();
    setAssetRequestPending(true);
    setMessage("正在登记安全 HTTPS PDF 采集任务…");
    try {
      const response = await apiFetch(`${api}/api/rdagent/assets/acquisitions/manual-https`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: manualAssetUrl,
          title: manualAssetTitle,
          document_kind: manualAssetKind,
        }),
      });
      const body = await response.json();
      setMessage(response.ok ? `手工资料任务 ${String(body.id).slice(0, 8)} 已入队；服务端会校验 HTTPS、DNS、PDF 和哈希。` : body.detail ?? "手工资料入队失败");
      if (response.ok) {
        setManualAssetUrl("");
        setManualAssetTitle("");
        await load();
      }
    } finally {
      setAssetRequestPending(false);
    }
  }

  async function toggleRunDetail(runId: string) {
    if (openRunId === runId) { setOpenRunId(null); setRunDetail(null); return; }
    setOpenRunId(runId);
    setRunDetail(null);
    setDetailLoading(true);
    try {
      const response = await apiFetch(`${api}/api/rdagent/runs/${runId}`, { cache: "no-store" });
      if (!response.ok) throw new Error("读取运行详情失败");
      setRunDetail(await response.json());
    } catch {
      setMessage("无法读取该运行的安全审计详情。");
    } finally {
      setDetailLoading(false);
    }
  }

  async function queueGeneralModelValidation(artifactId: string) {
    if (!openRunId || !dataset || !featureSetId) {
      setMessage("请先在研究表单中选择用于论文模型验证的 Qlib 数据集和冻结特征集。");
      return;
    }
    setMessage("正在把论文模型实现送入 fin_model 独立门禁…");
    const response = await apiFetch(`${api}/api/rdagent/runs/${openRunId}/model-validation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ artifact_id: artifactId, dataset, feature_set_id: featureSetId }),
    });
    const body = await response.json();
    setMessage(response.ok ? `模型候选 ${String(body.model_candidate_id).slice(0, 8)} 已进入独立验证；最终 OOS 仍保持密封。` : body.detail ?? "论文模型验证入队失败");
    if (response.ok && openRunId) {
      const detailResponse = await apiFetch(`${api}/api/rdagent/runs/${openRunId}`, { cache: "no-store" });
      if (detailResponse.ok) setRunDetail(await detailResponse.json());
      await load();
    }
  }

  const submitBlocked = !selectedScenario.ready || !coverageReady || !featureSetReady || activeScenario || objective.length < 10 || (explicitAssetsRequired && parsedAssetIds.length === 0);

  return <>
    {message && <div className="notice">{message}</div>}

    <section className="scenario-panel">
      <div className="panel-heading"><div><p className="eyebrow">RD-AGENT RESEARCH CENTER</p><h2>选择研究场景</h2></div><button className="inline-action" type="button" onClick={runHealthCheck}>运行环境诊断</button></div>
      <div className="scenario-grid">{scenarios.map((item) => <button type="button" key={item.id} className={`scenario-card ${scenarioId === item.id ? "selected" : ""}`} onClick={() => { setScenarioId(item.id); setAssetIds(""); }}>
        <span>{item.category === "quant" ? "量化主线" : "研究实验室"}</span><strong>{item.label}</strong><small>{item.description}</small><em className={item.ready ? "ready" : "blocked"}>{item.ready ? "可运行" : "能力未满足"}</em>
      </button>)}</div>
    </section>

    <details className="research-automation">
      <summary><span>研究资料采集与台账</span><strong>{researchAssets.length} 份已登记 · {assetAcquisitions.filter((item) => ["queued", "running"].includes(item.status)).length} 个采集中</strong></summary>
      <div className="research-automation-body">
        <form onSubmit={acquireAutomaticAssets}>
          <p>服务端按上海日期筛选 Tushare research_report 与 arXiv，绑定已验证数据快照并去重；客户端不能传路径、命令或环境变量。</p>
          <button className="secondary-action" disabled={assetRequestPending}>采集今日研报与论文</button>
        </form>
        <form onSubmit={acquireManualAsset}>
          <p>手工资料仅接受公开 HTTPS PDF；下载器会阻断私网地址、危险重定向和非 PDF 内容。</p>
          <label>资料类型<select value={manualAssetKind} onChange={(event) => setManualAssetKind(event.target.value as "paper" | "research_report")}><option value="paper">论文 / 模型报告</option><option value="research_report">券商研报</option></select></label>
          <label>标题<input value={manualAssetTitle} minLength={3} maxLength={500} required onChange={(event) => setManualAssetTitle(event.target.value)} /></label>
          <label>公开 HTTPS PDF<input type="url" value={manualAssetUrl} pattern="https://.*" required placeholder="https://example.org/document.pdf" onChange={(event) => setManualAssetUrl(event.target.value)} /></label>
          <button className="secondary-action" disabled={assetRequestPending}>安全采集手工 PDF</button>
        </form>
        <div className="research-schedule-list research-asset-history">
          {researchAssets.slice(0, 6).map((item) => <article key={`asset-${item.id}`}><div><strong>{item.title || item.asset_key}</strong><small>{item.asset_type} · {item.source_kind || item.publisher || "受治理来源"}{item.consumption ? ` · 已用于 ${item.consumption.scenario}` : " · 尚未消费"}</small></div><span className={`state ${item.status === "registered" ? "ready" : "blocked"}`}>{item.status}</span><code>{shortHash(item.manifest_sha256)}</code></article>)}
          {assetAcquisitions.slice(0, 8).map((item) => <article key={item.id}><div><strong>{item.mode === "automatic" ? "自动 Tushare + arXiv" : "手工 HTTPS PDF"}</strong><small>{new Date(item.created_at).toLocaleString("zh-CN", { hour12: false })} · 已发布 {item.result?.published ?? 0} · 阻断 {item.result?.blocked ?? 0}</small></div><span className={`state ${item.status === "succeeded" ? "ready" : item.status === "failed" ? "blocked" : "partial"}`}>{statusText[item.status] ?? item.status}</span><code>{item.id.slice(0, 8)}</code></article>)}
          {!assetAcquisitions.length && <div className="empty compact">尚无资料采集任务。</div>}
        </div>
      </div>
    </details>

    <section className="agent-hero">
      <article className="runtime-card">
        <div className="card-heading"><div><span>{selectedScenario.id}</span><strong>{selectedScenario.label}</strong></div><span className={`status-chip ${selectedScenario.ready ? "verified" : ""}`}>{selectedScenario.ready ? "可运行" : "已阻断"}</span></div>
        <div className="preflight-list">
          <div><i className={runtime?.status === "ok" ? "pass" : "block"} /><span>RD-Agent</span><strong>{runtime?.version ?? runtime?.status ?? "检查中"}</strong></div>
          <div><i className={runtime?.docker_available ? "pass" : "block"} /><span>隔离执行</span><strong>{runtime?.docker_available ? "Docker 可用" : "Docker 不可用"}</strong></div>
          <div><i className={runtime?.llm_credentials_configured ? "pass" : "block"} /><span>LLM 凭据</span><strong>{runtime?.llm_credentials_configured ? "已配置" : "未配置"}</strong></div>
          <div><i className={coverageReady ? "pass" : "block"} /><span>输入契约</span><strong>{coverageReady ? "满足" : "数据覆盖不足"}</strong></div>
        </div>
        {selectedScenario.blockers.length ? <div className="blocker-box"><b>当前阻断</b>{selectedScenario.blockers.map((item) => <span key={item}>{item}</span>)}</div> : null}
        <div className="pipeline"><span>受控输入</span><i>→</i><span>RD-Agent 实验</span><i>→</i><span>不可变制品</span><i>→</i><span>独立 Qlib 门禁</span><i>→</i><span>{selectedScenario.capital_eligible ? "人工批准 / 模拟盘" : "实验室归档"}</span></div>
      </article>

      <form className="agent-form" onSubmit={startRun}>
        <div className="card-heading"><div><span>新建受控研究</span><strong>{selectedScenario.label}</strong></div></div>
        {selectedScenario.category === "quant" && <label>策略研究配方<select value={recipeId} onChange={(event) => selectRecipe(event.target.value)}><option value="">自定义目标</option>{recipes.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.version}</option>)}</select></label>}
        {selectedRecipe && selectedScenario.category === "quant" && <div className="execution-note"><b>{selectedRecipe.name}</b><span>{selectedRecipe.description}</span></div>}
        <label>研究目标<textarea value={objective} minLength={10} maxLength={2000} onChange={(event) => setObjective(event.target.value)} /></label>
        {selectedScenario.requires_dataset && <label>Qlib 数据集<select value={dataset} onChange={(event) => setDataset(event.target.value)}>{datasets.map((item) => <option key={item.name} value={item.name}>{item.name} · {item.instruments} 标的 · {item.trading_days} 日</option>)}</select></label>}
        {selectedScenario.requires_feature_set && <label>冻结特征集<select value={featureSetId} disabled={!featureSets.length} onChange={(event) => setFeatureSetId(event.target.value)}>{!featureSets.length && <option value="">未加载到受治理特征集</option>}{featureSets.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.feature_count} 特征 · {shortHash(item.definition_sha256)}</option>)}</select></label>}
        {selectedScenario.requires_assets && <label>受治理资料（可多选；研报和论文不选时由服务端自动挑选未消费资料）<select multiple value={parsedAssetIds} onChange={(event) => setAssetIds(Array.from(event.currentTarget.selectedOptions, (option) => option.value).join(","))}>{selectableAssets.map((item) => <option key={item.id} value={item.id}>{item.title || item.asset_key} · {item.asset_type}{item.consumption ? ` · 已用于 ${item.consumption.scenario}` : ""}</option>)}</select></label>}
        {selectedScenario.requires_assets && parsedAssetIds.length > 0 && selectedScenario.auto_select_assets && <button className="inline-action" type="button" onClick={() => setAssetIds("")}>清空选择，改用自动筛选</button>}
        {selectedScenario.requires_assets && !selectableAssets.length && <small className="period-warning">当前没有匹配该场景的已登记资料，请先在“研究资料采集与台账”中采集。</small>}
        {explicitAssetsRequired && parsedAssetIds.length === 0 && <small className="period-warning">该高风险场景必须显式填写受治理资料 ID。</small>}
        <div className="form-row"><label>循环数<input type="number" min="1" max={runtime?.limits?.max_loops ?? 3} value={loopN} onChange={(event) => setLoopN(Number(event.target.value))} /></label><label>最长运行<select value={duration} onChange={(event) => setDuration(event.target.value)}><option value="30m">30 分钟</option><option value="1h">1 小时</option><option value="2h">2 小时</option></select></label></div>
        <button className="primary" disabled={submitBlocked}>启动受控研究</button>
      </form>
    </section>

    <section className="period-panel">
      <div className="panel-heading"><div><p className="eyebrow">GOVERNANCE BOUNDARY</p><h2>{selectedScenario.capital_eligible ? "独立验证与资本边界" : "实验室隔离边界"}</h2></div><span className={selectedScenario.capital_eligible ? "coverage-ok" : "coverage-bad"}>{selectedScenario.capital_eligible ? "可进入独立门禁" : "禁止直接进入投资链"}</span></div>
      {selectedScenario.capital_eligible ? <><div className="period-grid"><div><strong>近期窗口</strong><span>捕捉当前市场结构</span></div><div><strong>均衡窗口</strong><span>验证跨阶段稳定性</span></div><div><strong>稳健窗口</strong><span>长期压力验证</span></div></div><p className="period-warning">历史治理起始日 2021-01-11。RD-Agent 内部成绩仅供研究反馈；最终 OOS 只开放一次，正式回测后仍需人工批准和模拟盘。</p></> : <p className="period-warning">该场景的代码、评分或 checkpoint 只归档为实验制品。general_model 的实现可在运行审计中显式提交到 fin_model 独立门禁；其他实验室产物不能进入投资链。</p>}
    </section>

    <details className="research-automation">
      <summary><span>自动研究计划</span><strong>{schedules.filter((item) => item.status === "active").length} 个运行中</strong></summary>
      <div className="research-automation-body">
        <form onSubmit={createResearchSchedule}><p>按当前场景和输入契约定时入队；缺少权限、GPU、PDF 或数据时安全阻断。</p><label>计划名称<input value={scheduleName} minLength={3} maxLength={150} onChange={(event) => setScheduleName(event.target.value)} /></label><label>运行时间<input type="time" value={scheduleTime} onChange={(event) => setScheduleTime(event.target.value)} /></label><button className="secondary-action" disabled={submitBlocked}>保存并启用</button></form>
        <div className="research-schedule-list">{schedules.map((item) => <article key={item.id}><div><strong>{item.name}</strong><small>{String(item.payload.scenario ?? "fin_factor")} · {item.run_time} · 下次 {new Date(item.next_run_at).toLocaleString("zh-CN", { hour12: false })}</small></div><span className={`state ${item.status === "active" ? "ready" : "partial"}`}>{item.status}</span><button className="inline-action" type="button" onClick={() => toggleResearchSchedule(item)}>{item.desired_status === "active" ? "暂停" : "恢复"}</button></article>)}{!schedules.length && <div className="empty compact">尚未启用自动研究。</div>}</div>
      </div>
    </details>

    <section className="jobs-panel">
      <div className="panel-heading"><div><p className="eyebrow">RESEARCH RUNS / TRACE</p><h2>统一运行与安全审计记录</h2></div><span>{runs.length} 条记录</span></div>
      <div className="research-run-list">{runs.map((item) => <div className="research-run-entry" key={item.id}><article><span className={`job-state ${item.status}`} /><div><strong>{item.objective}</strong><small>{item.scenario ?? item.kind ?? "fin_factor"} · {item.dataset ?? "资料输入"} · {item.budget.loop_n ?? 0} 轮 · {item.budget.duration ?? "外层超时"}</small>{item.error && <small>{item.error}</small>}</div><div className="run-count"><strong>{(item.runtime?.candidates ?? 0) + (item.runtime?.model_candidates ?? 0) + (item.runtime?.quant_bundles ?? 0)}</strong><small>候选</small></div><code>{item.id.slice(0, 10)}</code><span>{statusText[item.status] ?? item.status}</span><button className="inline-action" type="button" onClick={() => toggleRunDetail(item.id)}>{openRunId === item.id ? "收起审计" : "查看审计"}</button></article>{openRunId === item.id && (detailLoading ? <div className="empty compact">正在读取安全审计证据…</div> : runDetail ? <RunAudit detail={runDetail} validationTarget={`${dataset || "未选择数据集"} / ${featureSetId || "未选择特征集"}`} onValidateGeneralModel={queueGeneralModelValidation} /> : null)}</div>)}{!runs.length && <div className="empty compact">尚无 RD-Agent 研究运行。</div>}</div>
    </section>
  </>;
}
