"use client";

import { FormEvent, useState } from "react";
import { apiFetch } from "./api-client";
import { usePolling } from "./use-polling";

type GatePolicy = {
  min_abs_ic: number; min_abs_icir: number; min_abs_rank_ic: number;
  max_turnover: number; max_correlation: number; min_test_days: number; version: string;
};

type Evaluation = {
  gate_status: "passed" | "failed";
  gate_reasons: string[];
  metrics: Record<string, number | string | null>;
  evaluator_version: string;
};

type Candidate = {
  id: string; name: string; description: string; formulation?: string | null;
  status: string; research_run_id: string; rdagent_decision?: boolean | null;
  latest_evaluation?: Evaluation | null; updated_at: string;
  economic_family?: string | null; family_tags?: string[];
  similarity_cluster_id?: string | null; factor_definition_id?: string | null;
};

type LibraryDefinition = { id: string; economic_family: string; calculability: string; aliases: string[] };
type LibraryVersion = { id: string; status: string; member_count: number; source_alias_counts: Record<string, number> };
type SotaMember = { factor_candidate_id: string; economic_family: string; similarity_cluster_id: string };
type SotaVersion = { id: string; status: string; created_at: string; members?: SotaMember[] };

async function jsonResponse<T>(request: Promise<Response>): Promise<T> {
  const response = await request;
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

const pct = (value: unknown) => typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
const decimal = (value: unknown) => typeof value === "number" ? value.toFixed(3) : "—";
const statusText: Record<string, string> = {
  awaiting_evaluation: "等待评估", rejected_by_rdagent: "实现未通过", gate_failed: "门槛未通过",
  gate_passed: "待人工晋级", promoted: "已晋级", retired: "已退役",
};

export function FactorLibraryPanel({ api }: { api: string }) {
  const [policy, setPolicy] = useState<GatePolicy | null>(null);
  const [factors, setFactors] = useState<Candidate[]>([]);
  const [definitions, setDefinitions] = useState<LibraryDefinition[]>([]);
  const [versions, setVersions] = useState<LibraryVersion[]>([]);
  const [sota, setSota] = useState<SotaVersion | null>(null);
  const [filter, setFilter] = useState("all");
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [reason, setReason] = useState("");
  const [message, setMessage] = useState("");
  const [loadWarning, setLoadWarning] = useState("");

  async function load() {
    const [policyResult, factorsResult, libraryResult, versionsResult, sotaVersionsResult] = await Promise.allSettled([
      jsonResponse<GatePolicy>(apiFetch(`${api}/api/factors/gate-policy`, { cache: "no-store" })),
      jsonResponse<Candidate[]>(apiFetch(`${api}/api/factors`, { cache: "no-store" })),
      jsonResponse<LibraryDefinition[]>(apiFetch(`${api}/api/factor-library?limit=2000`, { cache: "no-store" })),
      jsonResponse<LibraryVersion[]>(apiFetch(`${api}/api/factor-library/versions`, { cache: "no-store" })),
      jsonResponse<SotaVersion[]>(apiFetch(`${api}/api/research-sota`, { cache: "no-store" })),
    ]);

    if (policyResult.status === "fulfilled") setPolicy(policyResult.value);
    if (factorsResult.status === "fulfilled") setFactors(factorsResult.value);
    if (libraryResult.status === "fulfilled") setDefinitions(libraryResult.value);
    if (versionsResult.status === "fulfilled") setVersions(versionsResult.value);

    let sotaDetailFailed = false;
    if (sotaVersionsResult.status === "fulfilled") {
      const active = sotaVersionsResult.value.find((item) => item.status === "active");
      if (!active) {
        setSota(null);
      } else {
        try {
          setSota(await jsonResponse<SotaVersion>(
            apiFetch(`${api}/api/research-sota/${active.id}`, { cache: "no-store" }),
          ));
        } catch {
          // Keep the last good SOTA detail; its list endpoint remains independent.
          sotaDetailFailed = true;
        }
      }
    }

    const failed = [policyResult, factorsResult, libraryResult, versionsResult, sotaVersionsResult]
      .filter((item) => item.status === "rejected").length;
    setLoadWarning(
      failed === 5
        ? "无法读取因子治理记录，请确认 QuantLab API 正在运行。"
        : failed || sotaDetailFailed
          ? "部分因子治理状态暂未更新，已保留上次成功数据。"
          : "",
    );
  }

  usePolling(load, 8000);

  async function promote(event: FormEvent) {
    event.preventDefault();
    if (!selected) return;
    const response = await apiFetch(`${api}/api/factors/${selected.id}/promote`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actor: "local-operator", reason }),
    });
    const body = await response.json();
    if (!response.ok) {
      setMessage(body.detail ?? "晋级失败");
      return;
    }
    setMessage(`${body.name} 已晋级为可复用研究因子。`);
    setSelected(null);
    setReason("");
    await load();
  }

  const visible = factors.filter((item) => filter === "all" || item.status === filter);
  const activeLibrary = versions.find((item) => item.status === "active");
  const familyCount = new Set(definitions.map((item) => item.economic_family)).size;
  const sotaIds = new Set((sota?.members ?? []).map((item) => item.factor_candidate_id));

  return <>
    {loadWarning && <div className="notice">{loadWarning}</div>}
    {message && <div className="notice">{message}</div>}
    <section className="gate-strip">
      <div><span>统一库</span><strong>{activeLibrary?.member_count ?? definitions.length}</strong></div>
      <div><span>经济家族</span><strong>{familyCount || "—"}</strong></div>
      <div><span>研究 SOTA</span><strong>{sota?.members?.length ?? 0}</strong></div>
      <div><span>|IC|</span><strong>≥ {policy?.min_abs_ic ?? "—"}</strong></div>
      <div><span>|ICIR|</span><strong>≥ {policy?.min_abs_icir ?? "—"}</strong></div>
      <div><span>|Rank IC|</span><strong>≥ {policy?.min_abs_rank_ic ?? "—"}</strong></div>
      <div><span>平均换手</span><strong>≤ {policy ? pct(policy.max_turnover) : "—"}</strong></div>
      <div><span>最大相关性</span><strong>≤ {policy ? decimal(policy.max_correlation) : "—"}</strong></div>
      <div><span>样本外天数</span><strong>≥ {policy?.min_test_days ?? "—"}</strong></div>
    </section>

    <section className="factor-intro">
      <div><p className="eyebrow">GOVERNED FACTOR REGISTRY</p><h2>因子不是代码仓库，而是可审计的研究资产</h2><p>RD-Agent 的“实现通过”只代表代码可运行。平台会用 Qlib 独立复算、全库去重、三窗口和增量消融；合格因子只能进入研究 SOTA，还要随模型和买卖规则完成策略评估、正式 OOS 与前向模拟，满足全部证据门后才会自动晋级。</p></div>
      <div className="factor-flow"><span>候选</span><i>→</i><span>Qlib复算</span><i>→</i><span>家族去重</span><i>→</i><span>研究SOTA</span></div>
    </section>

    {selected ? <form className="promotion-box" onSubmit={promote}><div><span>人工晋级确认</span><strong>{selected.name}</strong><small>最新 Qlib 门槛已通过；填写本次晋级依据，操作会写入审计事件。</small></div><textarea value={reason} minLength={10} maxLength={2000} onChange={(event) => setReason(event.target.value)} placeholder="例如：样本外稳定、扣成本后为正，且与现有因子相关性可接受。" /><div><button type="button" onClick={() => setSelected(null)}>取消</button><button className="primary" disabled={reason.length < 10}>确认晋级</button></div></form> : null}

    <section className="data-panel">
      <div className="panel-heading"><div><p className="eyebrow">FACTOR CANDIDATES</p><h2>候选与晋级状态</h2></div><div className="segmented">{[["all", "全部"], ["gate_passed", "待晋级"], ["promoted", "已晋级"], ["gate_failed", "未通过"]].map(([value, label]) => <button key={value} className={filter === value ? "selected" : ""} onClick={() => setFilter(value)}>{label}</button>)}</div></div>
      <div className="table-wrap"><table className="factor-table"><thead><tr><th>因子</th><th>家族 / 相关簇</th><th>状态</th><th>IC / ICIR</th><th>Rank IC</th><th>换手</th><th>相关性</th><th>扣成本年化</th><th>操作</th></tr></thead><tbody>{visible.map((item) => {
        const metrics = item.latest_evaluation?.metrics ?? {};
        return <tr key={item.id}><td><strong>{item.name}</strong><small>{item.description}</small></td><td><strong>{item.economic_family ?? "未归类"}</strong><small>{item.similarity_cluster_id?.slice(0, 18) ?? "待聚类"}</small></td><td><span className={`state ${sotaIds.has(item.id) || item.latest_evaluation?.gate_status === "passed" ? "ready" : item.status === "gate_failed" ? "failed" : "partial"}`}>{sotaIds.has(item.id) ? "当前 SOTA" : statusText[item.status] ?? item.status}</span></td><td>{decimal(metrics.ic)} <small>/ {decimal(metrics.icir)}</small></td><td>{decimal(metrics.rank_ic)}</td><td>{pct(metrics.turnover)}</td><td>{decimal(metrics.max_correlation)}</td><td>{pct(metrics.cost_adjusted_return)}</td><td>{item.status === "gate_passed" ? <button className="inline-action" onClick={() => setSelected(item)}>审阅晋级</button> : <span className="muted">—</span>}</td></tr>;
      })}</tbody></table>{!visible.length && <div className="empty">暂无符合筛选条件的因子。RD-Agent 生成后会自动进入独立 Qlib 评估，不会直接晋级。</div>}</div>
    </section>
  </>;
}
