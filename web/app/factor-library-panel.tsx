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
};

const pct = (value: unknown) => typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
const decimal = (value: unknown) => typeof value === "number" ? value.toFixed(3) : "—";
const statusText: Record<string, string> = {
  awaiting_evaluation: "等待评估", rejected_by_rdagent: "实现未通过", gate_failed: "门槛未通过",
  gate_passed: "待人工晋级", promoted: "已晋级", retired: "已退役",
};

export function FactorLibraryPanel({ api }: { api: string }) {
  const [policy, setPolicy] = useState<GatePolicy | null>(null);
  const [factors, setFactors] = useState<Candidate[]>([]);
  const [filter, setFilter] = useState("all");
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [reason, setReason] = useState("");
  const [message, setMessage] = useState("");

  async function load() {
    try {
      const [policyResponse, factorsResponse] = await Promise.all([
        apiFetch(`${api}/api/factors/gate-policy`, { cache: "no-store" }),
        apiFetch(`${api}/api/factors`, { cache: "no-store" }),
      ]);
      setPolicy(await policyResponse.json());
      setFactors(await factorsResponse.json());
    } catch {
      setMessage("无法读取因子治理记录，请确认 Python 后端正在运行。");
    }
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

  return <>
    {message && <div className="notice">{message}</div>}
    <section className="gate-strip">
      <div><span>|IC|</span><strong>≥ {policy?.min_abs_ic ?? "—"}</strong></div>
      <div><span>|ICIR|</span><strong>≥ {policy?.min_abs_icir ?? "—"}</strong></div>
      <div><span>|Rank IC|</span><strong>≥ {policy?.min_abs_rank_ic ?? "—"}</strong></div>
      <div><span>平均换手</span><strong>≤ {policy ? pct(policy.max_turnover) : "—"}</strong></div>
      <div><span>最大相关性</span><strong>≤ {policy ? decimal(policy.max_correlation) : "—"}</strong></div>
      <div><span>样本外天数</span><strong>≥ {policy?.min_test_days ?? "—"}</strong></div>
    </section>

    <section className="factor-intro">
      <div><p className="eyebrow">GOVERNED FACTOR REGISTRY</p><h2>因子不是代码仓库，而是可审计的研究资产</h2><p>RD-Agent 的“实现通过”只代表代码可运行。只有独立 Qlib 样本外、方向一致性、换手、相关性和扣成本收益全部过关，才会进入人工晋级队列。</p></div>
      <div className="factor-flow"><span>候选</span><i>→</i><span>实现校验</span><i>→</i><span>独立评估</span><i>→</i><span>人工晋级</span></div>
    </section>

    {selected ? <form className="promotion-box" onSubmit={promote}><div><span>人工晋级确认</span><strong>{selected.name}</strong><small>最新 Qlib 门槛已通过；填写本次晋级依据，操作会写入审计事件。</small></div><textarea value={reason} minLength={10} maxLength={2000} onChange={(event) => setReason(event.target.value)} placeholder="例如：样本外稳定、扣成本后为正，且与现有因子相关性可接受。" /><div><button type="button" onClick={() => setSelected(null)}>取消</button><button className="primary" disabled={reason.length < 10}>确认晋级</button></div></form> : null}

    <section className="data-panel">
      <div className="panel-heading"><div><p className="eyebrow">FACTOR CANDIDATES</p><h2>候选与晋级状态</h2></div><div className="segmented">{[["all", "全部"], ["gate_passed", "待晋级"], ["promoted", "已晋级"], ["gate_failed", "未通过"]].map(([value, label]) => <button key={value} className={filter === value ? "selected" : ""} onClick={() => setFilter(value)}>{label}</button>)}</div></div>
      <div className="table-wrap"><table className="factor-table"><thead><tr><th>因子</th><th>状态</th><th>IC / ICIR</th><th>Rank IC</th><th>换手</th><th>相关性</th><th>扣成本年化</th><th>操作</th></tr></thead><tbody>{visible.map((item) => {
        const metrics = item.latest_evaluation?.metrics ?? {};
        return <tr key={item.id}><td><strong>{item.name}</strong><small>{item.description}</small></td><td><span className={`state ${item.latest_evaluation?.gate_status === "passed" ? "ready" : item.status === "gate_failed" ? "failed" : "partial"}`}>{statusText[item.status] ?? item.status}</span></td><td>{decimal(metrics.ic)} <small>/ {decimal(metrics.icir)}</small></td><td>{decimal(metrics.rank_ic)}</td><td>{pct(metrics.turnover)}</td><td>{decimal(metrics.max_correlation)}</td><td>{pct(metrics.cost_adjusted_return)}</td><td>{item.status === "gate_passed" ? <button className="inline-action" onClick={() => setSelected(item)}>审阅晋级</button> : <span className="muted">—</span>}</td></tr>;
      })}</tbody></table>{!visible.length && <div className="empty">暂无符合筛选条件的因子。RD-Agent 生成后会自动进入独立 Qlib 评估，不会直接晋级。</div>}</div>
    </section>
  </>;
}
