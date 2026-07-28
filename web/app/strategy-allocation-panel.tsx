"use client";

import { FormEvent, useMemo, useState } from "react";
import { apiFetch } from "./api-client";
import { usePolling } from "./use-polling";

type Dataset = { name: string; ready: boolean; reproducible: boolean; end_date: string };
type StrategyVersion = { id: string; version: number; status: string; strategy_type: string };
type Strategy = { id: string; name: string; versions: StrategyVersion[] };
type Member = {
  strategy_version_id: string; recommendation_portfolio_id?: string | null; target_weight: number;
  role: "core" | "satellite"; risk_budget: number; member_cap: number;
  annualized_volatility: number; risk_contribution: number;
};
type AllocationEvent = {
  id: number; severity: string; event_type: string; rule: string; status: string;
  observed?: number; limit_value?: number; resolution_reason?: string | null;
};
type AllocationAutomation = {
  status: string; effective_status?: string; timezone: string; run_time: string;
  members: Array<{
    id: string; status: string; desired_status: string;
    payload: { recommendation_portfolio_id?: string };
    suspension_reason?: string | null; next_run_at?: string | null;
  }>;
};
type Allocation = {
  id: string; name: string; status: string; total_capital: number; cash_reserve: number;
  nav: number; members: Member[];
  analysis: { highest_pairwise_correlation: number; portfolio_volatility: number };
  nav_history: Array<{ drawdown: number; annualized_volatility: number }>;
  events: AllocationEvent[];
  automation?: AllocationAutomation | null;
};

const money = (value: number) => new Intl.NumberFormat("zh-CN", {
  style: "currency", currency: "CNY", maximumFractionDigits: 0,
}).format(value);
const pct = (value?: number) => typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";

export function StrategyAllocationPanel({ api }: { api: string }) {
  const [allocations, setAllocations] = useState<Allocation[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [selectedVersions, setSelectedVersions] = useState<Record<string, boolean>>({});
  const [memberRoles, setMemberRoles] = useState<Record<string, "core" | "satellite">>({});
  const [memberRiskBudgets, setMemberRiskBudgets] = useState<Record<string, number>>({});
  const [memberCaps, setMemberCaps] = useState<Record<string, number>>({});
  const [name, setName] = useState("低相关核心卫星组合");
  const [dataset, setDataset] = useState("");
  const [capital, setCapital] = useState(5_000_000);
  const [method, setMethod] = useState("risk_parity");
  const [lookback, setLookback] = useState(252);
  const [targetVolatility, setTargetVolatility] = useState(0.15);
  const [maxPairwiseCorrelation, setMaxPairwiseCorrelation] = useState(0.70);
  const [maxStrategyWeight, setMaxStrategyWeight] = useState(0.70);
  const [maxMemberDrawdown, setMaxMemberDrawdown] = useState(0.08);
  const [maxDrawdownReduce, setMaxDrawdownReduce] = useState(0.10);
  const [maxDrawdownLiquidate, setMaxDrawdownLiquidate] = useState(0.15);
  const [approvalReason, setApprovalReason] = useState("");
  const [riskResolution, setRiskResolution] = useState("");
  const [scheduleTime, setScheduleTime] = useState("15:30");
  const [scheduleMisfireGrace, setScheduleMisfireGrace] = useState(1800);
  const [message, setMessage] = useState("");
  const approved = useMemo(() => strategies.flatMap((strategy) => strategy.versions
    .filter((version) => version.status === "approved")
    .map((version) => ({ strategy, version }))), [strategies]);
  const selected = allocations.find((item) => item.id === selectedId) ?? allocations[0];

  async function load() {
    const [allocationResponse, strategyResponse, datasetResponse] = await Promise.all([
      apiFetch(`${api}/api/strategy-allocations`, { cache: "no-store" }),
      apiFetch(`${api}/api/strategies`, { cache: "no-store" }),
      apiFetch(`${api}/api/qlib/datasets`, { cache: "no-store" }),
    ]);
    if (!allocationResponse.ok) return;
    const body: Allocation[] = await allocationResponse.json();
    if (strategyResponse.ok) setStrategies(await strategyResponse.json() as Strategy[]);
    if (datasetResponse.ok) {
      const available = (await datasetResponse.json() as Dataset[]).filter(
        (item) => item.ready && item.reproducible,
      );
      setDatasets(available);
    }
    setAllocations(body);
    if (!selectedId && body.length) setSelectedId(body[0].id);
    if (!dataset && datasets.length) setDataset(datasets[0].name);
  }

  usePolling(load, 8000);

  async function create(event: FormEvent) {
    event.preventDefault();
    const ids = Object.entries(selectedVersions).filter(([, value]) => value).map(([id]) => id);
    if (!(maxMemberDrawdown < maxDrawdownReduce && maxDrawdownReduce < maxDrawdownLiquidate)) {
      setMessage("回撤阈值必须依次递增：成员熔断 < 组合减仓 < 组合清仓。");
      return;
    }
    const response = await apiFetch(`${api}/api/strategy-allocations`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name, dataset, total_capital: capital, allocation_method: method,
        lookback_days: lookback, target_volatility: targetVolatility,
        max_pairwise_correlation: maxPairwiseCorrelation, max_strategy_weight: maxStrategyWeight,
        max_member_drawdown: maxMemberDrawdown, max_drawdown_reduce: maxDrawdownReduce,
        max_drawdown_liquidate: maxDrawdownLiquidate,
        members: ids.map((strategy_version_id) => {
          const role = memberRoles[strategy_version_id] ?? "core";
          return {
            strategy_version_id,
            weight: method === "fixed" ? 1 / ids.length : undefined,
            role,
            risk_budget: memberRiskBudgets[strategy_version_id] ?? 1,
            member_cap: memberCaps[strategy_version_id] ?? (role === "satellite" ? 0.15 : 0.70),
          };
        }), actor: "allocation-owner",
      }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "策略组合分析失败"); return; }
    setSelectedId(body.id);
    setMessage("相关性和风险预算分析通过，等待第二位管理员审批。");
    await load();
  }

  async function approve(event: FormEvent) {
    event.preventDefault();
    if (!selected) return;
    const response = await apiFetch(`${api}/api/strategy-allocations/${selected.id}/approve`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actor: "allocation-approver", reason: approvalReason }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "策略组合审批失败"); return; }
    setMessage("核心 / 卫星分配已审批，可作为统一模拟盘的受治理来源。");
    setApprovalReason("");
    await load();
  }

  async function refreshAllocation() {
    if (!selected) return;
    const response = await apiFetch(`${api}/api/strategy-allocations/${selected.id}/refresh`, {
      method: "POST",
    });
    const body = await response.json();
    setMessage(response.ok ? `组合净值刷新：${body.refresh_status}` : body.detail ?? "刷新失败");
    await load();
  }

  async function changeStatus(status: "active" | "paused") {
    if (!selected) return;
    const response = await apiFetch(`${api}/api/strategy-allocations/${selected.id}/status`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, actor: "local-operator" }),
    });
    const body = await response.json();
    setMessage(response.ok ? (status === "active" ? "主组合及子组合已恢复。" : "主组合及活动子组合已暂停。") : body.detail ?? "组合状态更新失败");
    await load();
  }

  async function configureAutomation() {
    if (!selected) return;
    const response = await apiFetch(`${api}/api/strategy-allocations/${selected.id}/schedule`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        timezone: "Asia/Shanghai", run_time: scheduleTime,
        trading_days_only: true,
        misfire_grace_seconds: scheduleMisfireGrace, actor: "local-operator",
      }),
    });
    const body = await response.json();
    setMessage(response.ok ? "全部成员的盘后推荐刷新计划已配置。" : body.detail ?? "推荐刷新计划配置失败");
    await load();
  }

  async function changeAutomationStatus(status: "active" | "paused") {
    if (!selected) return;
    const response = await apiFetch(`${api}/api/strategy-allocations/${selected.id}/schedule/status`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, actor: "local-operator" }),
    });
    const body = await response.json();
    setMessage(response.ok ? (status === "active" ? "组合自动刷新已恢复。" : "组合自动刷新已暂停。") : body.detail ?? "推荐刷新计划状态更新失败");
    await load();
  }

  async function retireAutomation() {
    if (!selected) return;
    const response = await apiFetch(`${api}/api/strategy-allocations/${selected.id}/schedule`, {
      method: "DELETE", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actor: "local-operator" }),
    });
    const body = await response.json();
    setMessage(response.ok ? "组合自动刷新已退休，历史运行记录仍保留。" : body.detail ?? "推荐刷新计划退休失败");
    await load();
  }

  async function actOnRisk(item: AllocationEvent, action: "acknowledge" | "resolve") {
    if (!selected) return;
    const response = await apiFetch(`${api}/api/strategy-allocations/${selected.id}/events/${item.id}/${action}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actor: "local-operator", ...(action === "resolve" ? { reason: riskResolution } : {}) }),
    });
    const body = await response.json();
    if (!response.ok) { setMessage(body.detail ?? "组合级风险事件处置失败"); return; }
    setMessage(action === "acknowledge" ? "组合级风险事件已确认。" : "组合级风险事件已完成处置；全部严重事件关闭后才能恢复。");
    if (action === "resolve") setRiskResolution("");
    await load();
  }

  const selectedCount = Object.values(selectedVersions).filter(Boolean).length;
  const selectedCoreCount = approved.filter(({ version }) =>
    selectedVersions[version.id]
    && (memberRoles[version.id] ?? (version.strategy_type === "pair" ? "satellite" : "core")) === "core",
  ).length;
  const validDrawdownThresholds = maxMemberDrawdown < maxDrawdownReduce
    && maxDrawdownReduce < maxDrawdownLiquidate;
  const latestNav = selected?.nav_history[0];
  return <section className="data-panel strategy-allocation-panel">
    <div className="panel-heading"><div><p className="eyebrow">CORE SATELLITE / QLIB OPTIMIZER</p><h2>核心 / 卫星风险预算与组合级风控</h2><p>核心权重不少于 70%，卫星不高于 30%，单卫星上限 15%；风险数值求解复用 Qlib，项目只执行治理上限。</p></div><span className={`state ${selected?.status === "active" ? "ready" : "partial"}`}>{selected?.status ?? "尚未创建"}</span></div>
    {message && <div className="notice">{message}</div>}
    <div className="portfolio-lower">
      <form className="portfolio-launcher" onSubmit={create}>
        <label>主组合名称<input value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label>Qlib 快照<select value={dataset} onChange={(event) => setDataset(event.target.value)}>{datasets.map((item) => <option key={item.name} value={item.name}>{item.name} · 截至 {item.end_date}</option>)}</select></label>
        <label>分配方法<select value={method} onChange={(event) => setMethod(event.target.value)}><option value="risk_parity">风险平价</option><option value="inverse_volatility">逆波动率</option><option value="fixed">固定等权</option></select></label>
        <label>总资金<input type="number" min="500000" step="100000" value={capital} onChange={(event) => setCapital(Number(event.target.value))} /></label>
        <label>回看交易日<input type="number" min="60" value={lookback} onChange={(event) => setLookback(Number(event.target.value))} /></label>
        <div className="risk-grid">
          <label>目标年化波动率（%）<input type="number" min="0.1" max="50" step="0.5" value={targetVolatility * 100} onChange={(event) => setTargetVolatility(Number(event.target.value) / 100)} /></label>
          <label>策略相关性上限（%）<input type="number" min="-99" max="99" step="1" value={maxPairwiseCorrelation * 100} onChange={(event) => setMaxPairwiseCorrelation(Number(event.target.value) / 100)} /></label>
          <label>单策略权重上限（%）<input type="number" min="1" max="100" step="1" value={maxStrategyWeight * 100} onChange={(event) => setMaxStrategyWeight(Number(event.target.value) / 100)} /></label>
          <label>成员回撤熔断（%）<input type="number" min="0.1" max="50" step="0.5" value={maxMemberDrawdown * 100} onChange={(event) => setMaxMemberDrawdown(Number(event.target.value) / 100)} /></label>
          <label>组合回撤减仓（%）<input type="number" min="0.1" max="50" step="0.5" value={maxDrawdownReduce * 100} onChange={(event) => setMaxDrawdownReduce(Number(event.target.value) / 100)} /></label>
          <label>组合回撤清仓（%）<input type="number" min="0.1" max="50" step="0.5" value={maxDrawdownLiquidate * 100} onChange={(event) => setMaxDrawdownLiquidate(Number(event.target.value) / 100)} /></label>
        </div>
        {!validDrawdownThresholds && <small className="danger-text">成员熔断、组合减仓、组合清仓阈值必须依次递增。</small>}
        <div className="strategy-picker"><span>已审批策略（至少两个，至少一个核心）</span>{approved.map(({ strategy, version }) => {
          const selectedMember = Boolean(selectedVersions[version.id]);
          const role = memberRoles[version.id] ?? (version.strategy_type === "pair" ? "satellite" : "core");
          const cap = memberCaps[version.id] ?? (role === "satellite" ? 0.15 : 0.70);
          return <div className="allocation-member-editor" key={version.id}>
            <label><input type="checkbox" checked={selectedMember} onChange={(event) => {
              setSelectedVersions((current) => ({ ...current, [version.id]: event.target.checked }));
              if (event.target.checked) {
                setMemberRoles((current) => ({ ...current, [version.id]: role }));
                setMemberRiskBudgets((current) => ({ ...current, [version.id]: current[version.id] ?? 1 }));
                setMemberCaps((current) => ({ ...current, [version.id]: current[version.id] ?? cap }));
              }
            }} />{strategy.name} · v{version.version} · {version.strategy_type === "pair" ? "配对" : "多因子"}</label>
            {selectedMember && <div className="form-row">
              <label>角色<select value={role} onChange={(event) => {
                const nextRole = event.target.value as "core" | "satellite";
                setMemberRoles((current) => ({ ...current, [version.id]: nextRole }));
                setMemberCaps((current) => ({ ...current, [version.id]: Math.min(current[version.id] ?? (nextRole === "satellite" ? 0.15 : 0.70), nextRole === "satellite" ? 0.15 : 0.70) }));
              }}><option value="core">核心</option><option value="satellite">卫星</option></select></label>
              <label>风险预算<input type="number" min="0.01" max="1" step="0.05" value={memberRiskBudgets[version.id] ?? 1} onChange={(event) => setMemberRiskBudgets((current) => ({ ...current, [version.id]: Number(event.target.value) }))} /></label>
              <label>成员上限（%）<input type="number" min="1" max={role === "satellite" ? 15 : 70} step="1" value={cap * 100} onChange={(event) => setMemberCaps((current) => ({ ...current, [version.id]: Number(event.target.value) / 100 }))} /></label>
            </div>}
          </div>;
        })}</div>
        <button className="primary" disabled={selectedCount < 2 || selectedCoreCount < 1 || !dataset || !validDrawdownThresholds}>计算风险预算并创建草案</button>
      </form>
      <article className="portfolio-summary">
        <label>当前主组合<select value={selected?.id ?? ""} onChange={(event) => setSelectedId(event.target.value)}>{allocations.length ? allocations.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.status}</option>) : <option value="">尚无主组合</option>}</select></label>
        <div className="portfolio-kpis"><div><span>主组合净值</span><strong>{money(selected?.nav ?? 0)}</strong></div><div><span>现金储备</span><strong>{money(selected?.cash_reserve ?? 0)}</strong></div><div><span>历史相关性</span><strong>{pct(selected?.analysis.highest_pairwise_correlation)}</strong></div><div><span>预期波动率</span><strong>{pct(selected?.analysis.portfolio_volatility)}</strong></div><div><span>实际波动率</span><strong>{pct(latestNav?.annualized_volatility)}</strong></div><div><span>组合回撤</span><strong>{pct(latestNav?.drawdown)}</strong></div></div>
        {selected?.status === "draft" && <form className="approval-panel" onSubmit={approve}><textarea value={approvalReason} onChange={(event) => setApprovalReason(event.target.value)} placeholder="第二位管理员填写相关性、风险预算和容量审批依据" /><button className="primary" disabled={approvalReason.length < 10}>第二人审批并创建子组合</button></form>}
        {selected && selected.status !== "draft" && <><button className="secondary-action" onClick={refreshAllocation}>刷新组合净值与风险状态</button><button className="secondary-action" disabled={["liquidation_pending", "risk_reduction_pending"].includes(selected.status)} onClick={() => changeStatus(selected.status === "active" ? "paused" : "active")}>{selected.status === "active" ? "暂停组合" : "恢复组合"}</button><div className="approval-panel"><strong>推荐组合自动刷新</strong><small>组合状态和风控暂停分开保存；风险减仓与清仓不会被人工暂停覆盖。</small><label>盘后刷新时间<input type="time" min="15:10" value={scheduleTime} onChange={(event) => setScheduleTime(event.target.value)} /></label><label>错过宽限（秒）<input type="number" min="60" max="86400" step="60" value={scheduleMisfireGrace} onChange={(event) => setScheduleMisfireGrace(Number(event.target.value))} /></label><div><button className="primary" onClick={configureAutomation}>{selected.automation ? "更新全部推荐刷新计划" : "启用全部推荐刷新计划"}</button>{selected.automation && <><button className="secondary-action" onClick={() => changeAutomationStatus(selected.automation?.status === "active" ? "paused" : "active")}>{selected.automation.status === "active" ? "暂停自动刷新" : "恢复自动刷新"}</button><button className="secondary-action" onClick={retireAutomation}>退休自动刷新</button></>}</div>{selected.automation && <small>状态：{selected.automation.effective_status ?? selected.automation.status} · {selected.automation.members.filter((item) => item.status === "active").length}/{selected.automation.members.length} 个推荐刷新计划运行中</small>}</div></>}
      </article>
    </div>
    {selected && <div className="table-wrap"><table className="portfolio-table"><thead><tr><th>策略版本</th><th>角色</th><th>目标权重 / 上限</th><th>风险预算 / 贡献</th><th>策略波动率</th><th>研究推荐</th></tr></thead><tbody>{selected.members.map((member) => <tr key={member.strategy_version_id}><td><code>{member.strategy_version_id.slice(0, 12)}</code></td><td>{member.role === "satellite" ? "卫星" : "核心"}</td><td>{pct(member.target_weight)} / {pct(member.member_cap)}</td><td>{member.risk_budget.toFixed(2)} / {pct(member.risk_contribution)}</td><td>{pct(member.annualized_volatility)}</td><td>{member.recommendation_portfolio_id ? <code>{member.recommendation_portfolio_id.slice(0, 12)}</code> : "—"}</td></tr>)}</tbody></table></div>}
    {selected && <div className="portfolio-lower"><label>组合级处置结论<textarea value={riskResolution} onChange={(event) => setRiskResolution(event.target.value)} placeholder="记录清仓/减仓执行、子账本核对和恢复依据（至少 10 字）" /></label><div className="risk-list">{selected.events.filter((item) => item.severity === "critical").map((item) => <article key={item.id}><span className="job-state failed" /><div><strong>{item.rule}</strong><small>{item.event_type} · {pct(item.observed)} / {pct(item.limit_value)}{item.resolution_reason ? ` · ${item.resolution_reason}` : ""}</small></div><span>{item.status}</span><div>{item.status === "open" && <button className="inline-action" onClick={() => actOnRisk(item, "acknowledge")}>确认</button>}{["open", "acknowledged"].includes(item.status) && <button className="inline-action" disabled={riskResolution.trim().length < 10} onClick={() => actOnRisk(item, "resolve")}>完成处置</button>}</div></article>)}</div></div>}
  </section>;
}
