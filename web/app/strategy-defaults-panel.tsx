"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { apiFetch } from "./api-client";

type StrategyConfig = Record<string, number | string>;
type DefaultsState = {
  config: StrategyConfig;
  source: "built_in" | "database";
  revision: number;
  updated_by?: string | null;
  updated_at?: string | null;
};
type FieldSpec = {
  key: string;
  label: string;
  min?: number;
  max?: number;
  step: number;
  scale?: number;
};
type FieldGroup = { title: string; fields: FieldSpec[] };

const groups: FieldGroup[] = [
  {
    title: "组合构建与基础风控",
    fields: [
      { key: "topk", label: "持仓数量", min: 5, max: 500, step: 1 },
      { key: "n_drop", label: "缓冲替换数量", min: 0, max: 100, step: 1 },
      { key: "max_position_weight", label: "单票权重上限（%）", min: 0.1, max: 20, step: 0.1, scale: 100 },
      { key: "max_daily_turnover", label: "单日换手上限（%）", min: 0.1, max: 100, step: 1, scale: 100 },
      { key: "max_daily_loss", label: "单日亏损熔断（%）", min: 0.1, max: 20, step: 0.5, scale: 100 },
      { key: "stop_loss", label: "单票止损（%）", min: 0.1, max: 50, step: 0.5, scale: 100 },
      { key: "take_profit_partial", label: "首次止盈阈值（%）", min: 0.1, max: 200, step: 1, scale: 100 },
      { key: "take_profit_partial_fraction", label: "首次止盈减仓比例（%）", min: 1, max: 99, step: 1, scale: 100 },
      { key: "take_profit", label: "最终止盈清仓（%）", min: 0.1, max: 500, step: 1, scale: 100 },
      { key: "max_drawdown_reduce", label: "组合回撤降仓（%）", min: 0.1, max: 50, step: 0.5, scale: 100 },
      { key: "drawdown_reduction_exposure", label: "降仓后目标仓位（%）", min: 1, max: 99, step: 1, scale: 100 },
      { key: "max_drawdown_liquidate", label: "组合回撤清仓（%）", min: 0.1, max: 80, step: 0.5, scale: 100 },
    ],
  },
  {
    title: "行业、风格、流动性与容量",
    fields: [
      { key: "max_industry_weight", label: "行业权重上限（%）", min: 0.1, max: 100, step: 1, scale: 100 },
      { key: "max_industry_deviation", label: "行业偏离上限（%）", min: 0, max: 30, step: 0.5, scale: 100 },
      { key: "max_size_deviation", label: "市值风格偏离上限", min: 0, max: 2, step: 0.05 },
      { key: "min_average_daily_amount", label: "最低日均成交额（元）", min: 1000000, max: 100000000000, step: 1000000 },
      { key: "liquidity_lookback_days", label: "流动性回看交易日", min: 5, max: 252, step: 1 },
      { key: "capacity_notional", label: "容量测试资金（元）", min: 100000, max: 10000000000, step: 100000 },
      { key: "max_volume_participation", label: "日成交量参与上限（%）", min: 0.1, max: 20, step: 0.1, scale: 100 },
      { key: "min_capacity_fill_ratio", label: "容量最低成交率（%）", min: 0, max: 100, step: 1, scale: 100 },
    ],
  },
  {
    title: "基准相对组合优化",
    fields: [
      { key: "optimizer_alpha_weight", label: "因子收益权重", min: 0, max: 10, step: 0.01 },
      { key: "optimizer_tracking_penalty", label: "基准跟踪惩罚", min: 0, max: 100, step: 0.1 },
      { key: "optimizer_turnover_penalty", label: "换手惩罚", min: 0, max: 100, step: 0.01 },
    ],
  },
  {
    title: "回测、稳健性与压力门槛",
    fields: [
      { key: "max_tracking_error", label: "最大跟踪误差（%）", min: 0.1, max: 100, step: 0.5, scale: 100 },
      { key: "max_drawdown", label: "回测最大回撤（%）", min: 0.1, max: 100, step: 0.5, scale: 100 },
      { key: "max_turnover", label: "最大平均换手（%）", min: 0.1, max: 200, step: 1, scale: 100 },
      { key: "min_information_ratio", label: "最小信息比率", min: -5, max: 10, step: 0.1 },
      { key: "min_sharpe_ratio", label: "最小 Sharpe", min: -5, max: 10, step: 0.1 },
      { key: "min_sortino_ratio", label: "最小 Sortino", min: -5, max: 20, step: 0.1 },
      { key: "min_robustness_pass_rate", label: "稳健场景通过率（%）", min: 0, max: 100, step: 1, scale: 100 },
      { key: "rolling_window_days", label: "滚动窗口交易日", min: 60, max: 1260, step: 1 },
      { key: "rolling_step_days", label: "滚动步长交易日", min: 20, max: 504, step: 1 },
      { key: "min_rolling_windows", label: "最少滚动窗口数", min: 2, max: 20, step: 1 },
      { key: "min_rolling_pass_rate", label: "滚动窗口通过率（%）", min: 0, max: 100, step: 1, scale: 100 },
      { key: "event_window_days", label: "事件压力窗口交易日", min: 20, max: 126, step: 1 },
      { key: "event_count", label: "历史压力事件数量", min: 1, max: 20, step: 1 },
      { key: "max_event_underperformance", label: "事件最大落后（%）", min: 0, max: 50, step: 0.5, scale: 100 },
      { key: "min_event_stress_pass_rate", label: "事件压力通过率（%）", min: 0, max: 100, step: 1, scale: 100 },
      { key: "min_backtest_days", label: "最少回测交易日", min: 252, max: 2520, step: 1 },
    ],
  },
  {
    title: "交易成本",
    fields: [
      { key: "buy_commission_rate", label: "买入佣金（bp）", min: 0, max: 200, step: 0.1, scale: 10000 },
      { key: "sell_commission_rate", label: "卖出佣金（bp）", min: 0, max: 200, step: 0.1, scale: 10000 },
      { key: "stock_sell_stamp_duty_rate", label: "股票卖出印花税（bp）", min: 0, max: 200, step: 0.1, scale: 10000 },
      { key: "etf_sell_stamp_duty_rate", label: "ETF卖出印花税（bp）", min: 0, max: 200, step: 0.1, scale: 10000 },
      { key: "transfer_fee_rate", label: "过户费（bp）", min: 0, max: 200, step: 0.1, scale: 10000 },
      { key: "fixed_slippage_rate", label: "固定冲击（bp）", min: 0, max: 200, step: 0.1, scale: 10000 },
      { key: "impact_at_max_participation", label: "容量上限冲击（bp）", min: 0, max: 1000, step: 0.1, scale: 10000 },
      { key: "min_commission", label: "最低佣金（元）", min: 0, max: 1000, step: 0.1 },
    ],
  },
];

export function StrategyDefaultsPanel({ api }: { api: string }) {
  const [state, setState] = useState<DefaultsState | null>(null);
  const [draft, setDraft] = useState<StrategyConfig>({});
  const [reason, setReason] = useState("");
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    const response = await apiFetch(`${api}/api/settings/strategy-defaults`, { cache: "no-store" });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail ?? "无法读取策略默认模板");
    setState(body);
    setDraft(body.config);
  }, [api]);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      load().catch((error) => setMessage(error instanceof Error ? error.message : "读取失败"));
    }, 0);
    return () => window.clearTimeout(initial);
  }, [load]);

  function update(field: FieldSpec, displayed: number) {
    setDraft({ ...draft, [field.key]: displayed / (field.scale ?? 1) });
  }

  async function save(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setMessage("正在校验并保存新的默认模板……");
    try {
      const response = await apiFetch(`${api}/api/settings/strategy-defaults`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config: draft, reason }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "策略默认模板保存失败");
      setState(body);
      setDraft(body.config);
      setReason("");
      setMessage(`策略默认模板 r${body.revision} 已保存；现有策略版本不受影响。`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "策略默认模板保存失败");
    } finally {
      setSaving(false);
    }
  }

  return <section className="data-panel strategy-defaults-panel">
    <div className="panel-heading">
      <div><p className="eyebrow">VERSIONED STRATEGY DEFAULTS</p><h2>策略默认模板</h2></div>
      <span>{state?.source === "database" ? `数据库 r${state.revision}` : "系统内置 r0"}</span>
    </div>
    <p>这里只决定新建策略的初始值。每个策略版本会保存完整参数，已审批和运行中的版本不会随模板变化。</p>
    {message && <div className="notice">{message}</div>}
    <form onSubmit={save}>
      <section className="settings-card">
        <div className="card-heading"><div><span>PORTFOLIO CONSTRUCTION</span><strong>组合构建方式</strong></div></div>
        <label>默认方式<select value={String(draft.portfolio_construction ?? "topk_equal_weight")} onChange={(event) => setDraft({ ...draft, portfolio_construction: event.target.value })}><option value="topk_equal_weight">Top-K 等权</option><option value="benchmark_relative_qp">基准相对优化</option></select></label>
        <p>指数增强建议使用基准相对优化；波段策略可继续使用 Top-K 等权。</p>
      </section>
      {groups.map((group) => <section key={group.title} className="settings-card">
        <div className="card-heading"><div><span>CONFIGURATION GROUP</span><strong>{group.title}</strong></div></div>
        <div className="risk-grid">
          {group.fields.map((field) => <label key={field.key}>{field.label}<input
            type="number"
            min={field.min}
            max={field.max}
            step={field.step}
            value={typeof draft[field.key] === "number" ? Number(draft[field.key]) * (field.scale ?? 1) : ""}
            onChange={(event) => update(field, Number(event.target.value))}
            required
          /></label>)}
        </div>
      </section>)}
      <div className="execution-note"><b>固定安全语义</b><span>执行模型始终为 next_open；实盘开关、密钥和部署硬上限不能在这里放宽。</span></div>
      <label>修改理由<textarea minLength={10} maxLength={2000} value={reason} onChange={(event) => setReason(event.target.value)} placeholder="至少 10 个字符，说明为什么调整默认参数。" /></label>
      <button className="primary" disabled={saving || reason.trim().length < 10}>{saving ? "保存中……" : "保存为新的默认模板版本"}</button>
    </form>
  </section>;
}
