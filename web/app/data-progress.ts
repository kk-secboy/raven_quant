export type UnitStats = {
  planned: number;
  succeeded: number;
  pending: number;
  running: number;
  retry_waiting: number;
  terminal_failed: number;
  rate_limited: number;
  superseded: number;
  rows: number;
  next_retry_at?: string | null;
};

export type JobProgress = {
  status?: string;
  execution_phase?: string;
  phase_label?: string;
  datasets?: string[];
  target?: Record<string, unknown>;
  checkpoint?: Partial<UnitStats>;
  updated_at?: string;
  [key: string]: unknown;
};

export type DataJob = {
  id: string;
  kind: string;
  status: string;
  payload: Record<string, unknown>;
  progress?: JobProgress | null;
  error?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  next_attempt_at?: string | null;
  attempts?: number;
  max_attempts?: number;
};

export const phaseText: Record<string, string> = {
  blocked_prerequisite: "等待前置数据",
  ready_to_start: "可启动",
  queued: "排队中",
  planning: "正在规划窗口",
  prerequisites: "准备基础数据",
  downloading: "批量下载中",
  pagination: "继续分页",
  adaptive_recovery: "自适应拆分恢复",
  verifying: "验证分页完整性",
  snapshot: "构建不可变快照",
  rate_limit_cooldown: "限流冷却",
  retry_waiting: "等待自动重试",
  recoverable_failure: "可恢复失败",
  terminal_failure: "终止失败",
  partial: "需要补齐",
  verified: "已完成并验证",
};

export const jobKindText: Record<string, string> = {
  bootstrap: "A 股日频全量初始化",
  data_verify: "数据完整性验证",
  data_snapshot: "构建不可变快照",
  data_qlib: "构建 Qlib 数据集",
  qlib_baseline: "Qlib 基线研究",
  margin_eligibility_download: "融券资格历史",
  core_intraday_download: "核心资产 1 分钟线",
  ashare_5m_download: "全 A 股 5 分钟线",
  minute_qlib: "分钟 Qlib 数据集",
};

const bundleText: Record<string, string> = {
  cn_extended_daily: "A 股扩展日频",
  cn_funds: "ETF、指数与基金增强",
  cn_macro: "宏观经济与利率",
  cn_futures: "期货市场",
  cn_institutional: "机构研究与 ETF 申赎清单",
  cn_options_bonds: "期权、债券与可转债",
  hk_market: "港股市场",
  us_market: "美股市场",
  global_markets: "全球市场",
  cn_governance_risk: "公司治理与风险事件",
  cn_capital_flow: "全市场资金流",
  cn_fund_index_enhanced: "基金与指数增强",
  cn_derivatives_enhanced: "衍生品增强",
  global_rates_enhanced: "全球利率增强",
  research_corpus: "财经新闻语料",
};

export function jobDisplayName(job: Pick<DataJob, "kind" | "payload">) {
  const payload = job.payload ?? {};
  if (typeof payload.output_name === "string" && payload.output_name) return payload.output_name;
  if (typeof payload.snapshot_name === "string" && payload.snapshot_name) return payload.snapshot_name;
  if (typeof payload.bundle === "string" && payload.bundle) {
    return bundleText[payload.bundle] ?? String(payload.bundle).replaceAll("_", " ");
  }
  if (typeof payload.profile === "string" && payload.profile) {
    return `${jobKindText[job.kind] ?? job.kind} · ${payload.profile}`;
  }
  return jobKindText[job.kind] ?? job.kind.replaceAll("_", " ");
}

export function phaseLabel(phase?: string | null) {
  return phase ? phaseText[phase] ?? phase.replaceAll("_", " ") : "等待进度";
}

function stringValue(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : "";
}

export function targetText(payload?: Record<string, unknown> | null, progress?: JobProgress | null) {
  const target = progress?.target ?? {};
  const start = stringValue(target.start_date) || stringValue(payload?.start);
  const end = stringValue(target.end_date) || stringValue(payload?.end);
  const frequency = stringValue(target.frequency) || stringValue(payload?.frequency);
  const symbols = Number(target.symbols ?? 0);
  const parts = [];
  if (start || end) parts.push(`${start || "最早"} 至 ${end || "最新"}`);
  if (frequency) parts.push(frequency);
  if (symbols > 0) parts.push(`${symbols.toLocaleString("zh-CN")} 个标的`);
  return parts.join(" · ") || "使用任务配置范围";
}

export function retryTimeText(value?: string | null) {
  if (!value) return "—";
  return new Date(value).toLocaleString("zh-CN");
}
