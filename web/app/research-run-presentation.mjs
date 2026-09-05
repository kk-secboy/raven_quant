const ACTIVE_STATUSES = new Set(["queued", "running", "exporting", "evaluating"]);
const DISPLAY_STATUSES = new Set([
  "queued", "running", "succeeded", "failed", "blocked", "interrupted", "stopping", "unknown",
]);
const PHASE_LABELS = {
  proposal: "生成研究提案",
  parameter_search: "参数搜索与独立验证",
  evaluation: "独立评估",
  policy_only: "策略规则验证",
  full_stack: "完整策略验证",
  formal_final_oos: "正式最终样本外验证",
  settlement: "研究结算",
  unknown: "阶段待确认",
};
const STATUS_LABELS = {
  queued: "等待执行",
  running: "运行中",
  succeeded: "已完成",
  failed: "执行失败",
  blocked: "未通过",
  interrupted: "已中止",
  stopping: "正在停止",
  unknown: "状态待确认",
};

/**
 * @typedef {{
 *   display_status?: string, label?: string, reason_code?: string | null,
 *   safe_reason?: string | null, execution_phase?: string, phase_label?: string
 * }} SafeResearchPresentation
 * @typedef {{
 *   id: string, status: string, created_at?: string, finished_at?: string | null,
 *   presentation?: SafeResearchPresentation | null,
 *   linked_execution?: {
 *     job_id: string, kind: string, status: string, attempts?: number, max_attempts?: number,
 *     parameter_experiment_id?: string | null, updated_at?: string | null,
 *     presentation?: SafeResearchPresentation | null
 *   } | null
 * }} PresentableResearchRun
 */

/** @param {PresentableResearchRun} run */
export function isActiveResearchRun(run) {
  return ACTIVE_STATUSES.has(run.status);
}

/**
 * Keep the original audit status and exact IDs; never infer a recovery from similar objectives.
 * @template {PresentableResearchRun} T
 * @param {T[]} runs
 */
export function groupResearchRuns(runs) {
  return {
    active: runs.filter(isActiveResearchRun),
    history: runs.filter((run) => !isActiveResearchRun(run)),
  };
}

/** @param {string | null | undefined} value */
export function researchTime(value) {
  if (!value) return "未提供";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "未提供";
  return date.toLocaleString("zh-CN", {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
    hour12: false,
  });
}

/**
 * Only consume the safe presentation contract. Raw errors are deliberately not interpreted.
 * @param {PresentableResearchRun} run
 * @param {{ historical?: boolean, stale?: boolean }} options
 */
export function researchRunView(run, { historical = false, stale = false } = {}) {
  const presentation = run.presentation;
  const suppliedStatus = presentation?.display_status;
  const fallbackStatus = run.status === "cancelled" ? "interrupted"
    : run.status === "exporting" || run.status === "evaluating" ? "running" : run.status;
  const displayStatus = DISPLAY_STATUSES.has(suppliedStatus ?? "")
    ? suppliedStatus : DISPLAY_STATUSES.has(fallbackStatus) ? fallbackStatus : "unknown";
  let label = presentation?.label || STATUS_LABELS[displayStatus];
  if (historical && displayStatus === "failed") label = "历史失败";
  if (historical && displayStatus === "interrupted") label = "历史中止";
  if (historical && displayStatus === "blocked") label = "历史未通过";
  if (stale) label = `上次状态：${label}`;
  const phase = presentation?.execution_phase ?? "unknown";
  const phaseLabel = Object.hasOwn(PHASE_LABELS, phase)
    ? presentation?.phase_label || PHASE_LABELS[phase] : PHASE_LABELS.unknown;
  const hasPartialFailure = displayStatus === "running" && presentation?.reason_code === "partial_trial_failure";
  const tone = stale ? "muted" : displayStatus === "failed" ? "danger"
    : displayStatus === "succeeded" ? "success"
      : hasPartialFailure || ["interrupted", "stopping", "blocked"].includes(displayStatus) ? "warning"
        : ["queued", "running"].includes(displayStatus) ? "active" : "muted";
  return {
    label, displayStatus, tone, phaseLabel,
    safeReason: presentation?.safe_reason || null,
    reasonCode: presentation?.reason_code || null,
    originalStatus: run.status,
    jobId: run.linked_execution?.job_id ?? null,
    phaseUpdatedAt: researchTime(run.linked_execution?.updated_at),
    createdAt: researchTime(run.created_at),
    finishedAt: researchTime(run.finished_at),
  };
}

/**
 * Reject old unfiltered responses rather than placing historical failures in the active list.
 * @param {Response} response
 * @param {"active" | "history"} group
 * @param {number} page
 */
export async function readResearchPage(response, group, page) {
  if (!response.ok) throw new Error("Research status could not be refreshed");
  const rows = await response.json();
  if (!Array.isArray(rows) || rows.some((run) => !run?.id || typeof run.status !== "string"
    || (group === "active") !== isActiveResearchRun(run))) {
    throw new Error("Research status group could not be verified");
  }
  const total = Number(response.headers.get("X-Total-Count") ?? rows.length);
  if (!Number.isInteger(total) || total < rows.length) throw new Error("Research total is invalid");
  return { rows, total, page, cacheState: response.headers.get("X-QuantLab-Cache") };
}

/**
 * Retain the exact last successful page on a failed refresh, without advancing freshness.
 * @template T
 * @param {T} previous
 */
export function failedResearchPage(previous) {
  return { ...previous, state: "error" };
}

/**
 * Successful fresh reads alone advance the displayed update timestamp.
 * @template T
 * @param {{rows: T[], total: number, state: string, updatedAt: string | null, page?: number}} previous
 * @param {{rows: T[], total: number, cacheState: string | null, page?: number}} result
 * @param {string} now
 */
export function receivedResearchPage(previous, result, now) {
  const stale = result.cacheState === "stale";
  return {
    rows: result.rows,
    total: result.total,
    state: stale ? "stale" : "ready",
    updatedAt: stale ? previous.updatedAt : now,
    page: result.page ?? previous.page ?? 0,
  };
}

/** @param {{state: string, updatedAt: string | null}} page */
export function researchRefreshMessage(page) {
  const last = page.updatedAt ? `上次成功刷新 ${researchTime(page.updatedAt)}` : "尚未成功读取";
  if (page.state === "loading") return "正在读取研究状态…";
  if (page.state === "error") return `刷新失败，当前状态暂时无法确认。 ${last}`;
  if (page.state === "stale") return `当前显示上次读取的记录，状态可能已变化。 ${last}`;
  return `最近刷新 ${researchTime(page.updatedAt)}`;
}
