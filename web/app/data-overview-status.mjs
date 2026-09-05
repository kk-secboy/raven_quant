/**
 * The headline describes confirmed catalogue readiness, never the absence of running jobs.
 * @param {{
 *   ready_tasks?: number, actionable_tasks?: number, partial_tasks?: number,
 *   failed_tasks?: number, running_tasks?: number, waiting_tasks?: number,
 *   terminal_failed_tasks?: number
 * } | null | undefined} overview
 * @param {{loading?: boolean, refreshFailed?: boolean}} options
 */
export function dataReadinessView(overview, { loading = false, refreshFailed = false } = {}) {
  if (refreshFailed) return { state: "stale", label: "显示上次读取状态", ready: false };
  if (loading || !overview) return { state: "loading", label: "正在连接", ready: false };
  const counts = [overview.ready_tasks, overview.actionable_tasks, overview.partial_tasks,
    overview.failed_tasks, overview.running_tasks, overview.waiting_tasks, overview.terminal_failed_tasks];
  if (counts.some((value) => !Number.isInteger(value) || value < 0)
    || overview.ready_tasks > overview.actionable_tasks) {
    return { state: "unknown", label: "数据能力状态待确认", ready: false };
  }
  if (overview.failed_tasks > 0 || overview.terminal_failed_tasks > 0) {
    return { state: "failed", label: "有数据任务失败待处理", ready: false };
  }
  if (overview.running_tasks > 0) {
    return { state: "active", label: "数据任务正在执行", ready: false };
  }
  if (overview.actionable_tasks === 0) {
    return { state: "empty", label: "尚无已登记的数据能力", ready: false };
  }
  if (overview.ready_tasks === overview.actionable_tasks
    && overview.partial_tasks === 0 && overview.waiting_tasks === 0) {
    return { state: "ready", label: "数据能力已就绪", ready: true };
  }
  return { state: "incomplete", label: "仍有数据能力待准备", ready: false };
}
