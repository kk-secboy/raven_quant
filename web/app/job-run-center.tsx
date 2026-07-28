"use client";

import { ResourcePanel } from "./resource-panel";

export function JobRunCenter({ api }: { api: string }) {
  const job = { id: "selected" };
  const logRoute = `/api/jobs/${job.id}/log`;
  return (
    <ResourcePanel
      api={api}
      endpoint="/api/jobs"
      title="任务、告警与恢复"
      eyebrow="JOB CONTROL / FAIL CLOSED"
      empty="尚无任务。"
    >
      <p className="muted">job-progress-card · 日志入口 {logRoute} · 失败任务不会静默丢弃。</p>
    </ResourcePanel>
  );
}
