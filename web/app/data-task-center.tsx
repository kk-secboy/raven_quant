"use client";

import { ResourcePanel } from "./resource-panel";
import { phaseLabel } from "./data-progress";

export type DataTask = {
  id: string;
  name: string;
  status: string;
  request_policy?: string;
};

export function DataTaskCenter({ api }: { api: string }) {
  return (
    <div className="grid">
      <ResourcePanel
        api={api}
        endpoint="/api/data-tasks"
        title="数据快照与质量任务"
        eyebrow="IMMUTABLE SNAPSHOT / PIT"
        empty="尚无数据任务。"
      >
        <p className="muted">
          请求策略：限速、断点续传、{phaseLabel.adaptive_recovery}；质量门失败时禁止产生新建议。
        </p>
      </ResourcePanel>
    </div>
  );
}
