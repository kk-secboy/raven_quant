"use client";

import { ResourcePanel } from "./resource-panel";

export function RDAgentPanel({ api }: { api: string }) {
  return (
    <div className="grid">
      <ResourcePanel
        api={api}
        endpoint="/api/rdagent/runs"
        title="受治理的 RD-Agent 研究"
        eyebrow="RESEARCH / BUDGETED"
        empty="尚无自动研究运行。"
      >
        <p className="muted">
          配方目录由 /api/strategy-recipes/ 提供。RD-Agent 只能提出候选，不能批准策略或生成正式账户建议。
        </p>
      </ResourcePanel>
    </div>
  );
}
