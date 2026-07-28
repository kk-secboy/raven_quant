"use client";

import { ResourcePanel } from "./resource-panel";

const formalRecipes = ["full_market_multifactor", "industry_neutral_qp", "文档策略配方"];

export function BacktestPanel({ api }: { api: string }) {
  return (
    <ResourcePanel
      api={api}
      endpoint="/api/backtests"
      title="Qlib 正式回测与审批"
      eyebrow="OOS / COST / CAPACITY"
      empty="尚无正式回测。"
    >
      <p className="muted">
        {formalRecipes.join(" · ")}；必须绑定 execution_dataset、冻结 Recorder 与执行契约哈希。
      </p>
    </ResourcePanel>
  );
}
