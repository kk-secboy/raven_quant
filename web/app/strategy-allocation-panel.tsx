"use client";

import { ResourcePanel } from "./resource-panel";

const policyContract = {
  method: "risk_parity",
  role: "core_or_satellite",
  risk_budget: "frozen",
  member_cap: "hard_limit",
  max_drawdown_liquidate: "account_guard",
  recommendation_portfolio_id: "single_sender",
};

export function StrategyAllocationPanel({ api }: { api: string }) {
  return (
    <ResourcePanel
      api={api}
      endpoint="/api/strategy-allocations"
      title="核心 / 卫星账户分配"
      eyebrow="ONE BUDGET / ONE ARTIFACT"
      empty="尚无账户分配政策。"
    >
      <p className="muted">
        {policyContract.method} · {policyContract.role} · risk_budget · member_cap ·
        max_drawdown_liquidate · recommendation_portfolio_id
      </p>
      <p className="muted">
        推荐组合自动刷新由 /schedule/status 控制；非决策日复用当前 AllocationArtifact。
      </p>
    </ResourcePanel>
  );
}
