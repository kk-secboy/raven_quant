"use client";

import { ResourcePanel } from "./resource-panel";

const governedSources = ["recommendation", "strategy_version", "allocation"] as const;
const executionFrequencies = ["1min", "5min"] as const;

export function PortfolioPanel({ api }: { api: string }) {
  return (
    <div className="grid">
      <ResourcePanel
        api={api}
        endpoint="/api/recommendation-portfolios"
        title="账户综合建议"
        eyebrow="RECOMMENDATION TRACKING"
        empty="尚无综合建议账户。"
      >
        <p className="muted">建议层只生成账户目标，不产生订单、成交或券商指令。</p>
      </ResourcePanel>
      <ResourcePanel
        api={api}
        endpoint="/api/simulation-portfolios"
        title="统一模拟账本"
        eyebrow="UNIFIED SIMULATION LEDGER"
        empty="尚无模拟账户。"
      >
        <p className="muted">
          source_type：{governedSources.join(" / ")}；execution_frequency：{executionFrequencies.join(" / ")}。
        </p>
      </ResourcePanel>
    </div>
  );
}
