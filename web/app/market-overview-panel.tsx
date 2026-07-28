"use client";

import { ResourcePanel } from "./resource-panel";

export function MarketOverviewPanel({ api }: { api: string }) {
  return (
    <ResourcePanel
      api={api}
      endpoint="/api/market/overview"
      title="市场与数据健康"
      eyebrow="POINT-IN-TIME MARKET VIEW"
      empty="尚无市场快照。"
    />
  );
}
