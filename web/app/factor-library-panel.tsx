"use client";

import { ResourcePanel } from "./resource-panel";

export function FactorLibraryPanel({ api }: { api: string }) {
  return (
    <ResourcePanel
      api={api}
      endpoint="/api/factors"
      title="因子候选与独立准入"
      eyebrow="FACTOR GOVERNANCE"
      empty="没有已登记的因子候选。"
    >
      <p className="muted">失败、超时和拒绝的评估全部留痕；只有独立复算通过的候选才能晋升。</p>
    </ResourcePanel>
  );
}
