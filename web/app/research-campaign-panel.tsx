"use client";

import { ResourcePanel } from "./resource-panel";

export function ResearchCampaignPanel({ api }: { api: string }) {
  return (
    <ResourcePanel
      api={api}
      endpoint="/api/research-programs"
      title="连续研究活动"
      eyebrow="RESEARCH PROGRAM / CAMPAIGN"
      empty="尚无研究计划。"
    />
  );
}
