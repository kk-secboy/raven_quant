"use client";

import { ResourcePanel } from "./resource-panel";

const selectedVersion = "selected";
const researchRoutes = {
  strategy: "/api/pair-strategies",
  backtest: `/api/strategy-versions/${selectedVersion}/pair-backtests/`,
};

export function PairSatellitePanel({ api }: { api: string }) {
  return (
    <ResourcePanel
      api={api}
      endpoint="/api/pair-strategies"
      title="股票配对离线研究"
      eyebrow="RESEARCH ONLY / NO CAPITAL"
      empty="尚无配对研究。"
    >
      <p className="muted">
        只保存不可变关系研究与离线回测；不得批准、创建持久模拟账户、
        进入账户资本分配或发布个人建议。
      </p>
      <p className="muted">
        {researchRoutes.strategy} · {researchRoutes.backtest}
      </p>
    </ResourcePanel>
  );
}
