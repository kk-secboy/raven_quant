import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the QuantLab authenticated application shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html lang="zh-CN">/i);
  assert.match(html, /<title>QuantLab · 量化研究系统<\/title>/i);
  assert.match(html, /基于 Tushare、Qlib 与 RD-Agent 的本地量化研究平台/);
  assert.match(html, /正在检查安全会话/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|SQLite/i);
});

test("ships product authentication and one credentialed API client", async () => {
  const appRoot = new URL("../app/", import.meta.url);
  const files = (await readdir(appRoot)).filter((name) => /\.(tsx|ts)$/.test(name));
  const sources = await Promise.all(
    files.map(async (name) => [name, await readFile(new URL(name, appRoot), "utf8")]),
  );
  const sourceByName = Object.fromEntries(sources);
  const allSource = sources.map(([, source]) => source).join("\n");
  const packageJson = await readFile(new URL("../package.json", import.meta.url), "utf8");

  assert.match(sourceByName["api-client.ts"], /credentials:\s*"include"/);
  assert.match(sourceByName["page.tsx"], /\/api\/auth\/state/);
  assert.match(sourceByName["auth-panel.tsx"], /bootstrap.*login|login.*bootstrap/s);
  assert.match(sourceByName["operations-panel.tsx"], /\/api\/auth\/users/);
  assert.match(sourceByName["operations-panel.tsx"], /\/api\/audit/);
  assert.match(sourceByName["operations-panel.tsx"], /\/api\/operations\/readiness/);
  assert.match(sourceByName["operations-panel.tsx"], /正式可用验收/);
  assert.match(sourceByName["operations-panel.tsx"], /期望状态/);
  assert.match(sourceByName["operations-panel.tsx"], /组合统一托管/);
  assert.match(sourceByName["operations-panel.tsx"], /在多策略组合中操作/);
  assert.match(sourceByName["settings-panel.tsx"], /\/api\/settings\/alerts/);
  assert.match(sourceByName["settings-panel.tsx"], /Scheduler 无需重启即可生效/);
  assert.match(sourceByName["settings-panel.tsx"], /\/api\/settings\/broker/);
  assert.match(sourceByName["settings-panel.tsx"], /BROKER_MODE.*部署级安全锁/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /\/api\/strategy-allocations/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /risk_parity/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /max_drawdown_liquidate/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /events\/\$\{item\.id\}\/\$\{action\}/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /完成处置/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /子策略自动调度/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /schedule\/status/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /期望状态和风控暂停分开保存/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /策略相关性上限/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /成员回撤熔断/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /组合回撤减仓/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /组合回撤清仓/);
  assert.match(sourceByName["strategy-allocation-panel.tsx"], /错过宽限/);
  assert.match(sourceByName["portfolio-panel.tsx"], /risk-events\/\$\{item\.id\}\/\$\{action\}/);
  assert.match(sourceByName["portfolio-panel.tsx"], /恢复组合仍需单独操作/);
  assert.match(sourceByName["portfolio-panel.tsx"], /latest_compatible/);
  assert.match(sourceByName["portfolio-panel.tsx"], /追加式验证的新 Qlib 快照/);
  assert.match(sourceByName["page.tsx"], /PairTradingPanel/);
  assert.match(sourceByName["page.tsx"], /MarketOverviewPanel/);
  assert.match(sourceByName["market-overview-panel.tsx"], /\/api\/market\/overview/);
  assert.match(sourceByName["market-overview-panel.tsx"], /不可变快照 · 非实时行情/);
  assert.match(sourceByName["market-overview-panel.tsx"], /自选与策略观察池/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /\/api\/pair-strategies/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /\/pair-backtests/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /pair_research/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /pair_paper/);
  assert.match(sourceByName["data-task-center.tsx"], /\/api\/jobs\/margin-eligibility/);
  assert.match(sourceByName["data-task-center.tsx"], /\/api\/jobs\/core-intraday/);
  assert.match(sourceByName["data-task-center.tsx"], /\/api\/jobs\/minute-qlib/);
  assert.match(sourceByName["data-task-center.tsx"], /失败只重试构建/);
  assert.match(sourceByName["qlib-panel.tsx"], /\/api\/jobs\/minute-research/);
  assert.match(sourceByName["qlib-panel.tsx"], /结果只进入研究记录，不自动晋级策略/);
  assert.match(sourceByName["data-task-center.tsx"], /510300\.SH,159919\.SZ/);
  assert.match(sourceByName["job-run-center.tsx"], /X-Total-Count|x-total-count/i);
  assert.match(sourceByName["job-run-center.tsx"], /\/api\/jobs\/\$\{job\.id\}\/log/);
  assert.match(sourceByName["job-run-center.tsx"], /按原参数重试/);
  assert.match(sourceByName["job-run-center.tsx"], /取消任务/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /\/api\/pair-portfolios/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /pair_paper_rebalance/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /第二人风险审批/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /专用双腿价差账本已接入/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /dataset_roll_policy/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /execution_roll_policy/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /追加式验证的新执行快照/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /容量、成本、Kalman 与回测准入参数/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /成交量参与率/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /最低容量成交率/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /对冲比下限/);
  assert.match(sourceByName["pair-trading-panel.tsx"], /滚动协整通过率/);
  assert.match(sourceByName["operations-panel.tsx"], /修订回看天数/);
  assert.match(sourceByName["operations-panel.tsx"], /快照起始日/);
  assert.match(sourceByName["operations-panel.tsx"], /最大切片数/);
  assert.match(sourceByName["operations-panel.tsx"], /成交量参与率上限/);
  assert.match(sourceByName["portfolio-panel.tsx"], /模拟滑点/);
  assert.match(sourceByName["rdagent-panel.tsx"], /\/api\/strategy-recipes/);
  assert.match(sourceByName["rdagent-panel.tsx"], /rdagent_research/);
  assert.match(sourceByName["rdagent-panel.tsx"], /\/api\/schedules/);
  assert.match(sourceByName["rdagent-panel.tsx"], /文档策略配方/);
  assert.match(sourceByName["research-campaign-panel.tsx"], /\/api\/research-programs/);
  assert.match(sourceByName["research-campaign-panel.tsx"], /持续自动研究/);
  assert.match(sourceByName["research-campaign-panel.tsx"], /人工审批/);
  assert.match(sourceByName["backtest-panel.tsx"], /不可变配方基线/);
  assert.match(sourceByName["backtest-panel.tsx"], /recipe_version/);
  assert.match(sourceByName["backtest-panel.tsx"], /默认风控模板/);
  assert.match(sourceByName["backtest-panel.tsx"], /单票止损/);
  assert.match(sourceByName["backtest-panel.tsx"], /首次止盈减仓（%）/);
  assert.match(sourceByName["backtest-panel.tsx"], /组合回撤清仓/);
  assert.match(sourceByName["backtest-panel.tsx"], /日成交量参与上限/);
  assert.match(sourceByName["backtest-panel.tsx"], /行业权重上限/);
  assert.match(sourceByName["backtest-panel.tsx"], /执行风控重放/);
  assert.doesNotMatch(sourceByName["strategy-allocation-panel.tsx"], /target_volatility:\s*0\.15/);
  assert.doesNotMatch(sourceByName["operations-panel.tsx"], /lookback_days:\s*7/);
  assert.doesNotMatch(sourceByName["operations-panel.tsx"], /max_slices:\s*24/);
  assert.doesNotMatch(sourceByName["portfolio-panel.tsx"], /slippage:\s*0\.0005/);
  assert.doesNotMatch(allSource, /SQLite/i);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);

  const rawFetchFiles = sources
    .filter(([, source]) => /\bfetch\(/.test(source))
    .map(([name]) => name);
  assert.deepEqual(rawFetchFiles, ["api-client.ts"]);
});
