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

test("ships the Qlib and RD-Agent single-mainline interface", async () => {
  const appRoot = new URL("../app/", import.meta.url);
  const files = (await readdir(appRoot)).filter((name) => /\.(tsx|ts)$/.test(name));
  const sources = await Promise.all(
    files.map(async (name) => [name, await readFile(new URL(name, appRoot), "utf8")]),
  );
  const sourceByName = Object.fromEntries(sources);
  const allSource = sources.map(([, source]) => source).join("\n");

  assert.match(sourceByName["api-client.ts"], /credentials:\s*"include"/);
  assert.match(sourceByName["api-client.ts"], /inflightGets/);
  assert.match(sourceByName["api-client.ts"], /AbortController/);
  assert.match(sourceByName["api-client.ts"], /sessionStorage/);
  assert.match(sourceByName["api-client.ts"], /staleMs/);
  assert.match(sourceByName["api-client.ts"], /clearApiCache/);
  assert.match(sourceByName["use-polling.ts"], /finally[\s\S]*setTimeout/);
  assert.doesNotMatch(allSource, /setInterval/);
  assert.match(sourceByName["page.tsx"], /\/api\/auth\/state/);
  const coreBatchStart = sourceByName["page.tsx"].indexOf("Promise.allSettled([");
  const coreBatchEnd = sourceByName["page.tsx"].indexOf("]);", coreBatchStart);
  assert.ok(coreBatchStart >= 0 && coreBatchEnd > coreBatchStart);
  assert.doesNotMatch(sourceByName["page.tsx"].slice(coreBatchStart, coreBatchEnd), /data-retention/);
  assert.match(sourceByName["auth-panel.tsx"], /bootstrap.*login|login.*bootstrap/s);
  assert.match(sourceByName["page.tsx"], /StrategyAllocationPanel/);
  assert.match(sourceByName["page.tsx"], /PairSatellitePanel/);
  assert.doesNotMatch(sourceByName["page.tsx"], /<section hidden>/);
  assert.match(sourceByName["page.tsx"], /数据快照/);
  assert.match(sourceByName["page.tsx"], /因子准入/);
  assert.match(sourceByName["page.tsx"], /Qlib 回测与审批/);
  assert.match(sourceByName["page.tsx"], /核心 \/ 卫星分配/);
  assert.match(sourceByName["page.tsx"], /统一模拟盘/);
  assert.doesNotMatch(sourceByName["page.tsx"], /PairTradingPanel/);

  const portfolio = sourceByName["portfolio-panel.tsx"];
  assert.match(portfolio, /\/api\/recommendation-portfolios/);
  assert.match(portfolio, /RECOMMENDATION TRACKING/);
  assert.match(portfolio, /UNIFIED SIMULATION LEDGER/);
  assert.match(portfolio, /\/api\/simulation-portfolios/);
  assert.match(portfolio, /source_type/);
  assert.match(portfolio, /strategy_version/);
  assert.match(portfolio, /allocation/);
  assert.match(portfolio, /execution_frequency/);
  assert.doesNotMatch(portfolio, /hypothetical_performance/);
  assert.match(portfolio, /不产生订单、成交或券商指令/);

  const pair = sourceByName["pair-satellite-panel.tsx"];
  assert.match(pair, /\/api\/pair-strategies/);
  assert.match(pair, /\/pair-backtests/);
  assert.match(pair, /不得批准、创建持久模拟账户/);
  assert.doesNotMatch(
    pair,
    /\/approve|\/api\/simulation-portfolios|\/api\/pair-portfolios|pair_paper|paper[_ -]trading/i,
  );

  const allocation = sourceByName["strategy-allocation-panel.tsx"];
  assert.match(allocation, /\/api\/strategy-allocations/);
  assert.match(allocation, /risk_parity/);
  assert.match(allocation, /role/);
  assert.match(allocation, /risk_budget/);
  assert.match(allocation, /member_cap/);
  assert.match(allocation, /核心 \/ 卫星/);
  assert.match(allocation, /max_drawdown_liquidate/);
  assert.match(allocation, /recommendation_portfolio_id/);
  assert.match(allocation, /推荐组合自动刷新/);
  assert.match(allocation, /schedule\/status/);
  assert.doesNotMatch(allocation, /member\.portfolio_id/);
  assert.doesNotMatch(allocation, /模拟滑点/);

  assert.match(sourceByName["rdagent-panel.tsx"], /\/api\/strategy-recipes/);
  assert.match(sourceByName["backtest-panel.tsx"], /full_market_multifactor|文档策略配方/);
  assert.match(sourceByName["backtest-panel.tsx"], /industry_neutral_qp/);
  assert.match(sourceByName["backtest-panel.tsx"], /execution_dataset/);
  assert.match(sourceByName["backtest-panel.tsx"], /执行契约哈希/);
  assert.match(sourceByName["research-campaign-panel.tsx"], /\/api\/research-programs/);
  assert.match(sourceByName["market-overview-panel.tsx"], /\/api\/market\/overview/);
  assert.match(sourceByName["job-run-center.tsx"], /\/api\/jobs\/\$\{job\.id\}\/log/);
  assert.match(sourceByName["page.tsx"], /目录就绪度/);
  assert.match(sourceByName["page.tsx"], /成功 checkpoint/);
  assert.match(sourceByName["data-task-center.tsx"], /请求策略/);
  assert.match(sourceByName["job-run-center.tsx"], /job-progress-card/);
  assert.match(sourceByName["data-progress.ts"], /adaptive_recovery:\s*"自适应拆分恢复"/);
  assert.doesNotMatch(allSource, /SQLite/i);
  assert.doesNotMatch(allSource, /\/api\/broker|\/api\/pair-portfolios|settings\/broker/i);

  const rawFetchFiles = sources
    .filter(([, source]) => /\bfetch\(/.test(source))
    .map(([name]) => name);
  assert.deepEqual(rawFetchFiles, ["api-client.ts"]);
});
