import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";
import {
  factorSourceSelectionIsValid,
  recipeUsesQlibBaseline,
  visibleStrategyCreationRecipes,
} from "../app/strategy-recipe-policy.mjs";

test("strategy creation exposes only governed recipes and binds transparent baselines", () => {
  const ids = [
    "short_relative_strength",
    "swing_trend",
    "long_quality_value",
    "index_enhancement",
    "full_market_multifactor",
    "minute_mean_reversion",
    "pair_trading",
  ];
  const recipes = ids.map((id) => ({
    id,
    config_overrides: { factor_source_mode: "qlib_baseline" },
  }));

  assert.deepEqual(
    visibleStrategyCreationRecipes(recipes).map((recipe) => recipe.id),
    ids.slice(0, 5),
  );
  for (const id of ids.slice(0, 3)) {
    const recipe = recipes.find((item) => item.id === id);
    assert.equal(recipeUsesQlibBaseline(recipe), true);
    assert.equal(factorSourceSelectionIsValid(recipe, "qlib_baseline", 0, 0), true);
  }
  assert.equal(
    recipeUsesQlibBaseline(recipes.find((recipe) => recipe.id === "minute_mean_reversion")),
    false,
  );
  assert.equal(
    recipeUsesQlibBaseline(recipes.find((recipe) => recipe.id === "pair_trading")),
    false,
  );
  assert.equal(
    factorSourceSelectionIsValid(
      recipes.find((recipe) => recipe.id === "minute_mean_reversion"),
      "qlib_baseline",
      0,
      0,
    ),
    false,
  );
});

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
  assert.match(html, /基于 Tushare、Qlib 与 RD-Agent 的受控量化研究平台/);
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
  assert.match(sourceByName["api-client.ts"], /CACHE_SCHEMA_VERSION\s*=\s*"api-response-v2"/);
  assert.match(sourceByName["api-client.ts"], /NEXT_PUBLIC_CACHE_RELEASE/);
  assert.match(sourceByName["api-client.ts"], /SESSION_VERSION_KEY/);
  assert.match(sourceByName["api-client.ts"], /startsWith\(SESSION_ROOT_PREFIX\)/);
  assert.match(sourceByName["api-client.ts"], /function apiFetch[\s\S]{0,180}prepareSessionCache\(\)/);
  assert.match(
    sourceByName["api-client.ts"],
    /path === "\/api\/qlib\/status" \|\| path === "\/api\/rdagent\/status"[\s\S]{0,240}staleMs:\s*10_000,[\s\S]{0,80}persist:\s*false/,
  );
  assert.match(sourceByName["use-polling.ts"], /finally[\s\S]*setTimeout/);
  assert.doesNotMatch(allSource, /setInterval/);
  assert.match(sourceByName["page.tsx"], /\/api\/auth\/state/);
  assert.match(sourceByName["page.tsx"], /if \(activeNav !== 1\) return;/);
  assert.match(sourceByName["page.tsx"], /activeNav === 1 \? <button[\s\S]{0,160}>刷新概况<\/button> : null/);
  const coreBatchStart = sourceByName["page.tsx"].indexOf("Promise.allSettled([");
  const coreBatchEnd = sourceByName["page.tsx"].indexOf("]);", coreBatchStart);
  assert.ok(coreBatchStart >= 0 && coreBatchEnd > coreBatchStart);
  assert.doesNotMatch(sourceByName["page.tsx"].slice(coreBatchStart, coreBatchEnd), /data-retention/);
  assert.match(sourceByName["auth-panel.tsx"], /bootstrap.*login|login.*bootstrap/s);
  assert.match(sourceByName["page.tsx"], /StrategyAllocationPanel/);
  assert.doesNotMatch(sourceByName["page.tsx"], /PairSatellitePanel|配对卫星/);
  assert.doesNotMatch(sourceByName["page.tsx"], /<section hidden>/);
  assert.match(sourceByName["page.tsx"], /数据快照/);
  assert.match(sourceByName["page.tsx"], /因子库与准入/);
  assert.match(sourceByName["page.tsx"], /模型竞赛与试验/);
  assert.match(sourceByName["page.tsx"], /Qlib 回测与审批/);
  assert.match(sourceByName["page.tsx"], /核心 \/ 卫星分配/);
  assert.match(sourceByName["page.tsx"], /统一模拟盘/);
  assert.doesNotMatch(sourceByName["page.tsx"], /PairTradingPanel/);
  assert.doesNotMatch(sourceByName["page.tsx"], /ResearchCampaignPanel|连续研究/);
  assert.doesNotMatch(allSource, /\/api\/research-programs|\/api\/research-campaigns/);

  const autopilot = sourceByName["autopilot-panel.tsx"];
  assert.match(autopilot, /\/api\/advice\/today/);
  assert.match(autopilot, /\/api\/investor-profile/);
  assert.match(autopilot, /short_1_5d/);
  assert.match(autopilot, /swing_1_6m/);
  assert.match(autopilot, /long_1_3y/);
  assert.match(autopilot, /BUY/);
  assert.match(autopilot, /ADD/);
  assert.match(autopilot, /HOLD/);
  assert.match(autopilot, /REDUCE/);
  assert.match(autopilot, /EXIT/);
  assert.match(autopilot, /NO_ACTION/);
  assert.match(autopilot, /is_investment_advice/);
  assert.match(autopilot, /const visibleAction = card\.is_investment_advice \? card\.action : "NO_ACTION"/);
  assert.match(autopilot, /simulationOnly = !card\.is_investment_advice/);
  assert.match(autopilot, /仅供隔离模拟验证，不是荐股/);
  assert.match(autopilot, /今天的正式动作/);
  assert.match(autopilot, /统一账户建议/);
  assert.match(autopilot, /今日唯一操作清单/);
  assert.match(autopilot, /统一账户今日操作清单/);
  assert.match(autopilot, /查看三周期来源与研究详情/);
  assert.match(autopilot, /多周期 ·/);
  assert.match(autopilot, /本次数量/);
  assert.match(autopilot, /账户目标/);
  assert.match(autopilot, /失效条件/);
  assert.match(autopilot, /executionConstraintReasons\(record\)/);
  assert.match(autopilot, /blocked_reasons/);
  assert.match(autopilot, /wait_reasons/);
  assert.match(autopilot, /sourceSignals = cards\.flatMap/);
  assert.match(autopilot, /card\.is_investment_advice/);
  assert.match(autopilot, /account=\{advice\?\.unified_account/);
  assert.match(autopilot, /cards=\{advice\?\.cards \?\? \[\]\}/);
  assert.match(autopilot, /new Set\(\[\.\.\.tradeByInstrument\.keys\(\), \.\.\.targetByInstrument\.keys\(\)\]\)/);
  assert.ok(
    autopilot.indexOf("<UnifiedAccountCard") < autopilot.indexOf('<details className="novice-source-details">'),
    "the novice surface must show the one account answer before collapsed horizon research details",
  );
  assert.match(autopilot, /main_board/);
  assert.match(autopilot, /star_market/);
  assert.match(autopilot, /chi_next/);
  assert.match(autopilot, /beijing_exchange/);
  assert.match(autopilot, /etf/);
  assert.match(autopilot, /initial_capital/);
  assert.match(autopilot, /为什么现在不能买/);
  assert.match(autopilot, /\["REDUCE", "EXIT"\]\.includes\(constraintAction\)/);
  assert.match(autopilot, /displayedAction = simulationOnly \? signal\.action : accountAction/);
  assert.match(autopilot, /ACTION_LABELS\[displayedAction\]/);
  assert.doesNotMatch(autopilot, /ACTION_LABELS\[signal\.action\]<\/span>/);
  assert.match(autopilot, /cannot_buy_reasons/);
  assert.match(autopilot, /仅表示真实前向运行资历/);
  assert.match(autopilot, /advancedMode/);
  assert.match(sourceByName["page.tsx"], /今日选股与账户操作/);
  assert.match(sourceByName["page.tsx"], /今日建议/);
  assert.match(sourceByName["page.tsx"], /打开高级管理/);
  assert.match(sourceByName["page.tsx"], /advancedMode=\{advancedMode\}/);
  assert.match(autopilot, /\/api\/autopilot/);
  assert.match(autopilot, /current_stage/);
  assert.match(autopilot, /next_action/);
  assert.match(autopilot, /tournament/);
  assert.match(autopilot, /因子 \/ 模型竞赛/);
  assert.match(autopilot, /模拟盘 NAV/);
  assert.match(autopilot, /每日候选/);
  assert.match(autopilot, /系统下一步/);

  const qlib = sourceByName["qlib-panel.tsx"];
  assert.match(qlib, /\/api\/autopilot\/cycles/);
  assert.match(qlib, /\/api\/model-ensembles/);
  assert.match(qlib, /预注册模型试验/);
  assert.match(qlib, /等权日度 Rank/);

  const portfolio = sourceByName["portfolio-panel.tsx"];
  assert.match(portfolio, /\/api\/recommendation-portfolios/);
  assert.match(portfolio, /AUTOMATIC RECOMMENDATION/);
  assert.match(portfolio, /UNIFIED SIMULATION LEDGER/);
  assert.match(portfolio, /\/api\/simulation-portfolios/);
  assert.match(portfolio, /source_type/);
  assert.match(portfolio, /strategy_version/);
  assert.match(portfolio, /allocation/);
  assert.match(portfolio, /execution_frequency/);
  assert.doesNotMatch(portfolio, /hypothetical_performance/);
  assert.match(portfolio, /不生成演示数据/);

  assert.equal(sourceByName["pair-satellite-panel.tsx"], undefined);

  const allocation = sourceByName["strategy-allocation-panel.tsx"];
  assert.match(allocation, /\/api\/strategy-allocations/);
  assert.match(allocation, /没有 50 万元硬门槛/);
  assert.doesNotMatch(allocation, /min="500000"/);
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
  assert.match(sourceByName["rdagent-panel.tsx"], /\/api\/rdagent\/status/);
  assert.match(sourceByName["rdagent-panel.tsx"], /fin_model/);
  assert.match(sourceByName["rdagent-panel.tsx"], /fin_quant/);
  assert.match(sourceByName["rdagent-panel.tsx"], /fin_strategy/);
  assert.match(sourceByName["rdagent-panel.tsx"], /研究周期（必选）/);
  assert.match(sourceByName["rdagent-panel.tsx"], /短线 · 1～5 个交易日/);
  assert.match(sourceByName["rdagent-panel.tsx"], /中线 · 1～6 个月/);
  assert.match(sourceByName["rdagent-panel.tsx"], /长线 · 1～3 年以上/);
  assert.equal(
    sourceByName["rdagent-panel.tsx"].match(
      /requiresResearchHorizon \? \{ horizon: researchHorizon \} : \{\}/g,
    )?.length,
    2,
    "immediate and scheduled fin_strategy requests must both bind the selected horizon",
  );
  assert.match(
    sourceByName["rdagent-panel.tsx"],
    /requiresResearchHorizon && !researchHorizon/,
  );
  assert.match(
    sourceByName["rdagent-panel.tsx"],
    /HORIZON_RESEARCH_SCENARIOS = new Set<ScenarioId>\(\[\s*"fin_quant",\s*"fin_strategy",?\s*\]\)/,
  );
  assert.match(sourceByName["rdagent-panel.tsx"], /fin_factor_report/);
  assert.match(sourceByName["rdagent-panel.tsx"], /general_model/);
  assert.match(sourceByName["rdagent-panel.tsx"], /data_science/);
  assert.match(sourceByName["rdagent-panel.tsx"], /llm_finetune/);
  assert.match(sourceByName["rdagent-panel.tsx"], /runtimeUnknown \? "状态未知"/);
  assert.match(sourceByName["rdagent-panel.tsx"], /运行时状态尚未确认/);
  assert.doesNotMatch(sourceByName["rdagent-panel.tsx"], /正在恢复|正在读取运行时状态/);
  assert.match(sourceByName["rdagent-panel.tsx"], /!runtimeOperational \|\| !selectedScenario\.ready/);
  assert.doesNotMatch(sourceByName["rdagent-panel.tsx"], /Docker 不可用|LLM 未配置/);
  const backtest = sourceByName["backtest-panel.tsx"];
  assert.match(backtest, /visibleStrategyCreationRecipes/);
  assert.match(backtest, /recipeUsesQlibBaseline/);
  assert.match(backtest, /factorSourceSelectionIsValid/);
  assert.match(backtest, /factorSourceMode === "qlib_baseline"[\s\S]{0,40}\? \[\]/);
  assert.match(backtest, /\.\.\.\(recipe\?\.config_overrides \?\? \{\}\)/);
  assert.match(backtest, /recipe_id: recipe\?\.id \?\? "custom"/);
  assert.match(backtest, /recipe_version: recipe\?\.version \?\? "custom"/);
  assert.match(backtest, /isQlibBaselineRecipe && factorSourceMode !== "qlib_baseline"/);
  assert.match(backtest, /固定 Qlib 基线/);
  assert.match(backtest, /不需要 RD-Agent 因子/);
  assert.match(backtest, /必须先完成回测才能审批/);
  assert.match(backtest, /\/api\/strategy-versions\/\$\{selectedVersion\}\/approve/);
  assert.doesNotMatch(backtest, /\/api\/advice|recommendation-portfolios|pair-portfolios/);
  assert.match(backtest, /full_market_multifactor|文档策略配方/);
  assert.match(backtest, /industry_neutral_qp/);
  assert.match(backtest, /execution_dataset/);
  assert.match(backtest, /执行契约哈希/);
  assert.match(backtest, /预最终历史 \/ 最终 OOS/);
  assert.match(sourceByName["rdagent-panel.tsx"], /最终 OOS 只开放一次/);
  assert.match(sourceByName["rdagent-panel.tsx"], /官方 RDLoop \/ Trace（只读）/);
  assert.match(sourceByName["rdagent-panel.tsx"], /Hypothesis 与 Feedback/);
  assert.doesNotMatch(
    sourceByName["rdagent-panel.tsx"],
    /2021-01-11|人工批准 \/ 模拟盘|正式回测后仍需人工批准/,
  );
  assert.match(
    sourceByName["strategy-defaults-panel.tsx"],
    /min_pre_final_history_days/,
  );
  assert.match(sourceByName["market-overview-panel.tsx"], /\/api\/market\/overview/);
  assert.match(sourceByName["market-overview-panel.tsx"], /function MarketOverviewSkeleton/);
  assert.match(
    sourceByName["market-overview-panel.tsx"],
    /market-source-bar[\s\S]*market-hero[\s\S]*market-stat-strip[\s\S]*market-grid[\s\S]*watchlist-card/,
  );
  assert.match(
    sourceByName["market-overview-panel.tsx"],
    /if \(loading && !market\)[\s\S]{0,120}return <MarketOverviewSkeleton \/>/,
  );
  const globalCss = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  assert.match(globalCss, /market-skeleton-pulse/);
  assert.match(globalCss, /prefers-reduced-motion:reduce/);
  assert.match(sourceByName["job-run-center.tsx"], /\/api\/jobs\/\$\{job\.id\}\/log/);
  assert.match(sourceByName["job-run-center.tsx"], /历史条数不是当前故障数/);
  assert.match(sourceByName["job-run-center.tsx"], /后续任务已成功/);
  assert.match(sourceByName["job-run-center.tsx"], /后续任务运行中/);
  assert.match(sourceByName["job-run-center.tsx"], /retry_successor/);
  assert.match(sourceByName["page.tsx"], /平均目录覆盖度/);
  assert.match(sourceByName["page.tsx"], /成功 checkpoint/);
  assert.match(sourceByName["page.tsx"], /running_work_units/);
  assert.match(sourceByName["page.tsx"], /liveExecutionPhase/);
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
