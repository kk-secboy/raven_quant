import assert from "node:assert/strict";
import test from "node:test";
import {
  failedResearchPage,
  groupResearchRuns,
  readResearchPage,
  receivedResearchPage,
  researchRefreshMessage,
  researchRunView,
  researchTime,
} from "../app/research-run-presentation.mjs";

const createdAt = "2026-09-05T17:34:24Z";
const phaseAt = "2026-09-05T17:37:39Z";
const freshAt = "2026-09-05T18:00:00Z";
const run = (id, status, extra = {}) => ({ id, status, created_at: createdAt, ...extra });
const response = (rows, total = rows.length, extraHeaders = {}) => new Response(JSON.stringify(rows), {
  headers: { "Content-Type": "application/json", "X-Total-Count": String(total), ...extraHeaders },
});

test("current and history retain every exact run even when objectives match", () => {
  const runs = [
    run("new", "running", { objective: "same objective" }),
    run("old", "failed", { objective: "same objective" }),
    run("queued", "queued"), run("evaluation", "evaluating"), run("export", "exporting"),
    run("cancelled", "cancelled"), run("done", "succeeded"), run("blocked", "blocked"),
  ];
  const before = structuredClone(runs);
  const { active, history } = groupResearchRuns(runs);
  assert.deepEqual(active.map((item) => item.id), ["new", "queued", "evaluation", "export"]);
  assert.deepEqual(history.map((item) => item.id), ["old", "cancelled", "done", "blocked"]);
  assert.deepEqual(runs, before);
  assert.equal(researchRunView(history[0], { historical: true }).label, "历史失败");
});

test("explicit interruption changes only presentation and keeps original failed audit status", () => {
  const old = run("cancelled-during-release", "failed", { presentation: {
    display_status: "interrupted", execution_phase: "policy_only",
    reason_code: "operator_cancelled", safe_reason: "该次执行已由操作人员中止。",
  } });
  const before = structuredClone(old);
  const view = researchRunView(old, { historical: true });
  assert.equal(view.label, "历史中止");
  assert.equal(view.originalStatus, "failed");
  assert.equal(view.safeReason, old.presentation.safe_reason);
  assert.equal(view.reasonCode, "operator_cancelled");
  assert.equal(view.tone, "warning");
  assert.deepEqual(old, before);
});

test("raw error text never infers interruption, recovery, or a safe reason", () => {
  const view = researchRunView(run("old", "failed", {
    error: "Cancelled by operator; token=secret; recovered by new similar objective",
    objective: "same objective",
  }), { historical: true });
  assert.equal(view.label, "历史失败");
  assert.equal(view.safeReason, null);
  assert.equal(view.reasonCode, null);
  assert.doesNotMatch(JSON.stringify(view), /token|secret|recovered/);
});

test("research phase and task identity come from the safe exact-linked projection", () => {
  const value = run("research", "running", {
    job_id: "original-proposal-job", started_at: phaseAt,
    presentation: { display_status: "running", execution_phase: "policy_only", phase_label: "策略规则验证" },
    linked_execution: { job_id: "actual-parameter-job", kind: "parameter_experiment", status: "running", updated_at: phaseAt },
  });
  const view = researchRunView(value);
  assert.equal(view.jobId, "actual-parameter-job");
  assert.equal(view.phaseLabel, "策略规则验证");
  assert.equal(view.createdAt, researchTime(createdAt));
  assert.equal(view.phaseUpdatedAt, researchTime(phaseAt));
  assert.notEqual(view.createdAt, view.phaseUpdatedAt);
});

test("a queued successor stays queued and a completed child does not finish its research", () => {
  const queued = researchRunView(run("research", "running", {
    presentation: { display_status: "queued", execution_phase: "policy_only" },
  }));
  assert.equal(queued.label, "等待执行");
  const settling = researchRunView(run("research", "running", {
    presentation: { display_status: "running", execution_phase: "settlement" },
    linked_execution: { job_id: "done-child", kind: "parameter_experiment", status: "succeeded" },
  }));
  assert.equal(settling.label, "运行中");
  assert.equal(settling.phaseLabel, "研究结算");
});

test("confirmed partial trial failure stays active with its warning and safe count summary", () => {
  const active = run("still-evaluating", "running", { presentation: {
    display_status: "running", label: "运行中（已有试验失败）", execution_phase: "policy_only",
    reason_code: "partial_trial_failure", safe_reason: "参数试验已完成 1/2 项，其中 1 项失败，其余仍在执行。",
  } });
  const view = researchRunView(active);
  assert.equal(view.displayStatus, "running");
  assert.equal(view.label, "运行中（已有试验失败）");
  assert.equal(view.tone, "warning");
  assert.equal(view.safeReason, active.presentation.safe_reason);
  assert.deepEqual(groupResearchRuns([active]).active, [active]);
  const stale = researchRunView(active, { stale: true });
  assert.equal(stale.label, "上次状态：运行中（已有试验失败）");
  assert.equal(stale.tone, "muted");
});

test("recent terminal results show actual finish time without relabelling a new failure as history", () => {
  const latest = run("just-failed", "failed", { finished_at: phaseAt });
  const recent = researchRunView(latest);
  assert.equal(recent.label, "执行失败");
  assert.equal(recent.finishedAt, researchTime(phaseAt));
  assert.equal(recent.createdAt, researchTime(createdAt));
  assert.equal(researchRunView(latest, { historical: true }).label, "历史失败");
  assert.equal(researchRunView(run("legacy", "failed")).finishedAt, "未提供");
});

test("registered generic evaluation and rejection have accurate labels", () => {
  const evaluating = researchRunView(run("generic", "evaluating", {
    presentation: { display_status: "running", execution_phase: "evaluation" },
  }));
  assert.equal(evaluating.phaseLabel, "独立评估");
  const blocked = researchRunView(run("blocked", "blocked", {
    presentation: { display_status: "blocked", execution_phase: "full_stack" },
  }), { historical: true });
  assert.equal(blocked.label, "历史未通过");
  assert.equal(blocked.tone, "warning");
});

test("unknown phases never claim trial, IS, or OOS progress", () => {
  const view = researchRunView(run("research", "running", {
    progress: { phase: "trial_000_oos" },
    presentation: { display_status: "running", execution_phase: "unregistered", phase_label: "OOS 100%" },
  }));
  assert.equal(view.phaseLabel, "阶段待确认");
  assert.equal(view.jobId, null);
});

test("filtered pagination keeps actual server total and exact page identities", async () => {
  const page = await readResearchPage(response([run("old-21", "failed")], 51), "history", 1);
  assert.equal(page.total, 51);
  assert.equal(page.page, 1);
  assert.equal(page.rows[0].id, "old-21");
  assert.equal(page.cacheState, null);
});

test("malformed or old unfiltered API responses cannot mix historical failures into current research", async () => {
  await assert.rejects(readResearchPage(response([run("active", "running"), run("old", "failed")]), "active", 0));
  await assert.rejects(readResearchPage(response([run("active", "running")]), "history", 0));
  await assert.rejects(readResearchPage(response([{ id: "missing-status" }]), "history", 0));
  await assert.rejects(readResearchPage(response([run("old", "failed")], -1), "history", 0));
  await assert.rejects(readResearchPage(new Response("unavailable", { status: 503 }), "active", 0));
});

test("failed refresh preserves visible rows and loaded page while clearly invalidating freshness", () => {
  const previous = { rows: [run("active", "running")], total: 101, page: 1, state: "ready", updatedAt: freshAt };
  const failed = failedResearchPage(previous);
  assert.equal(failed.rows, previous.rows);
  assert.equal(failed.page, 1);
  assert.equal(failed.updatedAt, freshAt);
  assert.equal(previous.state, "ready");
  assert.match(researchRefreshMessage(failed), /刷新失败.*当前状态暂时无法确认.*上次成功刷新/);
  const view = researchRunView(failed.rows[0], { stale: failed.state !== "ready" });
  assert.equal(view.label, "上次状态：运行中");
  assert.equal(view.tone, "muted");
});

test("first-read failure is unknown, while a successful empty read confirms no active research", () => {
  const initial = { rows: [], total: 0, page: 0, state: "loading", updatedAt: null };
  const failed = failedResearchPage(initial);
  assert.match(researchRefreshMessage(failed), /尚未成功读取/);
  assert.equal(failed.updatedAt, null);
  const ready = receivedResearchPage(failed, { rows: [], total: 0, page: 0, cacheState: null }, freshAt);
  assert.equal(ready.state, "ready");
  assert.equal(ready.total, 0);
  assert.equal(ready.updatedAt, freshAt);
  assert.match(researchRefreshMessage(ready), /^最近刷新/);
});

test("stale cache data never advances successful refresh time and fresh recovery replaces old status", () => {
  const previous = { rows: [run("active", "running")], total: 1, state: "ready", updatedAt: freshAt };
  const cached = receivedResearchPage(previous, { rows: previous.rows, total: 1, cacheState: "stale" }, phaseAt);
  assert.equal(cached.state, "stale");
  assert.equal(cached.updatedAt, freshAt);
  assert.match(researchRefreshMessage(cached), /上次读取.*可能已变化/);
  const nextAt = "2026-09-05T18:00:08Z";
  const recovered = receivedResearchPage(cached, { rows: [], total: 0, cacheState: null }, nextAt);
  assert.equal(recovered.state, "ready");
  assert.equal(recovered.updatedAt, nextAt);
  assert.deepEqual(recovered.rows, []);
});
