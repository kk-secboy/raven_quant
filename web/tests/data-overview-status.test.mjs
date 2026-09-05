import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";
import { dataReadinessView } from "../app/data-overview-status.mjs";

const complete = {
  actionable_tasks: 33, ready_tasks: 33, partial_tasks: 0, failed_tasks: 0,
  running_tasks: 0, waiting_tasks: 0, terminal_failed_tasks: 0,
};

test("a lone failed data capability cannot make an otherwise idle catalogue ready", () => {
  const view = dataReadinessView({ ...complete, ready_tasks: 32, failed_tasks: 1, terminal_failed_tasks: 1 });
  assert.equal(view.ready, false);
  assert.equal(view.state, "failed");
  assert.equal(view.label, "有数据任务失败待处理");
});

test("only a confirmed nonempty complete catalogue is labelled ready", () => {
  assert.deepEqual(dataReadinessView(complete), { ready: true, state: "ready", label: "数据能力已就绪" });
  for (const counts of [
    { ...complete, actionable_tasks: 0, ready_tasks: 0 },
    { ...complete, ready_tasks: 32 },
    { ...complete, ready_tasks: 32, partial_tasks: 1 },
    { ...complete, ready_tasks: 32, waiting_tasks: 1 },
    { ...complete, ready_tasks: 32, running_tasks: 1 },
    { ...complete, failed_tasks: 1 },
    { ...complete, terminal_failed_tasks: 1 },
  ]) assert.equal(dataReadinessView(counts).ready, false);
});

test("missing, malformed or inconsistent counts remain unknown", () => {
  for (const counts of [
    {}, { ready_tasks: 33, actionable_tasks: 33 },
    { ...complete, actionable_tasks: -1 },
    { ...complete, ready_tasks: 34 },
    { ...complete, ready_tasks: Number.NaN },
    { ...complete, failed_tasks: "0" },
  ]) {
    const view = dataReadinessView(counts);
    assert.equal(view.ready, false);
    assert.equal(view.state, "unknown");
  }
  assert.equal(dataReadinessView(null).ready, false);
});

test("a refresh failure never reaffirms previously complete data as currently ready", () => {
  assert.equal(dataReadinessView(complete, { loading: true }).state, "loading");
  const stale = dataReadinessView(complete, { refreshFailed: true });
  assert.equal(stale.ready, false);
  assert.equal(stale.label, "显示上次读取状态");
});

test("running and incomplete catalogues describe their actual state", () => {
  const active = dataReadinessView({ ...complete, ready_tasks: 32, running_tasks: 1 });
  assert.equal(active.label, "数据任务正在执行");
  const partial = dataReadinessView({ ...complete, ready_tasks: 32, partial_tasks: 1 });
  assert.equal(partial.label, "仍有数据能力待准备");
});

const dataProgressSource = await readFile(new URL("../app/data-progress.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(dataProgressSource, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { jobDisplayName } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);

test("US and HK bundle requests sharing a snapshot still have distinct task names", () => {
  const shared = { snapshot_name: "same-full-snapshot", output_name: "same-output" };
  assert.equal(jobDisplayName({ kind: "supplemental_us_market", payload: { ...shared, bundle: "us_market" } }), "美股市场");
  assert.equal(jobDisplayName({ kind: "supplemental_hk_market", payload: { ...shared, bundle: "hk_market" } }), "港股市场");
});

test("legacy supplemental kinds remain identifiable without a payload bundle", () => {
  const shared = { snapshot_name: "same-full-snapshot" };
  assert.equal(jobDisplayName({ kind: "supplemental_us_market", payload: shared }), "美股市场");
  assert.equal(jobDisplayName({ kind: "supplemental_hk_market", payload: shared }), "港股市场");
});
