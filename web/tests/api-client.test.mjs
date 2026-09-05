import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL("../app/api-client.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
let moduleId = 0;

async function client(t, fetcher) {
  const oldWindow = globalThis.window;
  const oldFetch = globalThis.fetch;
  const entries = new Map();
  globalThis.window = {
    location: { origin: "http://quantlab.test" },
    setTimeout, clearTimeout,
    sessionStorage: {
      get length() { return entries.size; },
      key: (index) => [...entries.keys()][index] ?? null,
      getItem: (key) => entries.get(key) ?? null,
      setItem: (key, value) => entries.set(key, value),
      removeItem: (key) => entries.delete(key),
    },
  };
  globalThis.fetch = fetcher;
  t.after(() => { globalThis.window = oldWindow; globalThis.fetch = oldFetch; });
  const importedClient = await import(`data:text/javascript;base64,${Buffer.from(`${compiled}\n// ${moduleId++}`).toString("base64")}`);
  return { ...importedClient, entries };
}

test("live research state ignores a previously cached running result", async (t) => {
  let calls = 0;
  const { apiFetch } = await client(t, async () => {
    calls += 1;
    return Response.json({ status: calls === 1 ? "running" : "failed" });
  });
  assert.equal((await (await apiFetch("/api/rdagent/runs")).json()).status, "running");
  const live = await apiFetch("/api/rdagent/runs", { cache: "no-store" });
  assert.equal((await live.json()).status, "failed");
  assert.equal(live.headers.get("X-QuantLab-Cache"), "network");
  assert.equal(calls, 2);
});

test("live data responses are not persisted and follow the next server result", async (t) => {
  let count = 0;
  const { apiFetch, entries } = await client(t, async () => Response.json({ count: ++count }));
  assert.equal((await (await apiFetch("/api/data-tasks", { cache: "no-store" })).json()).count, 1);
  assert.equal((await (await apiFetch("/api/data-tasks", { cache: "no-store" })).json()).count, 2);
  assert.equal([...entries.keys()].some((key) => key.includes("/api/data-tasks")), false);
});

test("live refresh failures reach the panel instead of returning old success", async (t) => {
  let fail = false;
  const { apiFetch } = await client(t, async () => {
    if (fail) throw new Error("offline");
    return Response.json({ status: "running" });
  });
  await apiFetch("/api/jobs");
  fail = true;
  await assert.rejects(apiFetch("/api/jobs", { cache: "no-store" }), /offline/);
  await assert.rejects(apiFetch("/api/jobs", { forceRefresh: true }), /offline/);
});

test("concurrent live panels share a request but a later refresh makes a new request", async (t) => {
  let calls = 0;
  let release;
  const { apiFetch } = await client(t, async () => {
    calls += 1;
    await new Promise((resolve) => { release = resolve; });
    return Response.json({ status: "running" });
  });
  const first = apiFetch("/api/jobs", { cache: "no-store" });
  const second = apiFetch("/api/jobs", { cache: "no-store" });
  assert.equal(calls, 1);
  release();
  const responses = await Promise.all([first, second]);
  assert.deepEqual(await Promise.all(responses.map((response) => response.json())), [
    { status: "running" }, { status: "running" },
  ]);
  const next = apiFetch("/api/jobs", { cache: "no-store" });
  assert.equal(calls, 2);
  release();
  await next;
});

test("no-store supplied through a Request also bypasses application cache", async (t) => {
  let calls = 0;
  const { apiFetch } = await client(t, async () => Response.json({ version: ++calls }));
  const url = "http://quantlab.test/api/rdagent/runs";
  await apiFetch(url);
  const response = await apiFetch(new Request(url, { cache: "no-store" }));
  assert.equal((await response.json()).version, 2);
});

test("non-live cache use remains available for intentionally cached resources", async (t) => {
  let calls = 0;
  const { apiFetch } = await client(t, async () => Response.json({ version: ++calls }));
  await apiFetch("/api/strategy-recipes");
  const response = await apiFetch("/api/strategy-recipes");
  assert.equal((await response.json()).version, 1);
  assert.equal(response.headers.get("X-QuantLab-Cache"), "fresh");
  assert.equal(calls, 1);
});
