"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { apiFetch } from "./api-client";

type Job = {
  id: string;
  kind: string;
  status: string;
  payload: Record<string, unknown>;
  progress?: Record<string, unknown> | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  cancel_requested_at?: string | null;
  exit_code?: number | null;
  error?: string | null;
};

const statusOptions = [
  ["", "全部"],
  ["running", "运行中"],
  ["queued", "排队"],
  ["failed", "失败"],
  ["succeeded", "成功"],
  ["cancelled", "已取消"],
];

const statusText: Record<string, string> = {
  queued: "排队",
  running: "运行中",
  succeeded: "成功",
  failed: "失败",
  cancelled: "已取消",
};

function timeText(value?: string | null) {
  return value ? new Date(value).toLocaleString("zh-CN") : "—";
}

function displayName(job: Job) {
  return String(job.payload.output_name ?? job.payload.snapshot_name ?? job.payload.bundle ?? job.payload.profile ?? job.kind);
}

type Props = {
  api: string;
  canControl: boolean;
  onChanged: () => Promise<void>;
  onMessage: (message: string) => void;
};

export function JobRunCenter({ api, canControl, onChanged, onMessage }: Props) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState("");
  const [kind, setKind] = useState("");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Job | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const pageSize = 20;

  const load = useCallback(async () => {
    const params = new URLSearchParams({ limit: String(pageSize), offset: String(page * pageSize) });
    if (status) params.append("status", status);
    if (kind.trim()) params.append("kind", kind.trim());
    const response = await apiFetch(`${api}/api/jobs?${params}`, { cache: "no-store" });
    if (!response.ok) return;
    setJobs(await response.json());
    setTotal(Number(response.headers.get("x-total-count") ?? 0));
  }, [api, kind, page, status]);

  useEffect(() => {
    const initial = window.setTimeout(load, 0);
    const timer = window.setInterval(load, 5000);
    return () => { window.clearTimeout(initial); window.clearInterval(timer); };
  }, [load]);

  useEffect(() => setPage(0), [kind, status]);

  async function openJob(job: Job) {
    const [detailResponse, logResponse] = await Promise.all([
      apiFetch(`${api}/api/jobs/${job.id}`, { cache: "no-store" }),
      apiFetch(`${api}/api/jobs/${job.id}/log?tail=300`, { cache: "no-store" }),
    ]);
    if (!detailResponse.ok || !logResponse.ok) { onMessage("任务详情读取失败。"); return; }
    setSelected(await detailResponse.json());
    setLogs((await logResponse.json()).lines ?? []);
  }

  async function action(job: Job, name: "retry" | "cancel") {
    setBusy(true);
    try {
      const response = await apiFetch(`${api}/api/jobs/${job.id}/${name}`, { method: "POST" });
      const body = await response.json();
      if (!response.ok) { onMessage(body.detail ?? "任务操作失败"); return; }
      onMessage(name === "retry" ? "任务已按原参数重新排队。" : body.status === "cancelled" ? "排队任务已取消。" : "已请求 Worker 安全停止任务。");
      await load();
      await onChanged();
      await openJob(body);
    } finally {
      setBusy(false);
    }
  }

  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const kindsOnPage = useMemo(() => [...new Set(jobs.map((item) => item.kind))].sort(), [jobs]);

  return <section className="job-run-center">
    <div className="job-run-toolbar">
      <div><h2>运行记录</h2><p>筛选、分页、查看错误与日志；下载和 Qlib 构建可单独重试。</p></div>
      <div className="job-run-filters">
        <select aria-label="任务状态" value={status} onChange={(event) => setStatus(event.target.value)}>{statusOptions.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select>
        <input aria-label="任务类型" list="job-kinds" value={kind} onChange={(event) => setKind(event.target.value)} placeholder="全部任务类型" />
        <datalist id="job-kinds">{kindsOnPage.map((value) => <option value={value} key={value} />)}</datalist>
        <button type="button" onClick={load}>刷新</button>
      </div>
    </div>
    <div className={`job-run-layout ${selected ? "with-detail" : ""}`}>
      <div className="job-run-list">
        {jobs.map((job) => <button type="button" className={selected?.id === job.id ? "selected" : ""} onClick={() => openJob(job)} key={job.id}>
          <span className={`job-state ${job.status}`} />
          <div><strong>{displayName(job)}</strong><small>{job.kind} · {timeText(job.created_at)}</small>{job.error ? <em>{job.error}</em> : null}</div>
          <code>{job.id.slice(0, 10)}</code>
          <span className={`task-status ${job.status}`}>{statusText[job.status] ?? job.status}</span>
        </button>)}
        {!jobs.length ? <div className="empty compact">当前筛选条件下没有任务。</div> : null}
        <footer><span>共 {total} 条 · 第 {page + 1}/{pageCount} 页</span><div><button type="button" disabled={page === 0} onClick={() => setPage(page - 1)}>上一页</button><button type="button" disabled={page + 1 >= pageCount} onClick={() => setPage(page + 1)}>下一页</button></div></footer>
      </div>
      {selected ? <aside className="job-run-detail">
        <header><div><span>{selected.kind}</span><h3>{displayName(selected)}</h3></div><button type="button" onClick={() => setSelected(null)}>关闭</button></header>
        <dl><div><dt>任务 ID</dt><dd><code>{selected.id}</code></dd></div><div><dt>状态</dt><dd>{statusText[selected.status] ?? selected.status}{selected.cancel_requested_at && selected.status === "running" ? " · 正在安全停止" : ""}</dd></div><div><dt>开始 / 结束</dt><dd>{timeText(selected.started_at)} / {timeText(selected.finished_at)}</dd></div><div><dt>退出码</dt><dd>{selected.exit_code ?? "—"}</dd></div></dl>
        {selected.error ? <div className="job-run-error"><strong>失败原因</strong><p>{selected.error}</p></div> : null}
        <details><summary>任务参数</summary><pre>{JSON.stringify(selected.payload, null, 2)}</pre></details>
        <div className="job-log-head"><strong>最近日志</strong><button type="button" onClick={() => openJob(selected)}>刷新日志</button></div>
        <pre className="job-log">{logs.length ? logs.join("\n") : "尚无日志输出。"}</pre>
        {canControl ? <div className="job-run-actions">{["failed", "cancelled"].includes(selected.status) ? <button type="button" disabled={busy} onClick={() => action(selected, "retry")}>按原参数重试</button> : null}{["queued", "running"].includes(selected.status) ? <button type="button" className="danger-button" disabled={busy || !!selected.cancel_requested_at} onClick={() => action(selected, "cancel")}>{selected.cancel_requested_at ? "正在安全停止" : "取消任务"}</button> : null}</div> : null}
      </aside> : null}
    </div>
  </section>;
}
