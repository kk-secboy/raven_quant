"use client";

import { useCallback, useMemo, useState } from "react";
import { apiFetch } from "./api-client";
import { DataJob, jobDisplayName, phaseLabel, targetText } from "./data-progress";
import { usePolling } from "./use-polling";

type Job = DataJob & {
  cancel_requested_at?: string | null;
  exit_code?: number | null;
  outcome_status?: "blocked" | "passed" | "rejected" | null;
  outcome_message?: string | null;
};

async function jsonResponse<T>(request: Promise<Response>): Promise<T> {
  const response = await request;
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

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
  blocked: "研究阻断",
  passed: "门禁通过",
  rejected: "未通过门禁",
};

function displayedOutcome(job: Job, laterStatus = "") {
  if (laterStatus) return { text: laterStatus, className: "succeeded" };
  if (job.outcome_status) {
    return {
      text: statusText[job.outcome_status] ?? job.outcome_status,
      className: job.outcome_status === "passed" ? "succeeded" : job.outcome_status,
    };
  }
  return { text: statusText[job.status] || job.status, className: job.status };
}

function timeText(value?: string | null) {
  return value ? new Date(value).toLocaleString("zh-CN") : "—";
}

function pipelineIdentity(job: Job) {
  const pipelineId = job.payload?.pipeline_id;
  if (typeof pipelineId === "string" && pipelineId) return `${job.kind}:pipeline:${pipelineId}`;
  const snapshotName = job.payload?.snapshot_name;
  if (typeof snapshotName === "string" && snapshotName) {
    return `${job.kind}:snapshot:${snapshotName}`;
  }
  return "";
}

function laterAttemptStatus(job: Job, jobs: Job[]) {
  if (job.status !== "failed") return "";
  if (job.retry_successor?.status === "succeeded") return "后续已成功";
  if (["queued", "running"].includes(job.retry_successor?.status ?? "")) {
    return "已重试，后续运行中";
  }
  const identity = pipelineIdentity(job);
  if (!identity) return "";
  const laterAttempts = jobs.filter((candidate) =>
    candidate.id !== job.id
    && pipelineIdentity(candidate) === identity
    && new Date(candidate.created_at).getTime() > new Date(job.created_at).getTime()
  );
  if (laterAttempts.some((candidate) => candidate.status === "succeeded")) {
    return "后续已成功";
  }
  if (laterAttempts.some((candidate) => ["queued", "running"].includes(candidate.status))) {
    return "已重试，后续运行中";
  }
  return "";
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
  const [logError, setLogError] = useState("");
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

  usePolling(load, 5000);

  async function openJob(job: Job) {
    const previousSelectionIsSame = selected?.id === job.id;
    const [detailResult, logResult] = await Promise.allSettled([
      jsonResponse<Job>(apiFetch(`${api}/api/jobs/${job.id}`, { cache: "no-store" })),
      jsonResponse<{ lines?: string[] }>(apiFetch(`${api}/api/jobs/${job.id}/log?tail=300`, { cache: "no-store" })),
    ]);
    if (detailResult.status === "fulfilled") setSelected(detailResult.value);
    if (logResult.status === "fulfilled") {
      if (detailResult.status === "fulfilled" || previousSelectionIsSame) {
        setLogs(logResult.value.lines ?? []);
        setLogError("");
      }
    } else if (detailResult.status === "fulfilled" || previousSelectionIsSame) {
      setLogError(previousSelectionIsSame ? "日志刷新失败，继续显示上次成功内容。" : "日志暂不可用。");
      if (!previousSelectionIsSame) setLogs([]);
    }

    if (detailResult.status === "rejected" && logResult.status === "rejected") {
      onMessage("任务详情和日志暂时都无法读取。");
    } else if (detailResult.status === "rejected") {
      onMessage("任务详情暂时无法读取。");
    } else if (logResult.status === "rejected") {
      onMessage("任务详情已载入，日志暂时无法读取。");
    }
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
  const selectedLaterStatus = selected ? laterAttemptStatus(selected, jobs) : "";

  return <section className="job-run-center">
    <div className="job-run-toolbar">
      <div><h2>运行记录</h2><p>完整保留历史审计；同一流水线的旧失败会标明已重试或后续已成功。</p></div>
      <div className="job-run-filters">
        <select aria-label="任务状态" value={status} onChange={(event) => { setStatus(event.target.value); setPage(0); }}>{statusOptions.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select>
        <input aria-label="任务类型" list="job-kinds" value={kind} onChange={(event) => { setKind(event.target.value); setPage(0); }} placeholder="全部任务类型" />
        <datalist id="job-kinds">{kindsOnPage.map((value) => <option value={value} key={value} />)}</datalist>
        <button type="button" onClick={load}>刷新</button>
      </div>
    </div>
    <div className={`job-run-layout ${selected ? "with-detail" : ""}`}>
      <div className="job-run-list">
        {jobs.map((job) => {
          const laterStatus = laterAttemptStatus(job, jobs);
          const outcome = displayedOutcome(job, laterStatus);
          return <button type="button" className={selected?.id === job.id ? "selected" : ""} onClick={() => openJob(job)} key={job.id}>
          <span className={`job-state ${job.status}`} />
          <div><strong>{jobDisplayName(job)}</strong><small>{phaseLabel(job.progress?.execution_phase ?? (job.status === "running" ? "planning" : job.status === "queued" ? "queued" : null))} · {targetText(job.payload, job.progress)} · {timeText(job.created_at)}</small>{job.error ? <em>{job.error}</em> : null}</div>
          <code>{job.id.slice(0, 10)}</code>
          <span className={`task-status ${outcome.className}`}>{outcome.text}</span>
        </button>;
        })}
        {!jobs.length ? <div className="empty compact">当前筛选条件下没有任务。</div> : null}
        <footer><span>共 {total} 条 · 第 {page + 1}/{pageCount} 页</span><div><button type="button" disabled={page === 0} onClick={() => setPage(page - 1)}>上一页</button><button type="button" disabled={page + 1 >= pageCount} onClick={() => setPage(page + 1)}>下一页</button></div></footer>
      </div>
      {selected ? <aside className="job-run-detail">
        <header><div><span>{selected.kind}</span><h3>{jobDisplayName(selected)}</h3></div><button type="button" onClick={() => { setSelected(null); setLogError(""); }}>关闭</button></header>
        <dl><div><dt>任务 ID</dt><dd><code>{selected.id}</code></dd></div><div><dt>程序状态</dt><dd>{selectedLaterStatus || statusText[selected.status] || selected.status}{selected.cancel_requested_at && selected.status === "running" ? " · 正在安全停止" : ""}</dd></div>{selected.outcome_status ? <div><dt>研究结论</dt><dd>{statusText[selected.outcome_status] ?? selected.outcome_status}</dd></div> : null}<div><dt>开始 / 结束</dt><dd>{timeText(selected.started_at)} / {timeText(selected.finished_at)}</dd></div><div><dt>退出码</dt><dd>{selected.exit_code ?? "—"}</dd></div></dl>
        {selected.outcome_message ? <div className="job-run-error"><strong>研究结果</strong><p>{selected.outcome_message}</p></div> : null}
        {selected.progress ? <section className="job-progress-card">
          <div><span>当前阶段</span><strong>{phaseLabel(selected.progress.execution_phase ?? (selected.progress.status === "succeeded" ? "verified" : null))}</strong><small>{selected.progress.phase_label ?? targetText(selected.payload, selected.progress)}</small></div>
          {selected.progress.checkpoint ? <div className="job-progress-metrics">
            <span><small>成功 checkpoint</small><strong>{Number(selected.progress.checkpoint.succeeded ?? 0).toLocaleString("zh-CN")}</strong></span>
            <span><small>正在请求</small><strong>{Number(selected.progress.checkpoint.running ?? 0).toLocaleString("zh-CN")}</strong></span>
            <span><small>等待重试</small><strong>{Number(selected.progress.checkpoint.retry_waiting ?? 0).toLocaleString("zh-CN")}</strong></span>
            <span><small>终止失败</small><strong>{Number(selected.progress.checkpoint.terminal_failed ?? 0).toLocaleString("zh-CN")}</strong></span>
            <span><small>替代审计</small><strong>{Number(selected.progress.checkpoint.superseded ?? 0).toLocaleString("zh-CN")}</strong></span>
          </div> : null}
        </section> : null}
        {selected.error ? <div className="job-run-error"><strong>失败原因</strong><p>{selected.error}</p></div> : null}
        <details><summary>任务参数</summary><pre>{JSON.stringify(selected.payload, null, 2)}</pre></details>
        <div className="job-log-head"><strong>最近日志</strong><button type="button" onClick={() => openJob(selected)}>刷新日志</button></div>
        {logError ? <div className="job-run-error"><strong>日志状态</strong><p>{logError}</p></div> : null}
        <pre className="job-log">{logs.length ? logs.join("\n") : "尚无日志输出。"}</pre>
        {canControl ? <div className="job-run-actions">{["failed", "cancelled"].includes(selected.status) ? <button type="button" disabled={busy} onClick={() => action(selected, "retry")}>按原参数重试</button> : null}{["queued", "running"].includes(selected.status) ? <button type="button" className="danger-button" disabled={busy || !!selected.cancel_requested_at} onClick={() => action(selected, "cancel")}>{selected.cancel_requested_at ? "正在安全停止" : "取消任务"}</button> : null}</div> : null}
      </aside> : null}
    </div>
  </section>;
}
