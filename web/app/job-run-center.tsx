"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch } from "./api-client";
import { DataJob, jobDisplayName, phaseLabel, targetText } from "./data-progress";
import { usePolling } from "./use-polling";

type Job = DataJob & {
  cancel_requested_at?: string | null;
  exit_code?: number | null;
  outcome_status?: "blocked" | "passed" | "rejected" | null;
  outcome_message?: string | null;
  presentation?: {
    display_status: string;
    label: string;
    safe_reason?: string | null;
    phase_label?: string | null;
  } | null;
};

async function jsonResponse<T>(request: Promise<Response>): Promise<T> {
  const response = await request;
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

const statusOptions = [
  ["", "全部历史"],
  ["failed", "历史失败"],
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

function displayedOutcome(job: Job) {
  if (job.presentation?.display_status === "interrupted") {
    return { text: job.presentation.label, className: "cancelled" };
  }
  if (job.outcome_status) {
    return {
      text: statusText[job.outcome_status] ?? job.outcome_status,
      className: job.outcome_status === "passed" ? "succeeded" : job.outcome_status,
    };
  }
  return {
    text: job.status === "failed" ? "历史失败" : job.presentation?.label ?? statusText[job.status] ?? job.status,
    className: job.status,
  };
}

function timeText(value?: string | null) {
  return value ? new Date(value).toLocaleString("zh-CN") : "—";
}

function laterAttemptStatus(job: Job) {
  if (job.status !== "failed") return "";
  if (job.retry_successor?.status === "succeeded") return "后续任务已成功";
  if (["queued", "running"].includes(job.retry_successor?.status ?? "")) {
    return "后续任务运行中";
  }
  return "";
}

function reasonText(job: Job) {
  if (job.presentation?.safe_reason) return job.presentation.safe_reason;
  return job.error === "job execution failed" ? "任务执行失败，详细原因尚未提供。" : job.error;
}

function executionText(job: Job) {
  if (["succeeded", "failed", "cancelled"].includes(job.status)) {
    return job.presentation?.label ?? statusText[job.status] ?? job.status;
  }
  if (job.presentation?.phase_label) return job.presentation.phase_label;
  if (job.progress?.execution_phase) return phaseLabel(job.progress.execution_phase);
  if (job.status === "running") return "执行中，等待阶段进度";
  return statusText[job.status] ?? job.status;
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
  const [scope, setScope] = useState<"active" | "history">("active");
  const [status, setStatus] = useState("");
  const [kind, setKind] = useState("");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Job | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [logError, setLogError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [detailError, setDetailError] = useState("");
  const [detailUpdated, setDetailUpdated] = useState<string | null>(null);
  const requestVersion = useRef(0);
  const selectionVersion = useRef(0);
  const detailRequestVersion = useRef(0);
  const pageSize = 20;

  const load = useCallback(async () => {
    const version = ++requestVersion.current;
    const params = new URLSearchParams({ limit: String(pageSize), offset: String(page * pageSize) });
    const statuses = scope === "active" ? ["queued", "running"] : status ? [status] : ["failed", "succeeded", "cancelled"];
    for (const value of statuses) params.append("status", value);
    if (kind.trim()) params.append("kind", kind.trim());
    try {
      const response = await apiFetch(`${api}/api/jobs?${params}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const next: Job[] = await response.json();
      const nextTotal = Number(response.headers.get("x-total-count"));
      if (!Array.isArray(next) || next.some((job) => !job.id || !statuses.includes(job.status))
        || !response.headers.has("x-total-count") || !Number.isInteger(nextTotal)
        || nextTotal < next.length) throw new Error("Task scope could not be verified");
      if (version !== requestVersion.current) return;
      setJobs(next);
      setTotal(nextTotal);
      setLastUpdated(new Date().toISOString());
      setLoadError("");
    } catch {
      if (version === requestVersion.current) setLoadError("刷新失败，下方保留上次状态，不能据此判断任务仍在运行。");
    }
  }, [api, kind, page, scope, status]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => { window.clearTimeout(timer); requestVersion.current += 1; };
  }, [load]);

  usePolling(load, 5000);

  usePolling(async () => {
    if (!selected) return;
    const id = selected.id;
    const version = selectionVersion.current;
    const detailVersion = ++detailRequestVersion.current;
    try {
      const next = await jsonResponse<Job>(apiFetch(`${api}/api/jobs/${id}`, { cache: "no-store" }));
      if (version !== selectionVersion.current || detailVersion !== detailRequestVersion.current) return;
      setSelected((current) => current?.id === id ? next : current);
      setDetailUpdated(new Date().toISOString());
      setDetailError("");
    } catch {
      if (version === selectionVersion.current && detailVersion === detailRequestVersion.current) setDetailError("详情刷新失败，当前显示上次读取结果。");
    }
  }, 5000, Boolean(selected));

  async function openJob(job: Job) {
    const version = ++selectionVersion.current;
    const detailVersion = ++detailRequestVersion.current;
    const previousSelectionIsSame = selected?.id === job.id;
    if (!previousSelectionIsSame) {
      setSelected(job);
      setLogs([]);
      setLogError("");
      setDetailError("");
      setDetailUpdated(null);
    }
    const [detailResult, logResult] = await Promise.allSettled([
      jsonResponse<Job>(apiFetch(`${api}/api/jobs/${job.id}`, { cache: "no-store" })).then((detail) => {
        if (version === selectionVersion.current && detailVersion === detailRequestVersion.current) {
          setSelected(detail);
          setDetailUpdated(new Date().toISOString());
          setDetailError("");
        }
        return detail;
      }),
      jsonResponse<{ lines?: string[] }>(apiFetch(`${api}/api/jobs/${job.id}/log?tail=300`, { cache: "no-store" })),
    ]);
    if (version !== selectionVersion.current) return;
    if (detailResult.status === "rejected" && detailVersion === detailRequestVersion.current) {
      setDetailError("详情刷新失败，当前显示上次读取结果。");
    }
    if (logResult.status === "fulfilled") {
      if (detailResult.status === "fulfilled" || previousSelectionIsSame) {
        setLogs(logResult.value.lines ?? []);
        setLogError("");
      }
    } else if (detailResult.status === "fulfilled" || previousSelectionIsSame) {
      setLogError(previousSelectionIsSame ? "日志刷新失败，继续显示上次成功内容。" : "日志暂不可用。");
      if (!previousSelectionIsSame) setLogs([]);
    }

    if (detailVersion !== detailRequestVersion.current) return;
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
    } catch {
      onMessage("任务操作未得到服务器确认，请刷新状态后再检查结果。");
    } finally {
      setBusy(false);
    }
  }

  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const kindsOnPage = useMemo(() => [...new Set(jobs.map((item) => item.kind))].sort(), [jobs]);
  const selectedLaterStatus = selected ? laterAttemptStatus(selected) : "";

  return <section className="job-run-center">
    <div className="job-run-toolbar">
      <div><h2>任务运行</h2><p>{scope === "active" ? "当前排队和执行中的任务；每 5 秒从服务器更新。" : "已结束任务的原始记录。同一故障可能关联研究和参数试验，历史条数不是当前故障数。"}</p></div>
      <div className="job-run-filters">
        <select aria-label="任务范围" value={scope} onChange={(event) => { requestVersion.current += 1; setScope(event.target.value as "active" | "history"); setPage(0); setJobs([]); setLastUpdated(null); }}><option value="active">当前运行</option><option value="history">历史记录</option></select>
        {scope === "history" ? <select aria-label="任务状态" value={status} onChange={(event) => { requestVersion.current += 1; setStatus(event.target.value); setPage(0); setJobs([]); setLastUpdated(null); }}>{statusOptions.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select> : null}
        <input aria-label="任务类型" list="job-kinds" value={kind} onChange={(event) => { requestVersion.current += 1; setKind(event.target.value); setPage(0); setJobs([]); setLastUpdated(null); }} placeholder="全部任务类型" />
        <datalist id="job-kinds">{kindsOnPage.map((value) => <option value={value} key={value} />)}</datalist>
        <button type="button" onClick={load}>刷新</button>
      </div>
    </div>
    <p role="status">{lastUpdated ? `上次成功更新：${timeText(lastUpdated)}` : "正在读取任务状态…"}</p>
    {loadError ? <div className="job-run-error" role="alert">{loadError}</div> : null}
    <div className={`job-run-layout ${selected ? "with-detail" : ""}`}>
      <div className="job-run-list">
        {jobs.map((job) => {
          const laterStatus = laterAttemptStatus(job);
          const outcome = displayedOutcome(job);
          return <button type="button" className={selected?.id === job.id ? "selected" : ""} onClick={() => openJob(job)} key={job.id}>
          <span className={`job-state ${loadError ? "cancelled" : job.status}`} />
          <div><strong>{jobDisplayName(job)}</strong><small>{executionText(job)} · {targetText(job.payload, job.progress)} · {timeText(job.created_at)}</small>{reasonText(job) ? <em>{reasonText(job)}</em> : null}{laterStatus ? <small>{laterStatus} · {job.retry_successor?.id.slice(0, 10)}</small> : null}</div>
          <code>{job.id.slice(0, 10)}</code>
          <span className={`task-status ${loadError ? "cancelled" : outcome.className}`}>{loadError ? "上次状态：" : ""}{outcome.text}</span>
        </button>;
        })}
        {!jobs.length ? <div className="empty compact">{!lastUpdated ? loadError ? "尚未读到任务状态。" : "正在读取…" : scope === "active" ? "当前没有排队或执行中的任务。历史结果可切换查看。" : "当前筛选条件下没有历史任务。"}</div> : null}
        <footer><span>{lastUpdated ? `${scope === "active" ? "当前任务" : "历史记录"}共 ${total} 条 · 第 ${page + 1}/${pageCount} 页` : "等待服务器返回"}</span><div><button type="button" disabled={page === 0} onClick={() => { requestVersion.current += 1; setPage(page - 1); setJobs([]); setLastUpdated(null); }}>上一页</button><button type="button" disabled={!lastUpdated || page + 1 >= pageCount} onClick={() => { requestVersion.current += 1; setPage(page + 1); setJobs([]); setLastUpdated(null); }}>下一页</button></div></footer>
      </div>
      {selected ? <aside className="job-run-detail">
        <header><div><span>{selected.kind}</span><h3>{jobDisplayName(selected)}</h3></div><button type="button" onClick={() => { selectionVersion.current += 1; setSelected(null); setLogError(""); }}>关闭</button></header>
        <p role="status">{detailUpdated ? `详情更新：${timeText(detailUpdated)}` : "正在读取详情…"}</p>
        {detailError ? <div className="job-run-error" role="alert">{detailError}</div> : null}
        <dl><div><dt>任务 ID</dt><dd><code>{selected.id}</code></dd></div><div><dt>{detailError ? "上次程序状态" : "程序状态"}</dt><dd>{selected.presentation?.label ?? statusText[selected.status] ?? selected.status}{selected.cancel_requested_at && selected.status === "running" ? " · 正在安全停止" : ""}</dd></div>{selectedLaterStatus ? <div><dt>后续任务</dt><dd>{selectedLaterStatus} · {selected.retry_successor?.id}</dd></div> : null}{selected.outcome_status ? <div><dt>研究结论</dt><dd>{statusText[selected.outcome_status] ?? selected.outcome_status}</dd></div> : null}<div><dt>开始 / 结束</dt><dd>{timeText(selected.started_at)} / {timeText(selected.finished_at)}</dd></div><div><dt>退出码</dt><dd>{selected.exit_code ?? "—"}</dd></div></dl>
        {selected.outcome_message ? <div className="job-run-error"><strong>研究结果</strong><p>{selected.outcome_message}</p></div> : null}
        {selected.progress ? <section className="job-progress-card">
          <div><span>{detailError ? "上次执行状态" : "执行状态"}</span><strong>{executionText(selected)}</strong><small>{selected.progress.phase_label ?? targetText(selected.payload, selected.progress)}</small></div>
          {selected.progress.checkpoint ? <div className="job-progress-metrics">
            <span><small>成功 checkpoint</small><strong>{Number(selected.progress.checkpoint.succeeded ?? 0).toLocaleString("zh-CN")}</strong></span>
            <span><small>正在请求</small><strong>{Number(selected.progress.checkpoint.running ?? 0).toLocaleString("zh-CN")}</strong></span>
            <span><small>等待重试</small><strong>{Number(selected.progress.checkpoint.retry_waiting ?? 0).toLocaleString("zh-CN")}</strong></span>
            <span><small>终止失败</small><strong>{Number(selected.progress.checkpoint.terminal_failed ?? 0).toLocaleString("zh-CN")}</strong></span>
            <span><small>替代审计</small><strong>{Number(selected.progress.checkpoint.superseded ?? 0).toLocaleString("zh-CN")}</strong></span>
          </div> : null}
        </section> : null}
        {reasonText(selected) ? <div className="job-run-error"><strong>状态说明</strong><p>{reasonText(selected)}</p></div> : null}
        <details><summary>任务参数</summary><pre>{JSON.stringify(selected.payload, null, 2)}</pre></details>
        <div className="job-log-head"><strong>最近日志</strong><button type="button" onClick={() => openJob(selected)}>刷新日志</button></div>
        {logError ? <div className="job-run-error"><strong>日志状态</strong><p>{logError}</p></div> : null}
        <pre className="job-log">{logs.length ? logs.join("\n") : "尚无日志输出。"}</pre>
        {canControl ? <div className="job-run-actions">{["failed", "cancelled"].includes(selected.status) ? <button type="button" disabled={busy || !detailUpdated || !!detailError} onClick={() => action(selected, "retry")}>按原参数重试</button> : null}{["queued", "running"].includes(selected.status) ? <button type="button" className="danger-button" disabled={busy || !detailUpdated || !!detailError || !!selected.cancel_requested_at} onClick={() => action(selected, "cancel")}>{selected.cancel_requested_at ? "正在安全停止" : "取消任务"}</button> : null}</div> : null}
      </aside> : null}
    </div>
  </section>;
}
