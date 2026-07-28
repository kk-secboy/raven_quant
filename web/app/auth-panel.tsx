"use client";

import { FormEvent, useState } from "react";
import { apiFetch } from "./api-client";

export type AuthUser = {
  username: string;
  display_name: string;
  role: string;
  permissions: string[];
};

export type AuthState = {
  status: "disabled" | "bootstrap_required" | "login_required" | "authenticated";
  user: AuthUser | null;
};

export function AuthPanel({
  api,
  state,
  onAuthenticated,
}: {
  api: string;
  state: AuthState;
  onAuthenticated: () => Promise<void>;
}) {
  const bootstrap = state.status === "bootstrap_required";
  const [username, setUsername] = useState("admin");
  const [displayName, setDisplayName] = useState("系统管理员");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (bootstrap && password !== confirmation) {
      setMessage("两次输入的密码不一致。");
      return;
    }
    setSubmitting(true);
    setMessage("");
    try {
      const response = await apiFetch(`${api}/api/auth/${bootstrap ? "bootstrap" : "login"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          bootstrap
            ? { username, display_name: displayName, password }
            : { username, password },
        ),
      });
      const body = await response.json();
      if (!response.ok) {
        setMessage(body.detail ?? "认证失败");
        return;
      }
      setPassword("");
      setConfirmation("");
      await onAuthenticated();
    } catch {
      setMessage("无法连接认证服务。");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-shell">
      <section className="auth-card">
        <div className="brand"><span className="brand-mark">Q</span><span>QuantLab</span></div>
        <p className="eyebrow">LOCAL CONTROL PLANE / SECURED</p>
        <h1>{bootstrap ? "创建初始管理员" : "登录量化研究系统"}</h1>
        <p className="muted">
          {bootstrap
            ? "这是唯一一次无登录创建管理员的机会。"
            : "登录后才能查看研究、审批、建议和模拟账户。"}
        </p>
        <form onSubmit={submit}>
          <label>用户名<input value={username} onChange={(event) => setUsername(event.target.value)} /></label>
          {bootstrap && <label>显示名称<input value={displayName} onChange={(event) => setDisplayName(event.target.value)} /></label>}
          <label>密码<input type="password" value={password} onChange={(event) => setPassword(event.target.value)} /></label>
          {bootstrap && <label>确认密码<input type="password" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /></label>}
          {message && <p className="danger">{message}</p>}
          <button className="button primary" disabled={submitting}>
            {submitting ? "处理中…" : bootstrap ? "创建管理员并登录" : "登录"}
          </button>
        </form>
      </section>
    </main>
  );
}
