"use client";

import { FormEvent, useState } from "react";
import { apiFetch } from "./api-client";

export type AuthUser = {
  id?: string | null;
  username: string;
  display_name: string;
  role: string;
  permissions: string[];
};

export type AuthState = {
  status: "disabled" | "bootstrap_required" | "login_required" | "authenticated";
  user: AuthUser | null;
};

export function AuthPanel({ api, state, onAuthenticated }: {
  api: string; state: AuthState; onAuthenticated: () => Promise<void>;
}) {
  const bootstrap = state.status === "bootstrap_required";
  const [username, setUsername] = useState("admin");
  const [displayName, setDisplayName] = useState("系统管理员");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (bootstrap && password !== confirmPassword) { setMessage("两次输入的密码不一致。"); return; }
    setSubmitting(true); setMessage("");
    try {
      const response = await apiFetch(`${api}/api/auth/${bootstrap ? "bootstrap" : "login"}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bootstrap ? { username, display_name: displayName, password } : { username, password }),
      });
      const body = await response.json();
      if (!response.ok) { setMessage(body.detail ?? "认证失败"); return; }
      setPassword(""); setConfirmPassword(""); await onAuthenticated();
    } catch { setMessage("无法连接认证服务。"); } finally { setSubmitting(false); }
  }

  return <main className="auth-shell">
    <section className="auth-card">
      <div className="auth-brand"><span className="brand-mark">Q</span><span>Quant<span>Lab</span></span></div>
      <p className="eyebrow">LOCAL CONTROL PLANE / SECURED</p>
      <h1>{bootstrap ? "创建初始管理员" : "登录量化研究系统"}</h1>
      <p className="auth-copy">{bootstrap ? "这是唯一一次无登录创建管理员的机会。密码将使用 Argon2 保存，系统不会保存明文。" : "请输入本地 QuantLab 账户。连续五次失败会锁定账户十五分钟。"}</p>
      <form onSubmit={submit}>
        <label>用户名<input autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} /></label>
        {bootstrap && <label>显示名称<input autoComplete="name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} /></label>}
        <label>密码<input type="password" autoComplete={bootstrap ? "new-password" : "current-password"} value={password} onChange={(event) => setPassword(event.target.value)} /></label>
        {bootstrap && <label>确认密码<input type="password" autoComplete="new-password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} /></label>}
        {bootstrap && <div className="password-policy"><span>至少 12 个字符</span><span>包含字母、数字和符号</span><span>不会写入日志或审计正文</span></div>}
        {message && <div className="auth-error">{message}</div>}
        <button className="primary" disabled={submitting || username.length < 3 || password.length < (bootstrap ? 12 : 1)}>{submitting ? "处理中…" : bootstrap ? "创建管理员并登录" : "登录"}</button>
      </form>
    </section>
  </main>;
}
