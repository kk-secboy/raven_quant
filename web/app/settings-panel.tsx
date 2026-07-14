"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { apiFetch } from "./api-client";
import { StrategyDefaultsPanel } from "./strategy-defaults-panel";

type SecretState = {
  configured: boolean;
  source: "database" | "environment" | "missing";
  updated_at?: string | null;
};

type SettingsState = {
  storage_ready: boolean;
  storage_status: "ok" | "bootstrap_required" | "unavailable";
  storage_record_count: number;
  tushare: SecretState & { api_url: string; verified_at?: string | null };
  llm: SecretState & { api_base: string; chat_model: string };
  alerts: SecretState & { endpoint_host: string };
  broker: SecretState & { endpoint_host: string; mode: "disabled" | "sandbox" };
};

function sourceLabel(value: SecretState["source"]) {
  return { database: "服务端动态配置", environment: "部署环境变量", missing: "未配置" }[value];
}

export function SettingsPanel({ api }: { api: string }) {
  const [state, setState] = useState<SettingsState | null>(null);
  const [tushareUrl, setTushareUrl] = useState("https://api.tushare.pro");
  const [tushareToken, setTushareToken] = useState("");
  const [apiBase, setApiBase] = useState("");
  const [chatModel, setChatModel] = useState("gpt-4.1-mini");
  const [apiKey, setApiKey] = useState("");
  const [alertWebhook, setAlertWebhook] = useState("");
  const [brokerGateway, setBrokerGateway] = useState("");
  const [brokerSecret, setBrokerSecret] = useState("");
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState<"tushare" | "llm" | "alerts" | "broker" | null>(null);

  const refresh = useCallback(async () => {
    const response = await apiFetch(`${api}/api/settings`, { cache: "no-store" });
    if (!response.ok) throw new Error("无法读取系统设置");
    const value: SettingsState = await response.json();
    setState(value);
    setTushareUrl(value.tushare.api_url || "https://api.tushare.pro");
    setApiBase(value.llm.api_base || "");
    setChatModel(value.llm.chat_model || "gpt-4.1-mini");
  }, [api]);

  useEffect(() => {
    const initial = window.setTimeout(() => {
      refresh().catch(() => setMessage("无法读取服务器配置状态。"));
    }, 0);
    return () => window.clearTimeout(initial);
  }, [refresh]);

  async function saveTushare(event: FormEvent) {
    event.preventDefault();
    setSaving("tushare"); setMessage("正在验证 Tushare 凭据……");
    try {
      const response = await apiFetch(`${api}/api/settings/tushare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_url: tushareUrl, token: tushareToken }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "Tushare 配置失败");
      setTushareToken("");
      setMessage("Tushare 凭据已验证并加密保存，后续任务立即生效。");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Tushare 配置失败");
    } finally { setSaving(null); }
  }

  async function saveLlm(event: FormEvent) {
    event.preventDefault();
    setSaving("llm"); setMessage("正在保存 RD-Agent 模型配置……");
    try {
      const response = await apiFetch(`${api}/api/settings/llm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: apiKey, api_base: apiBase, chat_model: chatModel }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "模型配置失败");
      setApiKey("");
      setMessage("RD-Agent 模型凭据已加密保存，下一次研究任务立即生效。");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "模型配置失败");
    } finally { setSaving(null); }
  }

  async function saveAlerts(event: FormEvent) {
    event.preventDefault();
    setSaving("alerts"); setMessage("正在保存告警接收地址……");
    try {
      const response = await apiFetch(`${api}/api/settings/alerts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ webhook_url: alertWebhook }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "告警配置失败");
      setAlertWebhook("");
      setMessage(body.configured ? "告警地址已加密保存，Scheduler 无需重启即可生效。" : "动态告警投递已关闭，无需重启 Scheduler。");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "告警配置失败");
    } finally { setSaving(null); }
  }

  async function saveBroker(event: FormEvent) {
    event.preventDefault();
    setSaving("broker"); setMessage("正在保存券商沙箱连接凭据……");
    try {
      const response = await apiFetch(`${api}/api/settings/broker`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gateway_url: brokerGateway, hmac_secret: brokerSecret }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? "券商沙箱配置失败");
      setBrokerGateway("");
      setBrokerSecret("");
      setMessage(body.configured
        ? `券商沙箱凭据已加密保存并热加载；BROKER_MODE=${body.mode} 仍是部署级安全锁。`
        : "动态券商连接凭据已关闭；BROKER_MODE 部署级安全锁未改变。");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "券商沙箱配置失败");
    } finally { setSaving(null); }
  }

  return <>
    {message && <div className="notice">{message}</div>}
    <section className="settings-status">
      <article><span className={state?.storage_ready ? "pulse ok" : "pulse"} /><div><small>加密存储</small><strong>{state?.storage_ready ? "可用" : state?.storage_status === "unavailable" ? "无法解密" : "未启用"}</strong><p>{state?.storage_ready ? `${state.storage_record_count} 条密文记录已验证。` : "检查 PLATFORM_SECRET_KEY；不会回退到旧动态配置。"}</p></div></article>
      <article><span className={state?.tushare.configured ? "pulse ok" : "pulse"} /><div><small>Tushare</small><strong>{state?.tushare.configured ? "已配置" : "待配置"}</strong><p>{state ? sourceLabel(state.tushare.source) : "读取中"}</p></div></article>
      <article><span className={state?.llm.configured ? "pulse ok" : "pulse"} /><div><small>RD-Agent LLM</small><strong>{state?.llm.configured ? "已配置" : "待配置"}</strong><p>{state ? sourceLabel(state.llm.source) : "读取中"}</p></div></article>
      <article><span className={state?.alerts.configured ? "pulse ok" : "pulse"} /><div><small>告警 Webhook</small><strong>{state?.alerts.configured ? "已配置" : "未启用"}</strong><p>{state ? `${sourceLabel(state.alerts.source)}${state.alerts.endpoint_host ? ` · ${state.alerts.endpoint_host}` : ""}` : "读取中"}</p></div></article>
      <article><span className={state?.broker.configured ? "pulse ok" : "pulse"} /><div><small>券商沙箱网关</small><strong>{state?.broker.configured ? "凭据已配置" : "未配置"}</strong><p>{state ? `${sourceLabel(state.broker.source)} · 安全锁 ${state.broker.mode}${state.broker.endpoint_host ? ` · ${state.broker.endpoint_host}` : ""}` : "读取中"}</p></div></article>
    </section>

    <section className="settings-grid">
      <form className="settings-card" onSubmit={saveTushare}>
        <div className="card-heading"><div><span>MARKET DATA CREDENTIAL</span><strong>Tushare 数据源</strong></div><span className={state?.tushare.configured ? "status-chip verified" : "status-chip"}>{state?.tushare.configured ? "已就绪" : "未配置"}</span></div>
        <p>保存前会调用交易日历接口验证。Token 只提交一次，保存后不会回显。</p>
        <label>API 地址<input type="url" value={tushareUrl} onChange={(event) => setTushareUrl(event.target.value)} required /></label>
        <label>Token<input type="password" autoComplete="new-password" value={tushareToken} onChange={(event) => setTushareToken(event.target.value)} placeholder={state?.tushare.configured ? "输入新 Token 可覆盖现有配置" : "输入 Tushare Token"} required /></label>
        <button className="primary" disabled={saving !== null || tushareToken.length < 8}>{saving === "tushare" ? "验证中……" : "验证并保存"}</button>
      </form>

      <form className="settings-card" onSubmit={saveLlm}>
        <div className="card-heading"><div><span>RESEARCH MODEL CREDENTIAL</span><strong>RD-Agent 模型</strong></div><span className={state?.llm.configured ? "status-chip verified" : "status-chip"}>{state?.llm.configured ? "已就绪" : "未配置"}</span></div>
        <p>API Key 会加密保存，并只在 RD-Agent 子任务启动时注入运行环境。</p>
        <label>API Base<input type="url" value={apiBase} onChange={(event) => setApiBase(event.target.value)} placeholder="留空使用供应商默认地址" /></label>
        <label>模型名称<input value={chatModel} onChange={(event) => setChatModel(event.target.value)} required /></label>
        <label>API Key<input type="password" autoComplete="new-password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder={state?.llm.configured ? "输入新 Key 可覆盖现有配置" : "输入模型 API Key"} required /></label>
        <button className="primary" disabled={saving !== null || apiKey.length < 8}>{saving === "llm" ? "保存中……" : "加密保存"}</button>
      </form>

      <form className="settings-card" onSubmit={saveAlerts}>
        <div className="card-heading"><div><span>OPERATIONS NOTIFICATION</span><strong>告警 Webhook</strong></div><span className={state?.alerts.configured ? "status-chip verified" : "status-chip"}>{state?.alerts.configured ? "已启用" : "未启用"}</span></div>
        <p>地址按密文保存，Scheduler 每次投递前读取最新版；远程地址必须使用 HTTPS。</p>
        <label>Webhook 地址<input type="url" autoComplete="off" value={alertWebhook} onChange={(event) => setAlertWebhook(event.target.value)} placeholder={state?.alerts.configured ? "输入新地址可覆盖；留空提交可关闭" : "https://alerts.example.internal/quantlab"} /></label>
        <button className="primary" disabled={saving !== null}>{saving === "alerts" ? "保存中……" : alertWebhook ? "加密保存并热加载" : "关闭动态告警"}</button>
      </form>

      <form className="settings-card" onSubmit={saveBroker}>
        <div className="card-heading"><div><span>BROKER SANDBOX CREDENTIAL</span><strong>券商沙箱网关</strong></div><span className={state?.broker.configured ? "status-chip verified" : "status-chip"}>{state?.broker.configured ? "凭据已就绪" : "未配置"}</span></div>
        <p>URL 与 HMAC 密钥按密文成对保存，每次券商操作读取最新版。此处不能改变部署级 BROKER_MODE，也不能开启实盘。</p>
        <label>Gateway URL<input type="url" autoComplete="off" value={brokerGateway} onChange={(event) => setBrokerGateway(event.target.value)} placeholder={state?.broker.configured ? "输入新地址和密钥可覆盖；两项留空提交可关闭" : "https://broker.example.internal"} /></label>
        <label>HMAC 密钥<input type="password" autoComplete="new-password" value={brokerSecret} onChange={(event) => setBrokerSecret(event.target.value)} placeholder="至少 32 个无空白字符" /></label>
        <button className="primary" disabled={saving !== null || Boolean(brokerGateway) !== Boolean(brokerSecret) || (Boolean(brokerSecret) && brokerSecret.length < 32)}>{saving === "broker" ? "保存中……" : brokerGateway ? "加密保存并热加载" : "关闭动态券商凭据"}</button>
      </form>
    </section>
    <StrategyDefaultsPanel api={api} />
  </>;
}
