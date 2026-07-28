"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "./api-client";

export function ResourcePanel({
  api,
  endpoint,
  title,
  eyebrow,
  empty,
  children,
}: {
  api: string;
  endpoint: string;
  title: string;
  eyebrow: string;
  empty: string;
  children?: React.ReactNode;
}) {
  const [items, setItems] = useState<Record<string, unknown>[]>([]);
  const [message, setMessage] = useState("");

  const refresh = useCallback(async () => {
    try {
      const response = await apiFetch(`${api}${endpoint}`, { cache: "no-store" });
      if (!response.ok) throw new Error("request failed");
      const body = await response.json();
      setItems(Array.isArray(body) ? body : Array.isArray(body.items) ? body.items : [body]);
      setMessage("");
    } catch {
      setMessage("当前数据不可用；未执行任何写操作。");
    }
  }, [api, endpoint]);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  return (
    <section className="panel wide">
      <div className="panel-header">
        <div><p className="eyebrow">{eyebrow}</p><h2>{title}</h2></div>
        <button className="button" onClick={() => void refresh()}>刷新</button>
      </div>
      {children}
      {message && <p className="warning">{message}</p>}
      {!message && items.length === 0 && <div className="empty">{empty}</div>}
      {items.length > 0 && (
        <div className="list">
          {items.slice(0, 12).map((item, index) => (
            <div className="row" key={String(item.id ?? item.name ?? index)}>
              <strong>{String(item.name ?? item.title ?? item.id ?? `记录 ${index + 1}`)}</strong>
              <span className="badge">{String(item.status ?? item.state ?? "available")}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
