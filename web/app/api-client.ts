const inflightGets = new Map<string, Promise<Response>>();

export type ApiRequestInit = RequestInit & { timeoutMs?: number };

function requestKey(input: RequestInfo | URL) {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  return input.url;
}

function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit, timeoutMs: number) {
  const controller = new AbortController();
  const inheritedSignal = init.signal;
  const abortFromCaller = () => controller.abort(inheritedSignal?.reason);
  if (inheritedSignal?.aborted) abortFromCaller();
  else inheritedSignal?.addEventListener("abort", abortFromCaller, { once: true });
  const timer = window.setTimeout(() => controller.abort(new Error("API request timed out")), timeoutMs);

  return fetch(input, { ...init, credentials: "include", signal: controller.signal }).finally(() => {
    window.clearTimeout(timer);
    inheritedSignal?.removeEventListener("abort", abortFromCaller);
  });
}

export function apiFetch(input: RequestInfo | URL, init: ApiRequestInit = {}) {
  const { timeoutMs: requestedTimeout, ...requestInit } = init;
  const method = (requestInit.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
  const timeoutMs = requestedTimeout ?? (method === "GET" ? 15_000 : 30_000);

  if (method !== "GET" || requestInit.signal) {
    return fetchWithTimeout(input, requestInit, timeoutMs);
  }

  const key = requestKey(input);
  const existing = inflightGets.get(key);
  if (existing) return existing.then((response) => response.clone());

  const pending = fetchWithTimeout(input, requestInit, timeoutMs);
  inflightGets.set(key, pending);
  pending.finally(() => {
    if (inflightGets.get(key) === pending) inflightGets.delete(key);
  }).catch(() => undefined);
  return pending.then((response) => response.clone());
}
