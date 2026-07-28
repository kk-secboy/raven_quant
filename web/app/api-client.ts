type CachedResponse = {
  body: string;
  headers: [string, string][];
  status: number;
  statusText: string;
  storedAt: number;
};

type CachePolicy = {
  freshMs: number;
  staleMs: number;
  persist: boolean;
};

const SESSION_PREFIX = "quantlab:query:";
const MAX_PERSISTED_BODY = 2_000_000;
const inflightGets = new Map<string, Promise<CachedResponse>>();
const responseCache = new Map<string, CachedResponse>();
let cacheGeneration = 0;

export type ApiRequestInit = RequestInit & {
  forceRefresh?: boolean;
  timeoutMs?: number;
};

function requestKey(input: RequestInfo | URL) {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  return input.url;
}

function cachePolicy(key: string): CachePolicy | null {
  const url = new URL(key, window.location.origin);
  const path = url.pathname;
  if (path.startsWith("/api/auth/") || path.startsWith("/api/settings") || path.endsWith("/log")) {
    return null;
  }
  if (path === "/api/data-retention") {
    return { freshMs: 10 * 60_000, staleMs: 60 * 60_000, persist: true };
  }
  if (
    path === "/api/datasets"
    || path === "/api/snapshots"
    || path === "/api/qlib/datasets"
    || path === "/api/strategy-recipes"
    || path === "/api/factors/gate-policy"
  ) {
    return { freshMs: 5 * 60_000, staleMs: 30 * 60_000, persist: true };
  }
  if (path === "/api/market/overview") {
    return { freshMs: 30_000, staleMs: 5 * 60_000, persist: true };
  }
  if (path === "/api/jobs" || path.startsWith("/api/jobs/")) {
    return { freshMs: 3_000, staleMs: 15_000, persist: true };
  }
  return { freshMs: 5_000, staleMs: 5 * 60_000, persist: true };
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

function responseFromCache(value: CachedResponse, state: "fresh" | "stale" | "network") {
  const headers = new Headers(value.headers);
  headers.set("X-QuantLab-Cache", state);
  return new Response(value.body, {
    headers,
    status: value.status,
    statusText: value.statusText,
  });
}

function readCached(key: string, persist: boolean) {
  const memory = responseCache.get(key);
  if (memory || !persist) return memory;
  try {
    const raw = window.sessionStorage.getItem(`${SESSION_PREFIX}${key}`);
    if (!raw) return undefined;
    const parsed = JSON.parse(raw) as CachedResponse;
    if (!Number.isFinite(parsed.storedAt) || typeof parsed.body !== "string") return undefined;
    responseCache.set(key, parsed);
    return parsed;
  } catch {
    return undefined;
  }
}

function storeCached(key: string, value: CachedResponse, persist: boolean) {
  responseCache.set(key, value);
  if (!persist || value.body.length > MAX_PERSISTED_BODY) return;
  try {
    window.sessionStorage.setItem(`${SESSION_PREFIX}${key}`, JSON.stringify(value));
  } catch {
    // A full or restricted session store must never block the live response.
  }
}

async function requestPayload(
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs: number,
  key: string,
  policy: CachePolicy,
) {
  const generation = cacheGeneration;
  const response = await fetchWithTimeout(input, init, timeoutMs);
  const headers: [string, string][] = [];
  response.headers.forEach((value, name) => headers.push([name, value]));
  const payload: CachedResponse = {
    body: await response.text(),
    headers,
    status: response.status,
    statusText: response.statusText,
    storedAt: Date.now(),
  };
  if (response.ok && generation === cacheGeneration) storeCached(key, payload, policy.persist);
  return payload;
}

function networkGet(
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs: number,
  key: string,
  policy: CachePolicy,
) {
  const existing = inflightGets.get(key);
  if (existing) return existing;
  const pending = requestPayload(input, init, timeoutMs, key, policy);
  inflightGets.set(key, pending);
  pending.finally(() => {
    if (inflightGets.get(key) === pending) inflightGets.delete(key);
  }).catch(() => undefined);
  return pending;
}

export function clearApiCache() {
  cacheGeneration += 1;
  inflightGets.clear();
  responseCache.clear();
  try {
    for (let index = window.sessionStorage.length - 1; index >= 0; index -= 1) {
      const key = window.sessionStorage.key(index);
      if (key?.startsWith(SESSION_PREFIX)) window.sessionStorage.removeItem(key);
    }
  } catch {
    // Session storage may be unavailable in hardened browser modes.
  }
}

export function apiFetch(input: RequestInfo | URL, init: ApiRequestInit = {}) {
  const { forceRefresh = false, timeoutMs: requestedTimeout, ...requestInit } = init;
  const method = (requestInit.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
  const timeoutMs = requestedTimeout ?? (method === "GET" ? 15_000 : 30_000);

  if (method !== "GET" || requestInit.signal) {
    return fetchWithTimeout(input, requestInit, timeoutMs).then((response) => {
      if (method !== "GET" && response.ok) clearApiCache();
      return response;
    });
  }

  const key = requestKey(input);
  const policy = cachePolicy(key);
  if (!policy) return fetchWithTimeout(input, requestInit, timeoutMs);

  const cached = readCached(key, policy.persist);
  const age = cached ? Date.now() - cached.storedAt : Number.POSITIVE_INFINITY;
  if (!forceRefresh && cached && age <= policy.freshMs) {
    return Promise.resolve(responseFromCache(cached, "fresh"));
  }
  if (!forceRefresh && cached && age <= policy.staleMs) {
    void networkGet(input, requestInit, timeoutMs, key, policy).catch(() => undefined);
    return Promise.resolve(responseFromCache(cached, "stale"));
  }

  return networkGet(input, requestInit, timeoutMs, key, policy)
    .then((payload) => responseFromCache(payload, "network"))
    .catch((error) => {
      if (cached && age <= policy.staleMs) return responseFromCache(cached, "stale");
      throw error;
    });
}
