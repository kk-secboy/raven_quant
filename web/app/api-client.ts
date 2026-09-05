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

const SESSION_ROOT_PREFIX = "quantlab:query:";
const CACHE_SCHEMA_VERSION = "api-response-v2";
const CACHE_RELEASE_VERSION = process.env.NEXT_PUBLIC_CACHE_RELEASE ?? "quantlab-web-2026-08-26";
const SESSION_CACHE_VERSION = `${CACHE_SCHEMA_VERSION}:${CACHE_RELEASE_VERSION}`;
const SESSION_VERSION_KEY = `${SESSION_ROOT_PREFIX}active-version`;
const SESSION_PREFIX = `${SESSION_ROOT_PREFIX}${SESSION_CACHE_VERSION}:`;
const MAX_PERSISTED_BODY = 2_000_000;
const inflightGets = new Map<string, Promise<CachedResponse>>();
const responseCache = new Map<string, CachedResponse>();
let cacheGeneration = 0;
let sessionCachePrepared = false;

export type ApiRequestInit = RequestInit & {
  forceRefresh?: boolean;
  timeoutMs?: number;
};

// This is deliberately a paper-ledger projection, not the formal
// RecommendationStore contract.  It is populated only from an immutable Qlib
// order-plan already queued for the active isolated Autopilot account.
export type PaperTargetAdjustment = {
  instrument: string;
  target_weight: number;
  previous_weight: number;
  weight_change: number;
  action: "add" | "remove" | "increase" | "decrease" | "hold";
  reason: string;
  reason_basis: "frozen_qlib_order_plan_weight_delta";
};

export type PaperTargetProjection = {
  contract_version: "paper-target-projection-v1";
  status: "ready" | "waiting_for_paper_account" | "waiting_for_order_plan" | "blocked_invalid_order_plan";
  message: string;
  blocker?: string;
  mode: "paper_only";
  recommendation_enabled: false;
  real_trading_eligible: false;
  simulation_portfolio: {
    id: string;
    name: string;
    status: string;
    strategy_version_id: string;
  } | null;
  batch?: {
    id: string;
    status: string;
    order_plan_manifest_sha256: string;
    target_weights_sha256: string;
    previous_batch_id: string | null;
  };
  signal_date: string | null;
  trade_date: string | null;
  cash_weight?: number;
  targets: PaperTargetAdjustment[];
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
    // The payload is bound to an immutable snapshot and carries its as-of
    // date. Keep the last successful dashboard visible while the next daily
    // artifact is generated or the API is temporarily busy.
    return { freshMs: 30_000, staleMs: 7 * 24 * 60 * 60_000, persist: true };
  }
  if (path === "/api/qlib/status" || path === "/api/rdagent/status") {
    // Runtime capability changes must be observed quickly and must never be
    // restored from a previous browser session or frontend release.
    return { freshMs: 2_000, staleMs: 10_000, persist: false };
  }
  if (path === "/api/jobs" || path.startsWith("/api/jobs/")) {
    return { freshMs: 3_000, staleMs: 15_000, persist: true };
  }
  return { freshMs: 5_000, staleMs: 5 * 60_000, persist: true };
}

function prepareSessionCache() {
  if (sessionCachePrepared) return;
  sessionCachePrepared = true;
  try {
    if (window.sessionStorage.getItem(SESSION_VERSION_KEY) === SESSION_CACHE_VERSION) return;
    for (let index = window.sessionStorage.length - 1; index >= 0; index -= 1) {
      const key = window.sessionStorage.key(index);
      if (key?.startsWith(SESSION_ROOT_PREFIX)) window.sessionStorage.removeItem(key);
    }
    window.sessionStorage.setItem(SESSION_VERSION_KEY, SESSION_CACHE_VERSION);
  } catch {
    // A disabled session store falls back to the in-memory cache.
  }
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
  prepareSessionCache();
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
  prepareSessionCache();
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
  policy: CachePolicy | null,
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
  if (response.ok && policy && generation === cacheGeneration) {
    storeCached(key, payload, policy.persist);
  }
  return payload;
}

function networkGet(
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs: number,
  key: string,
  policy: CachePolicy | null,
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
  prepareSessionCache();
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
  prepareSessionCache();
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
  const requestCache = requestInit.cache ?? (input instanceof Request ? input.cache : undefined);
  if (requestCache === "no-store") {
    // A live-state request must not revive persisted history or hide a failed refresh.
    // Concurrent callers can share the network request without storing its response.
    return networkGet(input, requestInit, timeoutMs, `no-store:${key}`, null)
      .then((payload) => responseFromCache(payload, "network"));
  }
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
      if (!forceRefresh && cached && age <= policy.staleMs) {
        return responseFromCache(cached, "stale");
      }
      throw error;
    });
}
