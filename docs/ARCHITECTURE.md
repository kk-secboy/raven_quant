# QuantLab Architecture

QuantLab is a local-first quantitative research system for Chinese equities and ETFs.
It is intentionally separated from the previous application and uses Python for the
backend, orchestration, and research adapters.

## Product boundary

The system owns historical and incremental data ingestion, point-in-time snapshots,
Qlib experiments and backtests, bounded RD-Agent research, factor/model lifecycle,
job scheduling, research reports, and paper portfolios. Broker execution is a separate
provider adapter boundary and is not treated as part of Qlib or RD-Agent.

The implemented broker boundary is an integration sandbox, not a live adapter. It
replays successful paper orders to a separately deployed provider gateway so QMT,
CTP, or another broker SDK can be validated without coupling provider credentials or
Windows-only SDK processes to the research control plane. Live mode is rejected by
configuration until a provider-specific adapter, account reconciliation, compliance
approval, and small-capital rollout are separately completed.

The first provider adapter is a Windows-side QMT/MiniQMT process. XtQuant trading is
not placed in the Linux Compose stack. The gateway persists parent orders, scheduled
TWAP/VWAP children, provider order identifiers, failures, and replay nonces in
PostgreSQL; broker login state and trading credentials remain inside MiniQMT. A short
deterministic `order_remark` makes a recovered slice idempotent without exposing the
control-plane client order id to the provider.

Before each slice, the gateway requires the latest completed positive-volume QMT
minute bar and a fresh full-tick best bid/ask. The participation cap is applied to raw
A-share volume; excess quantity moves forward to a later signed slot, while excess in
the final slot fails closed. Every provider submission is an immutable attempt. A
stale active or partially filled attempt is canceled first; only after QMT reports the
cancel does the gateway submit the remaining quantity at the opposite quote bounded
by the configured reprice limit. Order/trade/error callbacks are retained separately
from query snapshots so recovery has both event and state evidence.

## Runtime structure

```text
Web console (vinext / React)
  -> same-origin gateway
      -> stateless FastAPI control plane
          -> local Argon2 authentication, revocable sessions, central RBAC
      -> PostgreSQL control plane (jobs, checkpoints, experiments)
      -> leased schedule and alert service
      -> capability-partitioned durable workers
          -> Tushare downloader and Parquet snapshots
          -> day/minute Qlib datasets, training, factor scans, backtest, and paper runner
          -> governed pair runner (cointegration, Kalman, minute two-leg execution)
          -> RD-Agent factor research runner
              -> private Docker-in-Docker sandbox (no host Docker socket)
      -> local data lake
          data/units
          data/snapshots
          data/qlib
          data/artifacts
```

PostgreSQL is authoritative for mutable metadata and uses row locks with
`SKIP LOCKED` for safe worker concurrency. Market and financial history is not
stored in PostgreSQL: immutable Parquet snapshots remain the canonical data lake,
while Qlib Bin is a derived research representation.

Operational secrets are a separate encrypted PostgreSQL boundary. Tushare and model
credentials are loaded when a worker starts each job; scheduler health snapshots use
the same latest Tushare record. The alert Webhook is loaded before every delivery pass,
and the sandbox broker Gateway URL/HMAC pair is loaded before every broker operation.
Environment values are bootstrap fallbacks, while an explicit encrypted database
record wins, including blank alert or broker records that disable delivery. Read APIs
return configuration state and endpoint hostnames—not tokens, API keys, full callback
URLs, or HMAC secrets. `BROKER_MODE` is intentionally excluded from runtime settings
and remains a deployment-level hard lock.
If an encrypted database override exists, failure to decrypt it never falls back to an
environment value. The runtime-secret store validates every ciphertext in health and
formal readiness checks; a missing or incorrect platform key creates a critical
component alert and disables affected external delivery paths. The API container health
endpoint uses the same validation, making secret recovery part of Compose, release, and
restore health rather than a dashboard-only warning.

Research runs, factor candidates, factor evaluations, and immutable audit events
also live in PostgreSQL. RD-Agent output is never promoted directly: its code and
factor-value artifacts are imported as candidates, then independently recalculated
against the selected Qlib snapshot. The versioned gate fails closed on missing
metrics and requires validation/test direction consistency, IC/RankIC/ICIR, at least
252 out-of-sample days, bounded turnover/correlation, and positive cost-adjusted
return. A human must record a reason before promotion.

Control-plane tables live in the dedicated `quantlab` schema. MLflow owns its
tracking tables in `public` with an explicit search path, preventing table-name
and migration-history collisions during upgrades. Alembic owns versioned control-
plane migrations; application startup no longer creates tables implicitly.

## Delivery slices

1. Data operations: credentials, bootstrap, incremental sync, coverage, quality,
   snapshots, Qlib conversion, durable jobs, logs, and Web status.
2. Qlib research: experiment templates, immutable configurations, Alpha158 baseline,
   LightGBM baseline, separate minute microstructure scans, prediction artifacts,
   transaction-cost backtests, and comparisons. Frequency mixing fails closed.
3. RD-Agent: constrained objectives, budgets, run timelines, generated code isolation,
   evaluation, and candidate registration.
4. Research governance: factor registry, IC/RankIC, decay, correlation, walk-forward
   gates, promotion states, and audit history.
5. Portfolio operations: daily scoring, explainable candidates, constraints, paper
   orders, holdings, NAV, attribution, alerts, and scheduled after-close workflows.

Continuous research is a control-plane policy above those slices, rather than a
second research engine. A program pins one verified Qlib lineage, fixed rolling
train/validation/test windows, bounded RD-Agent and experiment budgets, and a
maximum number of active campaigns. The scheduler creates at most one campaign for
each immutable dataset identity after the configured number of new trading days.
That campaign then follows the same governed path:

```text
verified same-lineage Qlib snapshot
  -> bounded RD-Agent factor proposals
  -> independent factor evaluation and gate
  -> Qlib baseline and parameter experiments
  -> challenger strategy and backtest evidence
  -> human strategy approval
  -> paper schedule
```

Program leases and the database uniqueness constraint make scheduler retries safe.
Waiting conditions remain visible without shortening safety windows; controller
failures release their lease and append an immutable failure event. No program can
approve a strategy or cross the sandbox-only broker boundary automatically.

Implemented vertical slices:

1. Web -> durable bootstrap job -> resumable downloader -> coverage and status.
2. Web -> durable Qlib experiment -> Alpha158 -> LightGBM -> predictions -> Top-K
   transaction-cost backtest -> immutable metrics and artifacts.
3. Gateway -> Web/API split -> PostgreSQL migration -> external pinned-Qlib worker ->
   MLflow tracking, with container health checks and shared-volume artifacts.
4. Web -> bounded RD-Agent run -> isolated code/value export -> independent Qlib
   factor evaluation -> fail-closed gate -> audited manual promotion.
5. Web -> immutable strategy version -> promoted factor artifacts -> next-open
   multi-factor Top-K simulation -> cost/risk metrics -> audited strategy approval.
6. Web -> approved strategy -> paper portfolio -> idempotent after-close batch ->
   next-open orders/fills -> atomic cash/position/NAV ledger -> persisted risk events ->
   operator acknowledgement -> guarded resolution -> separate safe reactivation.
7. Web -> approved Qlib strategy backtests -> aligned daily returns -> correlation
   gate and risk budget -> second-person approval -> child paper ledgers -> aggregate
   NAV and group-level reduction/liquidation circuit breakers.
   Group recovery requires reconciled child states, no active child batch, auditable
   resolution of every critical event, and an explicit operator resume action.
8. Web -> durable daily schedule -> leased run slot -> idempotent worker job ->
   failure/risk projection -> in-app acknowledgement and optional webhook delivery.
- Minute research: Web -> minute execution snapshot -> independent minute-Qlib conversion -> bounded
   multi-horizon factor scan. Download, conversion, and research are separate retryable
   jobs; none can silently enter the day-frequency strategy engine.

Every durable job remains append-only history. The API exposes filtered/paged queries,
bounded log tails and parameter/error detail. Cancellation is cooperative: queued work
is cancelled atomically, while a running worker observes `cancel_requested_at`, terminates
its owned child process, and records a terminal cancelled state without deleting output
or successful prerequisite work.

## Document six-layer mapping

The document's C++ sample is an architectural decomposition, not a requirement to
maintain a second trading engine. QuantLab keeps one Python production path and maps
the same responsibilities onto Qlib, RD-Agent and governed services:

| Document layer | Production owner | Enforced boundary |
| --- | --- | --- |
| Data | `quant_data`, immutable Parquet snapshots and derived Qlib Bin | Research and execution consume only verified snapshot identities and hashes. |
| Factor | RD-Agent worker and factor registry | Generated code/value artifacts require independent Qlib evaluation and manual promotion. |
| Signal | Qlib feature/model workers and the native pair engine | Signals are point-in-time, reproducible and bound to an immutable dataset. |
| Portfolio | strategy versions, allocation store and ordinary/pair paper ledgers | Position, industry, turnover, capacity and drawdown limits are persisted per governed object. |
| Execution | next-open replay, TWAP/VWAP contract and sandbox-only QMT gateway | Participation, sessions, idempotency, two-person release and broker reconciliation fail closed. |
| Analyzer | backtest artifacts, rolling/event stress, reviews and readiness profiles | Approval revalidates metrics, provenance and on-disk evidence instead of trusting the Web response. |

This mapping intentionally avoids a parallel C++ path whose results could diverge from
the Qlib/RD-Agent path selected for this product.
   Multi-strategy allocations create and manage every child schedule atomically;
   desired operator state is retained separately from portfolio-driven suspension.
9. First visit -> one-time administrator bootstrap -> Argon2 password hash ->
   revocable strict-cookie session -> central API permission check -> metadata-only
   operation audit and Web user administration.
10. Maintenance window -> validate Fernet key -> quiesce every writer -> PostgreSQL
    custom dump + `/data` archive -> checksum and one-way key-fingerprint manifest ->
    target-key preflight before any restore mutation -> staged restore -> forward
    migration -> ciphertext-aware health check.
11. Scheduler -> component/data/queue probes -> immutable health snapshots ->
    deduplicated component alerts -> Operations health history.
12. Successful paper batch -> hashed idempotent outbox -> two-person sandbox release ->
    HMAC-signed external broker gateway -> append-only delivery events.
13. Web -> immutable pair version -> correlation/cointegration evidence -> online Kalman
    spread -> next-day minute windows -> atomic two-leg fill/reject -> cost, borrow,
    capacity, rolling-cointegration and robustness evidence -> second-person approval.
    The independent `pair_research` readiness profile stays blocked until immutable
    minute and daily shortability evidence exist. Pair versions are rejected by the
    long-only paper ledger until a dedicated atomic spread ledger is implemented.

## Broker execution boundary

Broker destinations contain only a non-secret account reference and non-sensitive
configuration. Gateway URL and HMAC secret are encrypted runtime credentials with
deployment-environment fallback; rotating them does not change the deployment-level
mode lock. Remote gateways must use HTTPS; loopback HTTP is allowed only for local
integration tests. Every
request signs timestamp, nonce, HTTP method, path, and canonical JSON body. The
gateway must attest `environment=sandbox`, echo the client order id, and return an
accepted broker order id.

Activation and release are separate two-person controls. One administrator requests
destination activation and another approves it. Paper-order replay is idempotent per
destination and source order; the creator cannot approve the batch. Before dispatch,
the API rechecks payload SHA-256, sandbox environment, account reference, current
notional cap, destination state, and retry budget. Dispatch is explicit and has no
schedule, so a strategy or RD-Agent run cannot cross the broker boundary by itself.

The provider-neutral execution contract supports TWAP and VWAP parent orders. TWAP
generates evenly spaced slices only inside 10:00-11:20 and 13:30-14:50 China time.
VWAP fails closed unless a point-in-time minute-volume profile supplies unique,
positive weights inside those sessions. Buy quantities must use 100-share lots; sell
orders may place an odd-lot remainder only in the final slice. Slice quantities must
reconcile exactly to the signed parent order. The external provider gateway remains
responsible for real-time participation checks, partial fills, cancel/replace, and
delivering signed provider snapshots and callbacks. The control plane owns comparison,
durable reconciliation evidence, alerts, and the fail-closed destination transition.

Each destination is bound to exactly one paper portfolio. Before the second
administrator may arm it, a fresh broker snapshot must match the portfolio's cash,
NAV, and instrument quantities. Scheduled post-close reconciliation additionally
checks snapshot age, all known and submitted client order ids, returned broker order
ids, terminal rejection/cancel states, and unknown orders or trades. A mismatch is a
state transition, not just a report: the destination becomes `locked_mismatch`, all
dispatch paths reject it, the full expected/observed/difference payload is retained,
and a critical alert is created. Only an explicit disarm followed by a new two-person
activation process can restore eligibility.

Later slices extend the same job and artifact boundaries instead of adding independent
scripts. Portfolio construction may consume only promoted factors, never raw RD-Agent
output or a factor whose latest evaluation failed.

Strategy versions normalize factor weights, preserve each factor's validation-selected
direction and exact factor-evaluation id, and cannot be edited after creation. The
foreign key prevents the evidence used by an immutable strategy from being deleted
or silently replaced. The strategy runner uses the signal
available after day `t` close, executes at day `t+1` open, applies separate buy and
sell costs, caps position weight and daily turnover, and writes daily-return and
position Parquet artifacts. Approval fails closed unless the latest backtest satisfies
the version's tracking-error, drawdown, turnover, information-ratio, Sharpe, Sortino,
robustness, and capacity limits. The approval backtest must contain at least 504
trading days and stay inside every included factor's independent test window.

Robustness is computed by rerunning the full portfolio simulation, not by rescaling a
finished NAV series. The worker independently evaluates double transaction costs,
75% of the configured turnover budget, a 20% narrower Top-K, and removal of the
retention buffer. Each scenario must retain positive annualized excess and satisfy
the strategy risk gates. Capacity uses Qlib's execution-day `$amount` field, the
configured test notional, and a maximum market-volume participation rate to constrain
each instrument's executable weight change before the turnover limit is applied.

The worker also resets positions and reruns the same execution model over overlapping
rolling windows. Historical event stresses are selected from the benchmark's worst
non-overlapping rolling losses inside the approved test interval, then rerun with the
same next-open, cost, turnover, and capacity rules. Strategy approval fails closed
unless the configured minimum number and pass rate of both rolling and event windows
are present in the immutable backtest result.

Paper portfolios may reference only an approved immutable strategy version. A unique
`portfolio + signal date` key makes retries safe: the worker can be restarted or the
operator can submit the same date again without duplicating orders, fills, or NAV.
The Qlib worker ranks the approved factor artifacts after day `t` close, resolves the
next trading date from the snapshot calendar, rejects suspended or unpriced names,
uses raw split-adjusted prices, and joins SW2021 L1 membership as of that trading day.
It enforces cash, lot, turnover, 20-day average amount, execution-day market-amount
participation, and aggregate industry limits. Daily-loss circuit breakers suppress
new buys. A-share up/down limits reject impossible orders, 12% profit takes half once,
20% profit exits fully, and stop-loss rules force realistic next-open exit orders.
Drawdown beyond 10% enters a risk-reduction state; beyond 15% enters liquidation.
These states remain schedulable and cannot be manually overridden until the required
next-open sales finish, after which the portfolio pauses for review.
PostgreSQL applies orders, fills, rejection reasons, point-in-time industries, and
every derived balance in one transaction. Critical pre-trade events or post-trade
position, industry, turnover, daily-loss, and drawdown breaches automatically pause
the portfolio.

The same ledger transaction creates exactly one post-trade review per batch. It
reconciles NAV change to pre-fee instrument contribution and fees, records active
return, execution fill/rejection evidence, risk rules, and the next portfolio state.
Because the review is persisted rather than reconstructed in the browser, a later
position change cannot rewrite historical attribution.

`strategy_allocations` are immutable evidence-backed master portfolios. Each member
pins both an approved strategy version and the successful Qlib backtest whose daily
return artifact was used for covariance, correlation, volatility and risk-contribution
analysis. Risk-parity and inverse-volatility weights are capped per strategy, then
scaled down to the target volatility without leverage. Approval provisions separate
`paper_portfolios`, preserving each strategy's orders, fills and risk state. The master
holds any unused capital as cash, records NAV only when every child has the same trade
date, and propagates member and aggregate drawdown actions without merging ledgers.

The scheduler is a separate stateless service. PostgreSQL owns every schedule,
scheduled slot, lease, retry attempt, and alert, so restarting or horizontally
duplicating the scheduler cannot duplicate a slot. Expired run leases are reclaimed;
the downstream job idempotency key prevents a recovered run from creating a second
job. Paper schedules validate the selected Qlib calendar before enqueueing, while
incremental data schedules use a configurable lookback window to absorb provider
corrections. Misfires fail closed and create critical alerts instead of silently
executing outside their risk window.

Allocation schedule groups are the automation boundary for a master allocation.
Configuration and pause/resume/retire operations lock the group and fail while a child
run is pending or running. A manually paused or closed child suspends its effective
schedule, but the desired state remains available for safe recovery. Conversely,
`risk_reduction_pending` and `liquidation_pending` are runnable safety states: their
next-open execution cannot be disabled by an ordinary schedule action. Retiring a
group preserves schedules and run history for audit instead of deleting them.
The ordinary schedule endpoint cannot mutate a group-managed child. Both ordinary
creation and group creation take the same portfolio-scoped PostgreSQL transaction lock
and reject a second non-retired rebalance schedule, closing the duplicate-enqueue race.

The scheduler is also the durable health observer. At `HEALTH_SNAPSHOT_SECONDS` it
records one PostgreSQL row containing component states and summary counts. Data health
uses the newest immutable Qlib dataset end date and `DATA_FRESHNESS_MAX_DAYS`; worker
health uses bounded HTTP probes; queue health detects excessive backlog, recent
failures, and running jobs older than `STALE_JOB_HOURS`. Missing bootstrap credentials
or datasets are represented separately from runtime degradation. Only degraded or
unavailable component states create deduplicated operational alerts.

Authentication is application-owned because this is a local/private deployment, not
a hosted workspace identity surface. `AUTH_MODE=required` is the deployment default.
Only the initial empty database may create the first administrator; PostgreSQL takes
an advisory transaction lock so concurrent first requests cannot create two. Every
protected request resolves the opaque session token by its SHA-256 hash, rejects
expired/revoked/disabled sessions, and applies a closed-by-default route permission.
Five failed password attempts lock the account for fifteen minutes. Password changes
revoke the account's other sessions. The audit table stores actor, action, route,
status, time, hashed client address, and user agent, but never request bodies,
passwords, raw cookies, or session tokens.
