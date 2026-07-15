# QuantLab: RD-Agent + Qlib Quant Research Platform

Independent Python implementation of a local-first quantitative research platform.
It does not import or modify the previous `E:\projects\rdagent` application.

The repository now contains three connected surfaces:

- `quant_data`: resumable Tushare-to-Parquet-to-Qlib data pipeline;
- `quant_platform`: FastAPI control plane and durable local job worker;
- `web`: React/vinext operations console.

The first milestone is deliberately narrow and production-oriented:

- pull 2024-to-present China equity data through the Tushare-compatible relay;
- use full-market requests (`trade_date`) instead of stock-by-stock daily loops;
- enforce one global request budget across concurrent workers;
- resume safely from PostgreSQL checkpoints with concurrent row locking;
- persist every successful work unit atomically as Parquet;
- build immutable, compacted Parquet snapshots for Qlib conversion;
- verify coverage, empty responses, duplicate rows, and date ranges.

## Quick start

```powershell
cd E:\projects\rdagent-python
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
Copy-Item deploy\.env.example deploy\.env
# Fill DATABASE_URL, then configure the Tushare token without echoing it:
.\.venv\Scripts\python.exe scripts\configure_tushare.py --env-file .env
.\.venv\Scripts\python.exe scripts\configure_tushare.py --env-file deploy\.env
docker compose --env-file deploy\.env -f deploy\compose.yaml up -d

.\.venv\Scripts\quant-data probe
.\.venv\Scripts\quant-data bootstrap --profile core --start 2024-01-01 --end latest
```

## Start the Web system

Terminal 1:

```powershell
cd E:\projects\rdagent-python
.\.venv\Scripts\quant-web --reload
```

Terminal 2:

```powershell
cd E:\projects\rdagent-python\web
npm run dev
```

Open the local URL printed by the Web process. The data center reads real checkpoint,
snapshot, Qlib, and job state from the Python API. Bootstrap actions are durable jobs:
closing the browser does not cancel or lose them.

The **Market Overview** page reads the latest immutable daily snapshot and aggregates
core indices, A-share breadth, 20-day market pulse, industry strength, active ETFs,
index futures, and an editable watchlist. It is deliberately labeled as reproducible
research data rather than live quotes; a future broker or streaming provider can be
added without changing the snapshot-backed research contract.

## Reproducible deployment

The production-shaped Compose stack separates PostgreSQL, FastAPI, the Qlib worker,
the RD-Agent worker and its private Docker sandbox, the Web server, and a same-origin
gateway. It applies versioned Alembic migrations before API startup and pins both
Qlib and RD-Agent to verified source revisions. The RD-Agent image tracks Microsoft's
current `main` commit `4f9ecb005881cddc08df0124a2e894c018007679` and pins its
otherwise-unbounded Pydantic-AI dependency to the compatible v1.90.0 API. Image builds
fail unless the real RD-Agent CLI, MCP connector and LiteLLM provider import together;
apt, GitHub and pip downloads also use bounded retries and timeouts. Worker health is
runtime-aware: the Qlib and RD-Agent containers return HTTP 503 if their assigned
runtime probe is unavailable, so a listening worker cannot create a false-positive
Compose health result.

```powershell
Copy-Item deploy\.env.example deploy\.env
# Configure deploy/.env first.
docker compose --env-file deploy\.env -f deploy\compose.yaml up -d --build
```

Open `http://127.0.0.1:38080`. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for
health checks, migrations, backups, upgrades, and safe shutdown.

For an existing installation, never rebuild or recreate containers while durable work
is active. The fail-closed preflight checks the job queue, work units, schema, service
health, Compose configuration, and rollback disk headroom:

```powershell
.\.venv\Scripts\python.exe .\scripts\release_preflight.py
```

The command always fails closed with a structured report. Invalid Compose variables,
database inspection failures, schema drift, busy work queues, unhealthy services, and
insufficient disk space remain distinct blockers instead of aborting with an opaque
subprocess traceback.
If Compose interpolation itself fails, preflight falls back to read-only Docker
project labels for the current deployment. This preserves independent evidence for
the running database revision, durable work queue and service health while the invalid
new configuration remains a hard blocker.

After the preflight is ready, the supported release path builds, takes a coordinated
backup, upgrades, health-checks, and automatically restores the previous data and
images on failure:

```powershell
.\.venv\Scripts\python.exe .\scripts\release_upgrade.py `
  --backup-root E:\quantlab-backups `
  --confirm-upgrade
```

Run `python scripts/release_upgrade_drill.py` first to exercise the same path in a
self-cleaning isolated Compose project.

On the first visit the deployment asks for the only initial administrator; there
is no repository or image default password. Passwords are Argon2id hashes and the
browser receives a revocable, `HttpOnly`, `SameSite=Strict` session cookie. The
administrator can create researcher, operator, viewer, and additional administrator
accounts from the Tasks page. Every API write is authorized again on the server and
recorded as metadata-only security audit evidence.

Create a coordinated PostgreSQL + data-volume backup and keep the newest fourteen:

```powershell
.\scripts\backup.ps1 -BackupRoot E:\quantlab-backups -RetentionCount 14
```

On Linux use the same Python implementation directly:

```bash
python scripts/backup.py --backup-root /opt/quantlab-backups --retention-count 14
```

The production host templates under `deploy/systemd/` run this coordinated backup
nightly only after the release preflight proves durable work is idle. Docker logs are
also bounded for all eight services through `LOG_MAX_SIZE` and `LOG_MAX_FILES`.
Keep the exact `PLATFORM_SECRET_KEY` in a separate protected recovery store; database
backups contain ciphertext but intentionally exclude that key. Missing or incorrect
keys are a readiness blocker and never cause a fallback to stale dynamic endpoints.
New backup manifests retain only a SHA-256 fingerprint of the key. Restore compares
that fingerprint with the target deployment before creating a staging volume, stopping
services, or changing PostgreSQL; the key itself is never written into the backup.
The public container health probe also returns HTTP 503 until the platform key is
present and every stored ciphertext can be decrypted, so upgrade and restore waits
cannot declare a broken secret store healthy.

Restores require an explicit destructive-operation switch and verify the archive
checksums plus the platform-key fingerprint before stopping application services:

```powershell
.\scripts\restore.ps1 -BackupDirectory E:\quantlab-backups\quantlab-YYYYMMDDTHHMMSSZ -ConfirmRestore
```

Run the self-cleaning, two-project restore acceptance after migrations or storage changes:

```bash
python scripts/restore_drill.py
```

The Qlib Research page uses the installed WSL runtime to run a reproducible
Alpha158 + LightGBM baseline. It automatically splits the available trading calendar
into train, validation, and test segments, records IC/RankIC, and runs a Top-K backtest
with explicit transaction costs. Experiment artifacts are stored under
`data/artifacts/qlib/<job-id>/`.

Day and minute research are separate contracts. Execution snapshots can be converted
with the durable `minute_qlib` job after the minute download finishes; conversion can be
retried without downloading bars again. The Qlib page then offers a bounded minute-factor
scan over momentum, VWAP deviation, volume surprise, range pressure, and realized
volatility for configurable 5/15/30-minute horizons and transaction costs. Minute results
are exploratory records only: they do not enter the daily Alpha158/RD-Agent strategy,
promotion, approval, or paper-trading path automatically.

The RD-Agent page runs bounded factor loops only after the runtime, Docker sandbox,
LLM credential, Qlib snapshot, and date coverage checks pass. Generated code and H5
factor values enter the factor registry as candidates. An independent Qlib evaluator
selects direction on the validation window, recomputes out-of-sample IC/RankIC,
turnover, correlation, and cost-adjusted return, and fails closed on missing evidence.
Only a gate-passed candidate can be promoted. Standalone research keeps the manual
promotion action; an autonomous campaign may perform a policy-recorded promotion after
deterministic Qlib ranking, but it still cannot approve a strategy or enter paper trading.

The **Automatic Research** page creates one durable campaign that pins a reproducible
Qlib dataset and advances through RD-Agent generation, Qlib factor gates, deterministic
ranking, immutable baseline backtest, bounded parameter experiment, challenger backtest,
and champion selection. PostgreSQL leases and idempotent child-job keys make every stage
restart-safe; the Web can pause future stages, resume, cancel, or retry only the failed
child stage. The champion remains `draft` until a human completes the existing strategy
approval. Only after that approval does the campaign create its paper portfolio and
after-close schedule. No campaign can enable broker live mode.

The same page can also persist a **Continuous Research Program**. A program follows
one verified Qlib dataset lineage, checks for its newest immutable descendant, waits
for a configurable number of new trading days, derives fixed train/validation/test
windows from the real trading calendar, and creates at most one campaign per dataset
identity. PostgreSQL leases, a unique program/dataset key and per-program concurrency
limits prevent duplicate or overlapping research after restarts. Insufficient history
is a waiting state rather than permission to shorten the 504-day out-of-sample gate.

The RD-Agent and Strategy Backtest pages share server-owned, versioned recipes derived
from the design document. Selecting index enhancement or A-share swing/trend loads the
documented research objective and factor guidance, then pins the recipe version and its
portfolio/risk/liquidity limits into the immutable strategy version. Unknown or stale
recipe versions are rejected; selecting a recipe never bypasses independent Qlib
evaluation, factor admission policy, backtest gates, or strategy approval.

The Strategy Backtest page builds immutable versions from promoted factors only.
Each factor is pinned to the exact independent evaluation that selected its direction.
Its Qlib runner composes cross-sectionally winsorized/z-scored factors, executes at
the next trading day's open, caps position weight and daily turnover, applies
asymmetric transaction costs, and records benchmark-relative return, tracking error,
information ratio, Sharpe, Sortino, tail loss, drawdown, turnover, daily NAV, and
holdings. Every governed run also reruns double-cost, tighter-turnover, narrower-Top-K,
and zero-buffer scenarios. Execution-day amount limits the tested portfolio notional
to the configured market-volume participation. Strategy approval is separate from
process success and fails closed unless at least 504 out-of-sample trading days,
factor test-window alignment, robustness pass rate, and capacity fill rate all satisfy
the immutable version limits.

The same governed run resets and reruns the portfolio across overlapping 252-day
out-of-sample windows and the five worst non-overlapping 20-day benchmark periods.
Rolling stability and historical-event pass rates are persisted with the backtest and
are mandatory approval evidence, not browser-side summaries.

The Pair Trading page implements the document's statistical-arbitrage satellite
strategy as a separate governed family. An immutable ETF/stock pair version pins the
formation window, correlation and Engle-Granger thresholds, Kalman parameters,
entry/exit/stop Z-scores, holding period, exposure, costs, borrow rate, capacity and
approval gates. Daily signals execute only in the next trading day's 10:00-11:20 or
13:30-14:50 minute windows. Both legs fill atomically or both reject when either leg is
suspended, price-limited, over capacity, missing a bar, or lacks explicit shortability
evidence. DuckDB reads only the selected instruments and period from immutable Parquet
instead of loading the complete minute lake.

Approval requires a native pair-engine result, rolling cointegration and robustness
passes, a closed final spread, SHA-256 provenance for daily/minute/shortability inputs,
and an operator different from the version creator. The Web page exposes the same
`pair_research` readiness checks. Approved pair versions then enter a dedicated
two-leg PostgreSQL paper ledger, never the long-only ledger. Every daily batch pins a
starting-state hash and immutable daily/minute/shortability provenance; the control
plane independently reconciles cash, signed leg quantities, NAV, gross/net exposure,
turnover, fees and borrow cost before atomically committing both legs. The scheduler
can run this ledger after the close, while critical-risk and liquidation states remain
fail closed. The separate `pair_paper` readiness profile requires one and the same
ledger to have an active schedule, five recent reviewed trading days, no unresolved
critical event, no stale batch, and no recent failed batch.

Execution evidence is now downloaded through two durable data-center jobs.
`margin-eligibility` requests the full eligible universe once per trading date and
persists explicit `shortable=true` evidence. `core-intraday` accepts explicit codes or
automatically selects the four major A-share indices, broad/industry/gold/bond ETFs,
historically active IF/IC/IM/IH continuous and deliverable contracts, liquid stocks and
active exchange ETF options from downloaded masters. It requests monthly symbol windows
(bounded fortnight windows for futures), rejects Tushare row-limit truncation and invalid
OHLCV, and creates one immutable `pair_execution` snapshot containing both minute bars,
selection evidence and the matching margin dataset. The Web task center exposes the same
configuration through `/api/jobs/margin-eligibility` and `/api/jobs/core-intraday`.

The supplemental task center also has independently resumable bundles for public funds,
macroeconomics, futures mappings/calendars, options and convertible-bond lifecycle data,
Hong Kong and US financials, international indices and foreign exchange. See
[`docs/DATA_CENTER.md`](docs/DATA_CENTER.md) for the exact interfaces and retry boundary.

Daily and execution snapshots now carry stable lineage IDs, parent-manifest hashes,
generation numbers, and exact source-unit identities. A successor is eligible for
automatic use only when every ancestor unit remains present with the same SHA-256 and
row count. Ingestion-code digests are part of the source lineage, so planner, provider,
validation, or snapshot-compaction changes cannot silently join an older lineage. Qlib
derives a separate stable dataset lineage from that verified source.
The Qlib builder digest and field contract are part of that lineage, so an implementation
change requires explicit re-approval. Legacy or rewritten snapshots remain usable only
in explicit pinned mode.

Every formal Qlib strategy backtest also produces a deterministic execution replay.
That replay uses the immutable strategy version's exact stop-loss, staged take-profit,
daily-loss, portfolio-drawdown, industry and participation settings. Approval rejects a
result whose replay is missing, whose thresholds differ from the version, or whose
replayed drawdown violates the configured gate. The embedded replay and the independent
`execution_replay.json` artifact must match the same canonical SHA-256 stored in
provenance. Before a Worker reports success and again before approval changes strategy
state, QuantLab re-reads `manifest.json` and the replay artifact, verifies their raw and
canonical digests, and compares the manifest config, benchmark, periods, factor weights,
directions, and code hashes with the immutable strategy version. It also re-hashes the
actual factor code and value artifacts, so replacing a promoted factor file after the
backtest invalidates Worker success or approval. This keeps research
metrics and the paper-trading execution contract independently visible instead of
silently assuming that a generic Top-K Qlib strategy exercised every production risk rule.

The Paper Portfolio page turns an approved strategy into a durable simulated
portfolio. It creates one idempotent batch per signal date, executes against the next
Qlib trading-day open with lot, cash, suspension, slippage, fee, position, turnover,
20-day liquidity, and market-amount participation constraints. SW2021 L1 industry
membership is resolved point-in-time from the immutable source snapshot. Daily-loss
circuit breakers, A-share price limits, staged 12%/20% take-profit, stop-loss exits,
10% drawdown reduction, 15% drawdown liquidation, and industry concentration rules
are evaluated before execution. Drawdown actions use non-overridable pending states,
so a safety pause cannot prevent the next-open risk sale. Orders, fills, rejection
reasons, positions, cash, NAV, and risk events are committed atomically to PostgreSQL.
Repeating the same request cannot duplicate a fill.
Both ordinary and pair-paper portfolios expose `pinned` and `latest_compatible` data
policies. The scheduler resolves the newest compatible descendant that covers the
signal/trade date, then persists its Qlib identity and execution-manifest hash on the
batch before enqueueing work. Worker results must return the same pinned evidence;
cross-lineage, rewritten, or silently substituted data fails closed.

Risk events have a separate acknowledgement and resolution lifecycle. Resolution stores
the responsible operator, timestamp, and written conclusion. Pending reduction or
liquidation, active batches, remaining liquidation positions, or excessive exposure
fail closed. A paused portfolio cannot be reactivated while any critical event remains
open or acknowledged.

Each successful paper batch also writes one immutable post-trade review in the same
database transaction. It reconciles starting and ending NAV, net and pre-fee market
PnL, benchmark-relative return, turnover, fees, fill/rejection rates, risk rules, and
the best/worst instrument contributors. Reviews are visible on the portfolio page and
cannot drift from the ledger they describe.

The same page can combine two to ten approved strategies into a governed master
allocation. It aligns their Qlib daily-return artifacts, rejects excessive pairwise
correlation, and supports risk-parity, inverse-volatility, or fixed allocation. A
second administrator must approve the result before capital is provisioned into
independent child paper ledgers. Target-volatility scaling retains unused capital as
cash instead of adding leverage. Child daily closes roll up automatically; an 8%
member drawdown, 10% master drawdown, or 15% master drawdown propagates reduction or
liquidation state to every affected child ledger. The same audited lifecycle applies at
master level; completing an event leaves the allocation paused, and resuming it requires
all critical events and every child ledger to be safe.

System Settings also owns a versioned default template for new multifactor strategies.
It exposes portfolio construction, stop-loss/take-profit, drawdown actions, industry
and liquidity constraints, capacity, costs, rolling validation and historical stress
gates as validated Web fields. Every save requires an operator reason and appends a
revision; creating a strategy copies the complete template into the immutable strategy
version. Changing the template never mutates an approved or running version.

Other business parameters are configured at the governed object that owns them instead
of being hidden constants. The Web forms expose master-allocation volatility,
correlation, weight and drawdown thresholds; pair-strategy capacity, cost, Kalman and
acceptance gates; data and portfolio schedule windows; simulated slippage; and broker
sandbox slicing and participation limits. Allocation, strategy-version, schedule and
broker-destination records persist the submitted values. Authentication, secret values,
live-mode enablement and non-loosenable deployment caps remain server-side boundaries.

The broker boundary is deliberately sandbox-only. `BROKER_MODE=live` is rejected
during configuration parsing. The sandbox Gateway URL and HMAC secret can be stored
only as encrypted runtime settings; read APIs expose the endpoint hostname but never
the URL or key. `BROKER_MODE` remains a deployment-time hard lock and cannot be
changed from the Web console. An administrator may replay filled paper orders into a durable
outbox only after a signed sandbox gateway is healthy and the destination has been
armed by a second administrator. A different administrator must also approve the
batch before explicit dispatch. Payload hashes, idempotency keys, notional caps,
bounded retries, and append-only broker events make every rehearsal fail closed and
auditable. Each parent order also carries a reconciled A-share TWAP or evidence-backed
VWAP slice plan restricted to 10:00-11:20 and 13:30-14:50, Asia/Shanghai, with
100-share buy lots. No automatic live-order schedule exists.

Every sandbox destination is bound one-to-one to a paper portfolio. Activation
requires a fresh matching account snapshot, and a durable daily reconciliation may
compare broker cash, equity, positions, accepted order ids, unknown orders/trades,
and snapshot age. Any difference atomically changes the destination to
`locked_mismatch`, blocks further dispatch, persists the evidence, and creates a
critical alert. Reconciliation never auto-rearms a destination.

The provider-side implementation now includes a Windows QMT/MiniQMT sandbox gateway.
It uses the official XtQuant stock-trading API, keeps scheduled parent/child execution
state and replay nonces in PostgreSQL, recovers idempotently through a 22-byte QMT
order remark, and exposes normalized signed asset/position/order/trade snapshots. The
gateway still requires the user's MiniQMT installation and simulation-account access;
it refuses live mode. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

The QMT runtime also enforces the signed participation cap from a completed native
one-minute bar and a fresh full-tick quote. Oversized slices are carried forward rather
than blindly sent. Partial fills time out into cancel-then-replace processing, with a
bounded price move and replacement count; every attempt and provider callback is
durable and survives control-plane recovery.

The Tasks and Alerts page manages PostgreSQL-backed daily schedules for incremental
data sync and after-close portfolio rebalancing. Schedule slots use expiring leases,
job idempotency keys, and explicit misfire windows, so scheduler restarts do not lose
or duplicate work. Failed jobs, missed schedules, and portfolio risk events enter an
auditable alert inbox; operators can acknowledge or resolve them, and deployments may
optionally forward open alerts to an internal webhook with bounded retries.
Administrators can replace or disable that webhook from System Settings. The URL is
encrypted in PostgreSQL, the API exposes only its hostname, and the scheduler reads
the latest value before every delivery pass, so no container restart is required.
Remote receivers must use HTTPS; `ALERT_WEBHOOK_URL` remains only a bootstrap fallback.

An approved multi-strategy allocation can create or update all child-portfolio
schedules in one database transaction. Each schedule stores the operator's desired
state separately from its effective safety state: pausing a portfolio suspends its
schedule without erasing the intent to resume, while pending risk-reduction or
liquidation schedules remain runnable and cannot be manually paused or retired.
PostgreSQL advisory locks serialize ordinary and allocation-managed schedule creation,
so one paper portfolio cannot acquire two non-retired rebalance schedules.

The scheduler also writes durable system-health snapshots at a configurable interval.
Each snapshot records PostgreSQL, Qlib and RD-Agent worker reachability, RD-Agent
runtime readiness, credentials, Qlib data freshness, and queue backlog/staleness.
Degraded or unavailable components project deduplicated alerts, while the Operations
page shows current evidence and history rather than browser-only polling.

`core` downloads reference data plus daily prices, adjustment factors, daily
valuation/liquidity fields, suspensions, price limits, index prices, and selected
index membership. `research` adds flows/rankings, exchange-traded funds,
point-in-time disclosure plans, and historical SW2021 industry membership.
`full` adds financial statements, name/ST history, dividends, repurchases,
unlocks, pledges, shareholder changes, announcements, and daily news windows.
Per-instrument endpoints never block the core snapshot.

## Commands

```powershell
quant-data probe
quant-data bootstrap --profile core --start 2024-01-01 --end latest
quant-data status
quant-data retry-failed
quant-data verify
quant-data snapshot --name cn-2024-2026
quant-data build-qlib --snapshot cn-2024-2026
```

## Storage layout

```text
PostgreSQL
  work_units                     # downloader checkpoints and leases
  jobs                           # durable Web/worker jobs
  research_runs                  # bounded RD-Agent runs and runtime evidence
  factor_candidates              # code/value artifacts and lifecycle state
  factor_evaluations             # versioned Qlib metrics and gate decisions
  research_events                # append-only research audit trail
  research_programs              # continuous same-lineage research policies
  research_program_events        # append-only lifecycle, trigger, and failure audit
  research_campaigns             # restart-safe autonomous research state machines
  research_campaign_events       # append-only campaign stage evidence
  strategies                     # stable strategy identity
  strategy_versions              # immutable configuration and approval state
  strategy_factors               # promoted factor weights and directions
  backtest_runs                  # governed strategy backtests and risk metrics
  strategy_events                # append-only strategy approval history
  strategy_allocations           # governed low-correlation master portfolios
  strategy_allocation_members    # risk budgets and provisioned child ledgers
  strategy_allocation_nav/events # aggregate NAV and group circuit breakers
  allocation_schedule_groups     # atomic automation policy per master allocation
  allocation_schedule_members    # child portfolio to durable schedule mapping
  paper_portfolios               # approved-strategy simulated accounts
  portfolio_batches              # idempotent after-close/next-open workflow
  paper_orders / paper_fills     # immutable simulated execution ledger
  paper_positions / portfolio_nav# current holdings and daily NAV history
  risk_events                    # acknowledged/resolved breaches and safe-recovery evidence
  portfolio_reviews              # immutable post-trade PnL/execution/risk review
  broker_destinations            # sandbox-only mappings and two-person arming
  broker_order_outbox            # hashed idempotent sandbox delivery ledger
  broker_events                  # append-only activation/release/delivery evidence
  broker_reconciliations         # expected/observed account state and mismatch proof
  schedules / schedule_runs      # leased daily automation and restart evidence
  alerts                         # deduplicated inbox, delivery, acknowledgement
  users / auth_sessions          # local RBAC identities and revocable sessions
  audit_events                   # metadata-only login, denial, and mutation trail
  mlflow tables                  # Qlib experiment metadata

data/
  raw/                         # optional compressed provider responses
  units/<dataset>/*.parquet    # atomic, resumable work-unit outputs
  snapshots/<name>/
    parquet/<dataset>/...
    manifest.json
    verification.json
```

Unit files are intentionally kept separate from compacted snapshots. A failed
snapshot build never damages completed downloads, and a new Qlib snapshot can be
generated without calling the provider again.

Snapshot succession is append-only and cryptographically evidenced. `manifest.json`
records `lineage_id`, `lineage_generation`, `parent_snapshot`,
`parent_manifest_sha256`, and each dataset's `source_units`. This lineage is evidence
for controlled rollover, not permission to overwrite an existing immutable snapshot.

PostgreSQL is the only control-plane database. Parquet remains the canonical
historical data lake, Qlib Bin is the derived research format, and large experiment
artifacts remain under `data/artifacts` until object storage is introduced.
Tests use a separate PostgreSQL database selected through `TEST_DATABASE_URL` so
verification never contaminates production task or experiment state.

Qlib datasets created before schema revision `0011_execution_risk_state` must be
rebuilt from their same-name immutable snapshots. The builder now requires `stk_limit`
and emits `$up_limit` / `$down_limit`; paper execution fails closed if either field is
missing.

## Request policy

The shared note describes an annual plan near 100 requests/minute and warns that
heavy overrun triggers a 3-10 minute cooldown. This project defaults to 90
requests/minute. Four workers overlap network latency, but one process-wide rate
gate controls the actual request start rate.

## Qlib boundary

The compacted snapshot is the canonical research data lake. Bootstrap builds a
per-symbol normalized staging view and calls Qlib's official
`scripts/dump_bin.py --file_suffix .parquet` by default. On this Windows host it
uses the Qlib environment in WSL. RD-Agent consumes the resulting Qlib binary
snapshot and never calls Tushare during an experiment. Use `--no-build-qlib`
only when downloading on a machine where Qlib is not installed.

Minute snapshots use `python -m quant_data.cli build-minute-qlib` and produce a separate
`1min` calendar and feature store. API validation rejects a minute dataset anywhere the
daily Alpha158, RD-Agent, allocation, backtest, or paper engine expects `day`; the minute
factor lab likewise rejects daily datasets. This prevents silent frequency mixing.

## Durable task operations

`GET /api/jobs` supports status/type filters plus limit/offset pagination and exposes the
matching total through `X-Total-Count`. The Web task center uses the job detail and bounded
log-tail endpoints instead of rendering an unbounded page. Failed or cancelled jobs retain
their parameters and can be retried. Queued jobs cancel immediately; running jobs record a
cancellation request and the owning worker terminates its child process cooperatively,
then preserves the cancelled record and log for audit.
