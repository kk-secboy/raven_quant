# QuantLab deployment

## Deployment boundary

The deployment stack contains eight isolated services:

- `gateway`: the only HTTP entry point; proxies `/api/*` to FastAPI and all other
  traffic to the Web console;
- `web`: the production vinext/React server;
- `api`: the stateless FastAPI control plane and Alembic migration runner;
- `scheduler`: leased PostgreSQL schedule runner, alert projector, and optional
  webhook delivery service;
- `worker`: downloads data and runs pinned Qlib/LightGBM and factor-evaluation jobs;
- `rdagent-worker`: runs only bounded `rdagent_factor` jobs from a pinned RD-Agent commit;
- `rdagent-docker`: a private Docker-in-Docker daemon used only by RD-Agent code
  sandboxes; it has no published port and the host Docker socket is never mounted;
- `postgres`: durable control-plane and MLflow metadata.

Market and financial history remains on the shared `quantlab_data` volume as
Parquet. Qlib Bin and experiment artifacts also use that volume. PostgreSQL does
not replace the historical data lake.

## First deployment

Requirements: Docker Engine with Compose v2, privileged-container support for the
private DinD service, and at least 16 GB of free memory. The Qlib and RD-Agent worker
images are intentionally large because they contain pinned Microsoft source trees
and native research dependencies.

```powershell
Copy-Item deploy\.env.example deploy\.env
```

Edit `deploy/.env` before startup. Use a long URL-safe PostgreSQL password and a
unique Fernet `PLATFORM_SECRET_KEY`. Keep both bind addresses at `127.0.0.1` unless
a separate authenticated TLS reverse proxy protects the service. Environment-based
Tushare and model credentials remain supported as a deployment fallback, but the
normal operator path is the administrator-only **System Settings** page.

Store `PLATFORM_SECRET_KEY` in an off-host secret manager or other protected recovery
location. The PostgreSQL backup contains encrypted runtime credentials but not this
key. Do not replace the key in `deploy/.env` as an ad-hoc rotation: if a database
override exists and cannot be decrypted, Workers, Scheduler health, alerts, and broker
operations fail closed instead of silently using stale environment values.

The settings page validates Tushare with a read-only trade-calendar request, then
encrypts the Token before storing it in PostgreSQL. Model API keys and the optional
alert Webhook use the same encrypted store. Secret values are never returned by the
API or rendered after saving; alert status exposes only the receiver hostname.
Workers read the latest credentials when each job starts, and the scheduler reads the
latest alert endpoint before each delivery pass, so these changes require neither an
image rebuild nor a container restart. A blank saved alert endpoint explicitly
disables Webhook delivery. Remote receivers must use HTTPS; HTTP is accepted only for
local development hosts.

Builds use the official Python and Debian package sources by default. Deployments in
regions where those sources are slow may override `PIP_INDEX_URL` and
`DEBIAN_MIRROR` in `deploy/.env`; these settings affect image construction only and
do not change the pinned Qlib or RD-Agent source commits.

The deployment example uses the official HTTPS Tushare endpoint. Configure and
validate the token through the hidden-input helper instead of placing it on a command
line or sending it through chat:

```powershell
.\.venv\Scripts\python.exe .\scripts\configure_tushare.py `
  --env-file .\deploy\.env
```

The helper performs a read-only `trade_cal` request, rejects an invalid credential,
then atomically replaces only `TUSHARE_API_URL` and `TUSHARE_TOKEN`. Recreate `api` and
`worker` after changing environment files. Failed child jobs retain the actionable
exception extracted from their log tail, while the token itself is never copied into
job payloads or error text.

```powershell
docker compose --env-file deploy\.env -f deploy\compose.yaml up -d --build
docker compose --env-file deploy\.env -f deploy\compose.yaml ps
```

Open `http://127.0.0.1:38080`. A healthy deployment must report:

The first page is a one-time administrator setup screen when the database contains no
users. Choose a unique password of 12-256 characters containing a letter, number, and
symbol. No default administrator password exists in source, images, or environment
files. After setup, the unauthenticated bootstrap path is permanently closed.

- `/api/health`: `status=ok`, `database=postgresql`, `worker_mode=external`, plus
  `runtime_secret_storage=ok` and the number of ciphertext rows validated; it returns
  HTTP 503 when `PLATFORM_SECRET_KEY` is absent, wrong, or cannot decrypt any row;
- `/api/qlib/status`: Qlib and LightGBM versions from the worker container;
- `/api/rdagent/status`: pinned RD-Agent version, Docker sandbox, credential presence,
  and configured limits from `rdagent-worker`;
- each worker's private `/health` endpoint validates the runtime assigned to that
  process. The Qlib worker therefore cannot be healthy with a broken Qlib import, and
  the RD-Agent worker returns HTTP 503 when its CLI/runtime probe is unavailable.
  Missing research credentials remain visible in `/api/rdagent/status` as a separate
  operator-remediable readiness blocker rather than being hidden by container health;
- `/api/strategies` and `/api/backtests`: PostgreSQL-backed immutable strategy
  versions and governed Qlib backtest records;
- `/api/portfolios`: approved-strategy paper portfolios with durable batches,
  orders, fills, positions, NAV history, risk events, and immutable post-trade reviews;
- `/api/schedules`, `/api/schedule-runs`, and `/api/alerts`: durable automation,
  retry evidence, alert delivery state, and operator acknowledgement;
- `/api/operations/health`: latest durable health observation plus historical
  PostgreSQL, worker, RD-Agent, data-freshness, credential, and queue evidence;
- `/api/operations/readiness`: evidence-backed go/no-go profiles for research,
  continuous paper operation, and the optional broker sandbox;
- `/api/broker`: administrator-only sandbox readiness, two-person destination state,
  hashed outbox, and delivery history. It never enables live trading;
- all eight Compose services running, with PostgreSQL, API, scheduler, both workers, and
  `rdagent-docker` healthy.

Except for `/api/health` and the authentication bootstrap/login state, API routes
require a valid session. Roles are enforced server-side:

- `viewer`: read-only system, research, portfolio, task, and alert views;
- `researcher`: viewer access plus Qlib/RD-Agent experiments and strategy drafting;
- `operator`: viewer access plus data initialization, portfolios, schedules, and
  alert handling;
- `admin`: all permissions, user lifecycle, approvals, and security audit access.

Use the Tasks page to create or disable users, change the current password, and inspect
the audit trail. Five failed logins lock an account for fifteen minutes. Disabling a
user revokes its sessions immediately, and the last active administrator cannot be
disabled. For HTTPS deployments set `AUTH_COOKIE_SECURE=true`; keep the gateway bound
to localhost unless a separately managed TLS reverse proxy protects it.

The default RD-Agent split is 2018-2021 training, 2022-2023 validation, and
2024-present out-of-sample testing. A Qlib snapshot containing only 2024-present
data is valid for recent backtests but is deliberately rejected for this full
research workflow.

Strategy backtests run on the Qlib worker, not in the API process. They use next-open
execution, explicit buy/sell costs, position limits, turnover throttling, benchmark
returns, tracking error, information ratio, and drawdown. A successful process is
not automatically an approved strategy. Approval additionally requires at least 504
trading days within the included factors' independent test windows, numeric Sharpe
and Sortino evidence, the configured share of four parameter/cost pressure scenarios,
and sufficient executable capacity at the configured test notional and volume
participation limit. Missing `$amount` data or any missing gate metric fails closed;
the configured risk gate and a recorded operator approval are both required.

Each governed backtest additionally produces `rolling.json` and `event_stress.json`.
The default gate requires at least three 252-day rolling windows and five worst
non-overlapping 20-day benchmark event windows. Both suites reset the portfolio and
rerun the execution model; missing or insufficient evidence prevents approval.

After at least two strategy versions are approved, the Paper Portfolio page can create
a governed multi-strategy allocation. The server reads each selected backtest's
`daily_returns.parquet`, requires the configured common lookback, rejects pairwise
correlation above the hard limit, and calculates risk-parity, inverse-volatility, or
fixed weights. Target volatility may leave part of the capital in the master cash
reserve; leverage is never introduced. The creator cannot approve the allocation.
Approval by a second administrator atomically creates one child paper ledger per
strategy. Their daily NAVs must align before a master NAV is written. Member drawdown
above 8%, master drawdown above 10%, and master drawdown above 15% propagate
risk-reduction or liquidation-pending state to child ledgers.

Paper rebalances also run on the Qlib worker. A signal dated at day `t` is resolved
against the snapshot calendar and executed at the next trading-day open. The API
returns the existing batch when the same portfolio/date is submitted twice. Ledger
updates are atomic in PostgreSQL; a worker failure leaves the previous cash,
positions, and NAV intact. Hard risk-limit breaches persist an open risk event and
pause the portfolio before another batch can be accepted.

Acknowledgement records who owns a risk event but does not clear the deployment gate.
Resolution requires a written conclusion and fails closed while a risk batch is active,
reduction exposure remains above its approved limit, or liquidation positions remain.
Completing an event leaves the affected portfolio or master allocation paused. A
separate resume action is accepted only after every critical event is resolved; master
resume also checks every child portfolio and resumes them atomically.

The paper worker requires the immutable Parquet snapshot with the same name as the
Qlib dataset. It resolves SW2021 L1 industry membership at the execution date and
uses Qlib `$amount` for the 20-day liquidity floor and per-order participation cap.
The Qlib build also requires snapshot `stk_limit` data and emits `$up_limit` and
`$down_limit`; rebuild pre-0011 datasets before enabling a portfolio schedule.
Missing industry evidence rejects new buys and pauses portfolios that already hold an
unclassified position. Daily-loss, stop-loss, take-profit, and industry thresholds are
part of the immutable approved strategy configuration.

The scheduler checks PostgreSQL every 15 seconds by default. Each due time becomes a
unique leased run record before any worker job is created. A scheduler crash is safe:
the lease expires, another instance reclaims the same run, and the job idempotency key
returns the original job if it was already inserted. `ALERT_WEBHOOK_URL` is a
deployment-time fallback only. The normal path is the administrator-only **System
Settings** page, whose encrypted database value takes effect on the next scheduler
tick without a restart. Failed deliveries are retained and retried up to ten times.
The Web alert inbox remains available even when Webhook delivery is disabled.

The same scheduler writes durable health snapshots every
`HEALTH_SNAPSHOT_SECONDS` (default 300). `DATA_FRESHNESS_MAX_DAYS` controls when the
newest Qlib dataset becomes stale, and `STALE_JOB_HOURS` controls when an unfinished
job is treated as stuck. Bootstrap-required states remain visible without paging as a
runtime outage; degraded and unavailable components create deduplicated alerts. Review
the Operations health history after restarts instead of relying only on current
container status.

Container health is necessary but not sufficient for formal use. Open **Tasks** and
review **Formal deployment acceptance**. The research profile fails closed until the
database schema, administrator authentication, Tushare verification, all five data
stages, a reproducible 504-day Qlib dataset, RD-Agent runtime, incremental schedule,
fresh health history, and critical-alert queue all pass. Paper operation additionally
requires an approved governed strategy, an active portfolio and schedule, at least five
recent trading-day reviews, and no open or acknowledged risk event. Broker sandbox acceptance inherits
both profiles and requires a signed runtime attestation, two-person destination,
scheduled reconciliation, recent matched snapshot, and no failed delivery.
The separate **multi-strategy paper** profile additionally requires a live
low-correlation allocation, fully provisioned risk-budget child ledgers, five aligned
master-NAV days, a complete active child-schedule group, and no open or acknowledged
group-level critical event.

Before rebuilding, recreating, migrating, or stopping an existing deployment, run the
release preflight from the project root:

```powershell
.\.venv\Scripts\python.exe .\scripts\release_preflight.py
```

The command exits with code 2 unless Compose is valid, PostgreSQL and all eight
services are healthy, the database revision is either current or has a recognized
forward-only Alembic path to the code head, disk headroom is available, and every
durable job/work unit is idle with no failed unit. An unknown, divergent, or newer
database revision fails closed. Do not use a restart to bypass an active initialization.
Save machine-readable evidence with
`--report artifacts\release-preflight.json`. This operational preflight protects the
currently running installation; the in-app readiness profiles separately decide
whether its data and research evidence are fit for use.

Preflight command failures are themselves structured blockers. In particular, an
unset required environment variable or invalid Compose file produces JSON evidence
and exit code 2; preflight does not continue through the invalid Compose context. It
uses read-only Docker project/service labels to inspect the already-running containers,
database revision, durable queue and health independently, so a missing new variable
does not hide whether the old deployment is safe to upgrade. It does not collapse the
result into a Python traceback. Command diagnostics retain bounded stderr without
echoing the command line or env-file contents.

## Broker sandbox

The default `BROKER_MODE=disabled` is a hard lock. The only accepted enabled value is
`sandbox`; `live` causes startup configuration to fail. To test a provider gateway,
set `BROKER_MODE=sandbox`, then use the administrator-only **System Settings** page to
save an absolute Gateway URL and a random HMAC secret of at least 32 characters. Both
values are encrypted in PostgreSQL and every broker operation reads the latest pair,
so rotating or disabling them requires no API or Scheduler restart. Environment values
`BROKER_GATEWAY_URL` and `BROKER_HMAC_SECRET` remain bootstrap fallbacks. The Web UI
cannot change `BROKER_MODE`, and no runtime setting can enable live trading. Use HTTPS
unless the gateway hostname is loopback or the private Compose service name. Keep the
provider login, certificate, and trading password inside the separate broker gateway
process.

After the gateway reports a signed sandbox health response, two different admin
accounts must request and approve destination activation. A successful paper batch
may then be staged; its creator cannot approve it. Dispatch remains an explicit admin
action, enforces `BROKER_MAX_ORDER_NOTIONAL`, and stops after
`BROKER_MAX_ATTEMPTS`. Failed deliveries stay in PostgreSQL for inspection and an
explicit retry. There is no automatic broker dispatch schedule and no live mode in
this release.

Each sandbox destination pins an execution policy. TWAP accepts 5-30 minute slice
intervals and creates orders only in the configured safe A-share sessions. VWAP also
requires a historical minute-volume profile; missing or out-of-session evidence
prevents staging. The signed payload contains every scheduled slice, its quantity,
participation cap, timezone-aware timestamp, and parent-order reconciliation.

Bind each sandbox destination to one paper portfolio. Activation approval is rejected
until a fresh signed account snapshot matches paper cash, NAV, and positions. Create a
`broker_reconcile` schedule after 15:10 China time for continuous checks. The run also
detects stale snapshots, missing submitted orders, unknown broker orders/trades,
broker-order-id changes, and terminal rejection/cancel states. Any difference locks
the destination, persists a `broker_reconciliations` record, and opens a critical
alert; a later matching snapshot does not automatically rearm it.

### Windows QMT sandbox gateway

XtQuant trading must run on the Windows host beside a logged-in MiniQMT client. It is
not supported by the Linux worker image. Copy `deploy/qmt-gateway.env.example` to the
ignored `deploy/qmt-gateway.env`, configure the MiniQMT `userdata_mini` path, simulation
stock account, PostgreSQL URL, account reference, and the same random HMAC secret used
by the control plane. Keep `QMT_ENVIRONMENT=sandbox`; any other value fails startup.

```powershell
.\scripts\start_qmt_gateway.ps1
```

The gateway applies pending schema migrations, connects and subscribes to MiniQMT,
then listens on port 8790. If it binds `0.0.0.0` for Docker access, restrict that port
with Windows Firewall. In `deploy/.env`, set:

```dotenv
BROKER_MODE=sandbox
# Configure URL/HMAC in System Settings after startup. Environment fallback:
# BROKER_GATEWAY_URL=http://host.docker.internal:8790
# BROKER_HMAC_SECRET=the-same-random-secret
```

Recreate `api` and `scheduler` only when changing the deployment-level `BROKER_MODE`
or environment fallback. Settings-page URL/HMAC changes hot-load without recreation.
Then use the administrator Broker panel to probe the gateway. Requests are timestamped,
HMAC signed, and protected by PostgreSQL-backed
nonce claims. Parent orders and every execution slice are durable; a restart retries a
previously claimed slice by its deterministic QMT remark, while a never-submitted slice
older than the configured lateness bound is marked failed instead of being sent late.
Each due slice also requires a completed QMT one-minute bar and a fresh best bid/ask.
The configured participation cap moves excess quantity into the next slot. Submitted
and partially filled attempts are canceled after `QMT_CANCEL_AFTER_SECONDS`; the
remaining board-lot quantity may be replaced at most `QMT_MAX_REPLACEMENTS` times and
never beyond `QMT_MAX_REPRICE_BPS`. Attempts, callback events, volume/quote evidence,
cancel requests, and replacement identifiers are all retained in PostgreSQL.
Do not configure a real-money account until a separately authorized live phase exists.

Before enabling the control-plane sandbox, run the read-only acceptance check in the
same shell that contains `BROKER_HMAC_SECRET`. It verifies health, the configured
account snapshot, and live minute/full-tick evidence without submitting an order:

```powershell
.\.venv\Scripts\python.exe .\scripts\accept_qmt_gateway.py `
  --account-ref QMT-SANDBOX-1 `
  --instrument SH600000
```

## Safe release upgrade

Never run a manual `docker compose up --build` against an existing installation. After
the initialization and all durable work are idle, use the coordinated upgrade command:

```bash
python3 scripts/release_upgrade.py \
  --backup-root /opt/quantlab-backups \
  --confirm-upgrade
```

The explicit switch is mandatory. The coordinator runs a preflight before and after
building, pins every current application image under a release-specific rollback tag,
creates a verified PostgreSQL + `/data` backup while leaving all writers stopped,
starts the new stack, waits for all health checks, and requires the database migration
state to be `current`. Any failure after the backup automatically restores both
archives with the previous API image and force-recreates all five previous application
images. A `rolled_back` or `rollback_failed` result exits non-zero; never treat it as a
successful deployment. JSON evidence is retained under `artifacts/release-upgrades/`.

The build uses locally cached base images by default. Add `--pull` only during a
planned window with verified registry connectivity. The newest three release-specific
rollback image sets are retained by default; change this with
`--rollback-image-retention` only after considering disk capacity.

Exercise the exact build/backup/recreate/health path in a disposable project before a
server upgrade:

```bash
python3 scripts/release_upgrade_drill.py
```

The drill uses random host ports and credentials and removes its containers, volumes,
network, backup, and rollback tags. The accepted 2026-07-13 drill built the complete
stack, created a coordinated backup, reached `ready` with zero blockers at schema
`0027_web_config_templates`, and left no Compose resources. Its report is retained at
`artifacts/release-upgrade-drills/release-upgrade-drill-20260713T093833Z.json`.
Migration 0025 adds portfolio data policies and per-batch lineage evidence, 0026 binds
factor promotion to immutable code/value/Qlib evidence, and 0027 adds two tables for
current and append-only Web configuration revisions. A separate isolated migration
round trip also exercised every downgrade and upgrade step from 0027 to
`0024_pair_paper_ledger` and back to 0027, ending with all 54 control-plane tables.

## Schema upgrades

The API applies Alembic migrations before accepting traffic. For an explicit
maintenance-window migration, first obtain a `ready` release-preflight result and a
coordinated backup, then run:

```powershell
docker compose --env-file deploy\.env -f deploy\compose.yaml run --rm api quant-db upgrade
```

Control-plane migration history lives in `quantlab.alembic_version`. MLflow is
forced to the `public` search path and owns a separate migration history.

## Backup and restore

Back up PostgreSQL and the data volume together so metadata and artifacts describe
the same state. The supported backup command stops every writer and the gateway,
creates a custom-format PostgreSQL dump plus a compressed `/data` archive, verifies
that both archives are readable, records SHA-256 checksums and the Alembic revision,
then restarts only the services that were previously running.

```powershell
.\scripts\backup.ps1 -BackupRoot E:\quantlab-backups -RetentionCount 14
```

The implementation is Python and runs identically on the Linux server:

```bash
python scripts/backup.py --backup-root /opt/quantlab-backups --retention-count 14
```

For the standard `/opt/quantlab` Linux installation, install the checked-in systemd
unit and timer after a successful manual backup and restore drill:

```bash
cd /opt/quantlab
python3.11 -m venv .venv
.venv/bin/pip install -e .
sudo install -m 0644 deploy/systemd/quantlab-backup.service /etc/systemd/system/
sudo install -m 0644 deploy/systemd/quantlab-backup.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quantlab-backup.timer
systemctl list-timers quantlab-backup.timer
```

The timer runs nightly at 03:20 Asia/Shanghai with a randomized delay and catches up
after host downtime. Its `ExecStartPre` runs the same fail-closed release preflight, so
it skips rather than interrupting an initialization or unresolved failed work unit.
Backups retain the newest fourteen sets in `/opt/quantlab-backups`. Inspect failures
with `journalctl -u quantlab-backup.service`; do not weaken the preflight to force a
backup through active work.

Copy completed `quantlab-*` directories to separate storage. A backup is not complete
if it lacks `manifest.json`, `quantlab-postgres.dump`, or
`quantlab-data.tar.gz`. Environment files and credentials are intentionally excluded.
Back up `PLATFORM_SECRET_KEY` separately in a protected secret manager: it must not be
placed beside these archives, but the exact original key is required to decrypt the
runtime-secret rows restored from PostgreSQL. The readiness gate validates every
encrypted row and blocks operation with a critical alert when the key is missing or
wrong. New manifests record only the SHA-256 fingerprint of the validated Fernet key,
never the key itself. This fingerprint is safe to keep with the backup and provides a
pre-destructive restore guard; it does not replace the separately protected recovery
copy of the key.

Restore only during a maintenance window:

```powershell
.\scripts\restore.ps1 `
  -BackupDirectory E:\quantlab-backups\quantlab-YYYYMMDDTHHMMSSZ `
  -ConfirmRestore
```

Linux uses the same validated implementation:

```bash
python scripts/restore.py \
  --backup-directory /opt/quantlab-backups/quantlab-YYYYMMDDTHHMMSSZ \
  --confirm-restore
```

The restore command refuses to run without `-ConfirmRestore`, validates checksums, and
compares the target `PLATFORM_SECRET_KEY` fingerprint before it creates a staging
volume, stops a service, or mutates PostgreSQL. A mismatch leaves the running system
untouched. Backups created before the fingerprint field was introduced remain readable;
for those archives, the restored API health check still validates every ciphertext and
fails closed. After preflight, restore expands the data archive into a staging volume,
stops all writers, restores PostgreSQL and `/data`, reapplies forward migrations, and
restarts services only after success. If restore fails, services remain stopped for
inspection.
The Python procedure was exercised on 2026-07-13 using two disposable Compose projects
at schema revision `0024_pair_paper_ledger`. It verified 52 control-plane tables,
the governed-strategy-allocation sentinel, its JSON evidence, and the acknowledged and
resolved risk-event actor/reason lifecycle,
the separate desired/effective schedule state and its portfolio suspension reason,
the dedicated pair-paper ledger with signed Y/X quantities, NAV, resolved risk event,
and immutable post-trade review,
the point-in-time industry and staged take-profit state, immutable review ledger,
durable system-health history, the locked zero-order broker boundary, and retained
expected/observed broker reconciliation evidence, durable QMT parent/child slices, and
PostgreSQL-backed replay nonces, provider attempts, callback evidence, and a real Fernet
runtime-secret sentinel decrypted with the restored platform key. It also proved that
the backup manifest contains the one-way platform-key fingerprint used by restore
preflight. The restored API, Web, scheduler,
both workers, RD-Agent Docker runtime, and gateway were all started and health-checked
before both isolated projects and every temporary volume were removed. The
machine-readable report is retained under
`artifacts/restore-drills/restore-drill-20260713T070311Z.json`.

The current accepted restore report is
`artifacts/restore-drills/restore-drill-20260713T094458Z.json`. It proves schema
`0027_web_config_templates` with 54 control-plane tables, matching database, data,
strategy-allocation, risk-lifecycle, schedule, pair-paper and Fernet runtime-secret
sentinels, a healthy restored eight-service stack, HTTP 200 through the gateway, and
complete cleanup of both isolated projects.

Repeat the same self-cleaning drill after later migrations or storage-layout changes:

```bash
python scripts/restore_drill.py
```

## Operations

Every service uses Docker's `json-file` rotation with a default limit of five 20 MiB
files. Change `LOG_MAX_SIZE` or `LOG_MAX_FILES` in `deploy/.env` only after estimating
host capacity; application audit and job evidence remains in PostgreSQL and `/data`.

```powershell
docker compose --env-file deploy\.env -f deploy\compose.yaml logs -f api scheduler worker rdagent-worker
docker compose --env-file deploy\.env -f deploy\compose.yaml restart scheduler worker rdagent-worker
docker compose --env-file deploy\.env -f deploy\compose.yaml down
```

Do not use `down -v` on a real installation: it removes PostgreSQL and the shared
data volume. Broker credentials and public-network exposure are deliberately not
enabled in this stack. The implemented portfolio path is paper execution only.
External paging policy, TLS/public-network hardening, and broker execution remain
mandatory gates before internet-facing or live-trading deployment.
