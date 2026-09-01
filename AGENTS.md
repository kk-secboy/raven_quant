# Repository Guidelines

## Project Structure & Module Organization

Python uses a `src/` layout. `src/quant_data/` owns Tushare/BaoStock ingestion, immutable snapshots, lineage, and Qlib conversion; `src/quant_platform/` contains APIs, workers, RD-Agent orchestration, governance, backtests, allocation, and simulation. Put Python tests in `tests/`. The React/Vinext frontend lives in `web/app/`, with browser-independent tests in `web/tests/` and static files in `web/public/`. Operational code belongs in `scripts/`, database revisions in `migrations/`, deployment assets in `deploy/`, and supporting design material in `docs/`. Within `src/quant_platform/`, `worker.py` keeps job claiming, process monitoring, settlement and scheduling; per-kind subprocess command assembly lives in `src/quant_platform/job_commands/` split by domain (`data`, `evaluation`, `simulation`, `research`, `ops`) — add command builders for new job kinds to the matching domain module there, never back into `worker.py`.

## Build, Test, and Development Commands

Use PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\.venv\Scripts\quant-db.exe upgrade
.\.venv\Scripts\quant-web.exe --reload
corepack enable
pnpm --dir web install --frozen-lockfile
pnpm --dir web run dev
```

Run validation before review:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
pnpm --dir web run lint
pnpm --dir web run test
```

The web test command builds the application before running Node tests.

## Coding Style & Naming Conventions

Target Python 3.11, use four-space indentation and 100-character lines, and keep imports Ruff-sorted. Ruff checks `E`, `F`, `I`, `UP`, and `B`. Use `snake_case` for modules/functions, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. TypeScript uses strict checking, two-space indentation, `camelCase` values, and `PascalCase` components/types. Follow nearby formatting; no separate autoformatter is configured.

## Testing Guidelines

Name Python files `test_*.py` and test functions `test_*`. Mark pure tests with `pytest.mark.no_database`; unmarked tests migrate and reset the configured disposable PostgreSQL database (`TEST_DATABASE_URL`). Add focused regression coverage for behavior changes. No numeric coverage threshold is configured. On Windows temp-permission failures, retry with a repository-local `--basetemp .codex_tmp\pytest-<name>`.

## Commit & Pull Request Guidelines

Use a focused, action-oriented subject of at most 72 characters, matching history such as `Add governed information pipeline`; recent work also uses `feat:` and `fix:` prefixes. PRs should explain the problem and approach, list exact checks run, link relevant issues, and flag migrations, configuration changes, or data backfills. Include screenshots for `web/` changes and keep unrelated work in separate commits.

## Security & Product Boundaries

Never commit `.env`, `*.pem`, tokens, generated `data/`, `artifacts/`, logs, or test-temp directories. Enter Tushare credentials through `scripts/configure_tushare.py`. This repository is simulation-only: do not introduce broker order routing. Product-semantic changes must update the authoritative root Markdown and its governance tests.
