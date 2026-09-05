from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from quant_platform.api import _public_job, _public_rdagent_run
from quant_platform.run_presentation import (
    ACTIVE_RESEARCH_STATUSES,
    RunPresentationStore,
    execution_phase,
    linked_execution_statement,
    public_linked_execution,
    research_page_statements,
    run_presentation,
)

pytestmark = pytest.mark.no_database


@pytest.mark.parametrize(
    ("error", "code"),
    [
        ("Cancelled by operator", "operator_cancelled"),
        ("Worker restarted after the bounded attempt limit", "worker_interrupted"),
        ("cannot allocate memory", "resource_exhausted"),
        ("out of memory", "resource_exhausted"),
        ("idempotency key is already bound to a different job payload", "identity_conflict"),
        (
            "point-in-time standardized style exposure missing rate exceeds 5%",
            "data_validation_failed",
        ),
        (
            "strategy validation window is shorter than its preregistered OOS",
            "data_validation_failed",
        ),
        ("runtime identity mismatch", "runtime_validation_failed"),
        (
            "post-discretization hard constraint violation: capacity_trade_value[SZ000878]",
            "capacity_validation_failed",
        ),
        ("point-in-time styles have no valid rows", "data_validation_failed"),
        ("point-in-time value missing rate exceeds 5%", "data_validation_failed"),
        ("governed signal has no eligible trading dates", "data_validation_failed"),
        ("fin_strategy dataset or research periods differ from the plan", "data_validation_failed"),
        ("Qlib baseline requires dataset provenance metadata", "data_validation_failed"),
        ("rdagent runtime identity changed after enqueue", "runtime_validation_failed"),
        (
            "transparent v23 runner bytes differ from the repair authorization",
            "runtime_validation_failed",
        ),
        ("process exited with code -9", "execution_failed"),
        ("Git diff failed", "execution_failed"),
    ],
)
def test_failure_reason_is_fixed_and_never_echoes_private_error(error, code):
    record = {
        "status": "failed", "kind": "parameter_experiment",
        "error": error + (
            " /private/a.log https://user:password@example.test Authorization: Basic secret"
        ),
    }
    before = copy.deepcopy(record)
    presented = run_presentation(record)
    assert presented["reason_code"] == code
    assert presented["safe_reason"]
    if code == "capacity_validation_failed":
        assert presented["safe_reason"] == "交易容量约束校验未通过，本次执行未完成。"
    assert record == before
    public_text = json.dumps(presented)
    for private in ("password", "example.test", "private", "Authorization", "secret", error):
        assert private not in public_text


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_cancelled_job_and_legacy_failed_research_keep_audit_status(status):
    raw = {"id": "a" * 32, "kind": "fin_strategy", "status": status,
           "error": "Cancelled by operator"}
    public = _public_rdagent_run(raw)
    assert public["status"] == status
    assert public["presentation"]["display_status"] == "interrupted"
    assert public["presentation"]["label"] == "已中止"
    assert public["error"] == "research run failed"
    assert raw["error"] == "Cancelled by operator"


def test_active_retry_does_not_display_previous_attempt_failure():
    presented = run_presentation({"status": "queued", "error": "Worker restarted"})
    assert presented["display_status"] == "queued"
    assert presented["reason_code"] is None


@pytest.mark.parametrize("error", [
    "point-in-time style missing rate is 0%", "runner bytes match the repair authorization",
    "source closure sha256=abc", "dataset provenance metadata verified",
    "runtime identity has not changed", "research periods match the plan",
    "process killed with exit code -9", "eligible trading dates available",
    "capacity trade value is low", "capacity_trade_value budget is available",
    "post-discretization hard constraint violation: max_position",
    "post-discretization hard constraint violation: capacity_trade_value_extra",
])
def test_informational_terms_and_exit_code_do_not_claim_a_specific_failure(error):
    assert run_presentation({"status": "failed", "error": error})[
        "reason_code"
    ] == "execution_failed"


@pytest.mark.parametrize("status", ["queued", "running"])
def test_pending_cancellation_is_stopping_until_terminal(status):
    public = _public_job({"status": status, "cancel_requested_at": "2026-09-06T01:00:00Z"})
    assert public["status"] == status
    assert public["presentation"]["display_status"] == "stopping"
    assert public["presentation"]["reason_code"] == "cancel_requested"


def test_proposal_success_is_not_research_completion_and_parameter_is_actual_execution():
    run = {"id": "a" * 32, "job_id": "b" * 32, "kind": "fin_strategy", "status": "running"}
    proposal = {"id": "b" * 32, "kind": "rdagent_run", "status": "succeeded"}
    waiting = _public_rdagent_run(run, linked_job=proposal)
    assert waiting["status"] == "running"
    assert waiting["presentation"]["display_status"] == "running"
    assert waiting["presentation"]["execution_phase"] == "settlement"
    parameter = {
        "id": "c" * 32, "kind": "parameter_experiment", "status": "running",
        "parameter_experiment_id": "d" * 32, "attempts": 1, "max_attempts": 1,
        "payload": {"strategy_competition_stage": "policy_only"},
        "started_at": "2026-09-06T01:00:00Z",
        "error": "private path /data/secrets", "log_path": "/data/secrets",
    }
    public = _public_rdagent_run(run, linked_job=parameter)
    assert public["job_id"] == "b" * 32  # Audit association remains the proposal.
    assert public["linked_execution"]["job_id"] == "c" * 32
    assert public["linked_execution"]["parameter_experiment_id"] == "d" * 32
    assert public["linked_execution"]["updated_at"] == parameter["started_at"]
    assert public["presentation"]["execution_phase"] == "policy_only"
    assert public["presentation"]["reason_code"] is None
    assert "secrets" not in json.dumps(public)


@pytest.mark.parametrize(
    ("job_status", "expected"),
    [("queued", "queued"), ("running", "running"), ("failed", "failed"),
     ("cancelled", "interrupted")],
)
def test_active_research_shows_linked_execution_during_settlement_lag(job_status, expected):
    public = _public_rdagent_run(
        {"id": "a" * 32, "kind": "fin_strategy", "status": "running"},
        linked_job={"id": "b" * 32, "kind": "parameter_experiment", "status": job_status},
    )
    assert public["status"] == "running"
    assert public["presentation"]["display_status"] == expected


def test_terminal_research_is_not_revived_by_inconsistent_active_linked_job():
    public = _public_rdagent_run(
        {"kind": "fin_strategy", "status": "failed", "error": "failed"},
        linked_job={"id": "b" * 32, "kind": "parameter_experiment", "status": "running"},
    )
    assert public["presentation"]["display_status"] == "failed"
    assert public["linked_execution"]["status"] == "running"


def test_unknown_or_private_phase_is_not_echoed_and_economic_outcome_is_not_inferred():
    for payload in (None, [], {"strategy_competition_stage": ["secret"]},
                    {"strategy_competition_stage": "https://private/token"}):
        public = run_presentation({"kind": "parameter_experiment", "status": "succeeded",
                                   "payload": payload, "progress": {"phase_label": "password"}})
        assert public["execution_phase"] == "parameter_search"
        assert public["display_status"] == "succeeded"
        assert public["reason_code"] is None
        assert "password" not in json.dumps(public)
    assert run_presentation({"status": "unrecognized"})["display_status"] == "unknown"
    assert public_linked_execution(None) is None
    assert execution_phase({"kind": "rdagent_factor_report"}) == "proposal"


def test_linked_public_projection_rejects_invalid_experiment_identifier():
    public = public_linked_execution({"id": "a" * 32, "kind": "parameter_experiment",
                                      "status": "running",
                                      "parameter_experiment_id": "/private?token=secret"})
    assert public["parameter_experiment_id"] is None
    assert "secret" not in json.dumps(public)


def test_generic_model_evaluation_is_not_mislabeled_as_strategy_parameter_search():
    assert execution_phase({"kind": "model_evaluate"}) == "evaluation"
    assert run_presentation({"status": "evaluating"}, research=True)[
        "execution_phase"
    ] == "evaluation"


@pytest.mark.parametrize("invalid_time", [False, True])
def test_public_progress_boundary_removes_extra_diagnostics_and_private_time(invalid_time):
    public = _public_job({
        "kind": "parameter_experiment", "status": "running",
        "_parameter_progress": {
            "state": "available", "completed_count": 1, "succeeded_count": 0,
            "failed_count": 1, "trial_count": 2, "score": 123.456,
            "error": "Authorization: Basic secret /data/private",
            "updated_at": "/data/private?token=secret" if invalid_time else "2026-09-06T01:00:00Z",
        },
    })
    serialized = json.dumps(public)
    for private in ("Authorization", "secret", "private", "123.456", "score"):
        assert private not in serialized
    assert public["presentation"]["progress"]["state"] == (
        "unavailable" if invalid_time else "available"
    )


def _sql(statement):
    return str(
        statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )


def test_linked_query_uses_real_schema_exact_owners_and_one_bounded_row_per_run():
    sql = _sql(linked_execution_statement(["a" * 32, "b" * 32]))
    assert "JOIN LATERAL" in sql
    assert "jobs.id = quantlab.research_runs.job_id" in sql
    assert "->> 'research_run_id'" in sql
    assert "->> 'fin_strategy_research_run_id'" in sql
    assert "CASE WHEN (quantlab.jobs.status IN ('queued', 'running')) THEN 0 ELSE 1 END" in sql
    assert "LIMIT 1" in sql
    assert "jobs.created_at DESC" in sql
    for forbidden in ("metrics_json", "summary_json", "log_path", "artifact_path", "objective"):
        assert forbidden not in sql
    assert "quantlab.jobs.payload_json," not in sql
    assert "quantlab.jobs.progress_json" not in sql


@pytest.mark.parametrize("group", [None, "active", "history"])
def test_research_pagination_counts_same_explicit_scope_and_has_stable_order(group):
    count, page = research_page_statements(limit=20, offset=40, status_group=group)
    count_sql, page_sql = _sql(count), _sql(page)
    assert "count(*)" in count_sql
    assert "LIMIT 20 OFFSET 40" in page_sql
    if group == "history":
        assert (
            "coalesce(quantlab.research_runs.finished_at, quantlab.research_runs.created_at)"
        ) in page_sql
    else:
        assert "created_at DESC, quantlab.research_runs.id DESC" in page_sql
    assert str(count.whereclause) == str(page.whereclause)
    if group:
        for status in ACTIVE_RESEARCH_STATUSES:
            assert repr(status) in count_sql
        assert ("NOT IN" in count_sql) is (group == "history")


def test_linked_store_reads_once_and_does_not_read_db_for_empty_page():
    stamp = datetime(2026, 9, 6, tzinfo=UTC)
    row = {"research_run_id": "a" * 32, "id": "b" * 32, "kind": "parameter_experiment",
           "status": "running", "created_at": stamp, "strategy_competition_stage": "policy_only",
           "fin_strategy_research_run_id": None, "parameter_experiment_id": "c" * 32}
    statements = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, statement):
            statements.append(statement)
            return [SimpleNamespace(_mapping=row)]

    store = RunPresentationStore(SimpleNamespace(connect=lambda: Connection()))
    assert store.linked_executions([]) == {}
    assert statements == []
    result = store.linked_executions([{"id": "a" * 32}, {"id": "d" * 32}])
    assert len(statements) == 3
    assert str(statements[0]) == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    assert "statement_timeout" in str(statements[1])
    assert result["a" * 32]["payload"]["strategy_competition_stage"] == "policy_only"
    assert result["a" * 32]["created_at"] == stamp.isoformat()
