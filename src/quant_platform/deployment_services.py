"""Single source of truth for the production Compose service topology."""

from __future__ import annotations

# Long-running services in the default (non-GPU) production profile.  One-shot
# builders and the deployment-only registry are deliberately excluded.
CORE_RUNTIME_SERVICES = frozenset(
    {
        "postgres",
        "api",
        "scheduler",
        "worker",
        "evaluation-worker",
        "paper-worker",
        "rdagent-docker",
        "rdagent-worker",
        "rdagent-model-worker",
        "rdagent-report-worker",
        "rdagent-quant-worker",
        "web",
        "gateway",
    }
)

OPTIONAL_PROFILE_SERVICES: dict[str, frozenset[str]] = {}

# Services whose images are built from this immutable release.  All aliases of
# the shared worker/RD-Agent images are named so Compose recreates every
# container and stamps one release identity rather than leaving stale aliases.
BUILT_APPLICATION_SERVICES = (
    "api",
    "scheduler",
    "worker",
    "evaluation-worker",
    "paper-worker",
    "rdagent-worker",
    "rdagent-model-worker",
    "rdagent-report-worker",
    "rdagent-quant-worker",
    "web",
)

PROFILE_BUILT_SERVICES: dict[str, tuple[str, ...]] = {}

# Stop every component capable of mutating PostgreSQL, governed data, research
# artifacts, or the nested execution daemon before a coordinated backup.  The
# gateway is stopped too so no write request can enter while API is quiesced.
WRITER_SERVICES = (
    "gateway",
    "scheduler",
    "worker",
    "evaluation-worker",
    "paper-worker",
    "rdagent-worker",
    "rdagent-model-worker",
    "rdagent-report-worker",
    "rdagent-quant-worker",
    "rdagent-docker",
    "api",
)

NON_HEALTHCHECK_SERVICES = frozenset({"web", "gateway"})
