"""Process-local environment isolation for destructive deployment drills."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

# Docker Compose gives inherited shell variables precedence over ``--env-file``.
# A production operator may legitimately export these values before invoking an
# operational command, so a drill must remove them before it renders or starts
# any disposable Compose project.  Build transport variables (Docker context,
# proxy and package mirrors) are intentionally not changed.
DRILL_COMPOSE_ENVIRONMENT_KEYS = frozenset(
    {
        "ALERT_WEBHOOK_URL",
        "AUTH_COOKIE_SECURE",
        "AUTH_MODE",
        "AUTH_SESSION_HOURS",
        "BROKER_MODE",
        "CHAT_MODEL",
        "COMPOSE_PROFILES",
        "DATA_FRESHNESS_MAX_DAYS",
        "DOWNLOAD_WORKERS",
        "HEALTH_SNAPSHOT_SECONDS",
        "HTTP_BIND_ADDRESS",
        "HTTP_PORT",
        "LOG_MAX_FILES",
        "LOG_MAX_SIZE",
        "MODEL_SANDBOX_IMAGE",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "PLATFORM_SECRET_KEY",
        "POSTGRES_BIND_ADDRESS",
        "POSTGRES_PASSWORD",
        "POSTGRES_PORT",
        "QUANTLAB_CANONICAL_BASELINE",
        "QUANTLAB_CONFIG_DIGEST",
        "QUANTLAB_DATA_HOST_PATH",
        "QUANTLAB_RELEASE_ALIAS_OF",
        "QUANTLAB_RELEASE_ID",
        "QUANTLAB_RELEASE_KIND",
        "RDAGENT_DATA_SCIENCE_IMAGE",
        "RDAGENT_DOCKER_HOST_PATH",
        "RDAGENT_ENABLED",
        "RDAGENT_FINETUNE_BENCHMARK_IMAGE",
        "RDAGENT_FINETUNE_GPU_PROBE_IMAGE",
        "RDAGENT_FINETUNE_IMAGE",
        "RDAGENT_FINETUNE_MIN_DISK_GB",
        "RDAGENT_FINETUNE_MIN_GPU_MEMORY_MB",
        "RDAGENT_LLM_KEY_ENV",
        "RDAGENT_QLIB_SANDBOX_IMAGE",
        "RDAGENT_REGISTRY_HOST_PATH",
        "RDAGENT_REGISTRY_PORT",
        "RDAGENT_RUNTIME_IMAGE_DIGEST",
        "RESEARCH_ASSET_AUTO_ENABLED",
        "RESEARCH_ASSET_AUTO_HOUR",
        "RESEARCH_ASSET_AUTO_MINUTE",
        "REQUESTS_PER_MINUTE",
        "SCHEDULER_POLL_SECONDS",
        "SCHEDULER_MAX_TICK_SECONDS",
        "STALE_JOB_HOURS",
        "TUSHARE_API_URL",
        "TUSHARE_TOKEN",
    }
)


def isolated_drill_environment(function: Callable[P, R]) -> Callable[P, R]:
    """Run a standalone drill without production Compose interpolation values.

    Drill entry points also hold the process-reentrant control-plane lock, so
    this temporary process-level change cannot overlap another supported
    control-plane operation in the same interpreter.  Every prior value is
    restored even when the drill fails.
    """

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        inherited = {
            name: os.environ[name]
            for name in DRILL_COMPOSE_ENVIRONMENT_KEYS
            if name in os.environ
        }
        try:
            for name in DRILL_COMPOSE_ENVIRONMENT_KEYS:
                os.environ.pop(name, None)
            return function(*args, **kwargs)
        finally:
            for name in DRILL_COMPOSE_ENVIRONMENT_KEYS:
                os.environ.pop(name, None)
            os.environ.update(inherited)

    return wrapped
