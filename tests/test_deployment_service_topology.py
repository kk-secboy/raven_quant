from __future__ import annotations

import re
from pathlib import Path

import pytest

from quant_platform.backup_restore import WRITER_SERVICES
from quant_platform.deployment_services import (
    BUILT_APPLICATION_SERVICES,
    CORE_RUNTIME_SERVICES,
    OPTIONAL_PROFILE_SERVICES,
)
from quant_platform.release_identity import RELEASE_IDENTITY_ENV_TO_LABEL
from quant_platform.release_preflight import EXPECTED_SERVICES
from quant_platform.release_upgrade import BUILT_SERVICES

pytestmark = pytest.mark.no_database


def _compose_services() -> set[str]:
    root = Path(__file__).resolve().parents[1]
    text = (root / "deploy" / "compose.yaml").read_text(encoding="utf-8")
    body = text.split("services:\n", 1)[1]
    return set(re.findall(r"^  ([a-z0-9-]+):\s*$", body, flags=re.MULTILINE))


def test_release_topology_covers_every_long_running_default_service() -> None:
    compose = _compose_services()

    assert EXPECTED_SERVICES == set(CORE_RUNTIME_SERVICES)
    assert set(CORE_RUNTIME_SERVICES) <= compose
    assert {
        "evaluation-worker",
        "paper-worker",
        "rdagent-model-worker",
        "rdagent-report-worker",
        "rdagent-quant-worker",
    } <= EXPECTED_SERVICES


def test_build_and_backup_topologies_cover_all_writer_aliases() -> None:
    assert BUILT_SERVICES == BUILT_APPLICATION_SERVICES
    assert {
        "evaluation-worker",
        "paper-worker",
        "rdagent-model-worker",
        "rdagent-report-worker",
        "rdagent-quant-worker",
    } <= set(BUILT_SERVICES)
    assert {
        "evaluation-worker",
        "paper-worker",
        "rdagent-model-worker",
        "rdagent-report-worker",
        "rdagent-quant-worker",
    } <= set(WRITER_SERVICES)


def test_every_built_service_has_an_explicit_stable_image_alias() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "deploy" / "compose.yaml").read_text(encoding="utf-8")

    aliases: dict[str, str] = {}
    for service in BUILT_SERVICES:
        match = re.search(
            rf"^  {re.escape(service)}:\s*$\n(?P<body>.*?)(?=^  [a-z0-9-]+:\s*$|\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match is not None, service
        image = re.search(r"^    image:\s+(?P<image>\S+)\s*$", match.group("body"), re.MULTILINE)
        assert image is not None, service
        aliases[service] = image.group("image")

    assert all("${" not in image for image in aliases.values())
    assert aliases["api"] == aliases["scheduler"]
    assert aliases["worker"] == aliases["evaluation-worker"] == aliases["paper-worker"]
    assert (
        aliases["rdagent-worker"]
        == aliases["rdagent-model-worker"]
        == aliases["rdagent-report-worker"]
        == aliases["rdagent-quant-worker"]
    )


def test_compose_stamps_every_stateless_runtime_with_release_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "deploy" / "compose.yaml").read_text(encoding="utf-8")

    assert "x-release-environment: &release-environment" in text
    assert text.count("<<: *release-environment") == 10
    assert text.count("environment: *release-environment") == 2
    assert "x-release-labels: &release-labels" in text
    assert text.count("labels: *release-labels") == 12
    for variable, label in RELEASE_IDENTITY_ENV_TO_LABEL.items():
        assert f"  {variable}: ${{{variable}:-}}" in text
        assert f"  {label}: ${{{variable}:-}}" in text

    stateless = set(CORE_RUNTIME_SERVICES).union(
        *OPTIONAL_PROFILE_SERVICES.values()
    ) - {"postgres"}
    for service in stateless:
        match = re.search(
            rf"^  {re.escape(service)}:\s*$\n(?P<body>.*?)(?=^  [a-z0-9-]+:\s*$|\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match is not None, service
        assert "labels: *release-labels" in match.group("body"), service


def test_every_rdagent_runtime_build_inherits_package_mirrors() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "deploy" / "compose.yaml").read_text(encoding="utf-8")

    runtime_builds = re.findall(
        r"^  (?P<service>[a-z0-9-]+):\s*$\n(?P<body>.*?deploy/Dockerfile\.rdagent.*?)"
        r"(?=^  [a-z0-9-]+:\s*$|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert runtime_builds
    for service, body in runtime_builds:
        assert "PIP_INDEX_URL: ${PIP_INDEX_URL:-https://pypi.org/simple}" in body, service
        assert "DEBIAN_MIRROR: ${DEBIAN_MIRROR:-deb.debian.org/debian}" in body, service


def test_capital_research_workers_can_verify_the_independent_evaluation_lane() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "deploy" / "compose.yaml").read_text(encoding="utf-8")

    for service in (
        "rdagent-worker",
        "rdagent-model-worker",
        "rdagent-report-worker",
        "rdagent-quant-worker",
    ):
        match = re.search(
            rf"^  {re.escape(service)}:\s*$\n(?P<body>.*?)(?=^  [a-z0-9-]+:\s*$|\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match is not None, service
        body = match.group("body")
        assert (
            "RDAGENT_EVALUATION_WORKER_URL: http://evaluation-worker:8770" in body
        ), service
        assert "MODEL_SANDBOX_IMAGE: ${MODEL_SANDBOX_IMAGE:-}" in body, service
