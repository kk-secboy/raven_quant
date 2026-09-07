from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.no_database

_RESEARCH_SERVICES = (
    "api",
    "scheduler",
    "rdagent-worker",
    "rdagent-model-worker",
    "rdagent-report-worker",
    "rdagent-quant-worker",
)


@pytest.mark.parametrize("service_name", _RESEARCH_SERVICES)
def test_research_services_receive_the_operator_loop_limit(service_name: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((project_root / "deploy" / "compose.yaml").read_text("utf-8"))
    environment = compose["services"][service_name]["environment"]

    # The API/scheduler admission limit and dedicated execution workers must
    # all receive the same operator override, retaining the existing default.
    assert environment["RDAGENT_MAX_LOOPS"] == "${RDAGENT_MAX_LOOPS:-3}"
