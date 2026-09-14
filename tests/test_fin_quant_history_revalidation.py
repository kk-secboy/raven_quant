from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_database


def module():
    path = Path(__file__).parents[1] / "scripts/revalidate_fin_quant_result.py"
    spec = importlib.util.spec_from_file_location("fin_quant_history_revalidation", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("changed", [
    None, "values", "consumer", "factor_owner", "code", "pit", "warmup", "domain", "result",
])
def test_preflight_is_bound_to_exact_original_values_and_current_consumer(tmp_path, changed):
    recovery = module()
    values = tmp_path / "values.h5"
    values.write_bytes(b"sealed-factor-values")
    values_sha256 = hashlib.sha256(values.read_bytes()).hexdigest()
    sources = {}
    for name in recovery.CONSUMERS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"consumer-source\n")
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    proof = {"status": "passed", "consumer_sources": sources, "factors": [{
        "candidate_id": "factor-1", "code_sha256": "a" * 64,
        "submitted_comparison": {"exact_match": True, "index_exact_match": True,
                                 "warmup_prefix_rows": 0,
                                 "submitted_sha256": values_sha256},
        "pit_invariance": {"status": "passed", "cutpoint_count": 3},
    }]}
    payload = {"candidates": [{"factors": [{"candidate_id": "factor-1", "code_sha256": "a" * 64,
                                           "submitted_values_path": str(values)}]}]}
    factor = proof["factors"][0]
    if changed == "values":
        values.write_bytes(b"changed-values")
    elif changed == "consumer":
        (tmp_path / recovery.CONSUMERS[0]).write_bytes(b"another-consumer")
    elif changed == "factor_owner":
        factor["candidate_id"] = "other-factor"
    elif changed == "code":
        factor["code_sha256"] = "b" * 64
    elif changed == "pit":
        factor["pit_invariance"]["status"] = "failed"
    elif changed == "warmup":
        factor["submitted_comparison"]["warmup_prefix_rows"] = 1
    elif changed == "domain":
        factor["submitted_comparison"]["index_exact_match"] = False
    elif changed == "result":
        proof["status"] = "failed"
    if changed:
        with pytest.raises(ValueError):
            recovery.verify_preflight(proof, payload, tmp_path)
    else:
        recovery.verify_preflight(proof, payload, tmp_path)
