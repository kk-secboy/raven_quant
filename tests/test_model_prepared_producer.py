"""Execute the real producer against synthetic data inside an explicit test sandbox."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("qlib")

from quant_platform.model_data_handler import build_model_handler_from_prepared_data
from quant_platform.model_data_request import file_digest, prepared_data_request
from quant_platform.model_prepared_data import canonical_key, load_prepared_data, manifest_sha256

pytestmark = [
    pytest.mark.no_database,
    pytest.mark.skipif(
        os.environ.get("QUANTLAB_PREPARED_PRODUCER_SANDBOX") != "1",
        reason="requires isolated /work, /qlib and /output tmpfs mounts",
    ),
]


def _assert_progress_audit(path, terminal):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["stage"] == "producer_started"
    assert records[-1]["status"] == terminal
    assert records[-1]["stage"] == f"producer_{terminal}"
    assert all(record["status"] == "running" for record in records[:-1])
    elapsed = [record["elapsed_seconds"] for record in records]
    assert elapsed == sorted(elapsed) and elapsed[0] >= 0
    for record in records:
        assert record["contract_version"] == "model-prepared-data-progress-v1"
        memory = record["memory"]
        assert memory["process_peak_rss_bytes"] >= memory["process_rss_bytes"] > 0
        assert memory["observed_memory_peak_bytes"] >= memory["process_peak_rss_bytes"]
        assert memory["cgroup_limit_bytes"] > 0
    return records


@pytest.fixture(scope="module")
def producer_fixture():
    import quant_platform.model_data_handler as handler_module

    workspace, provider = Path("/work"), Path("/qlib")
    for root in (workspace, provider, Path("/output")):
        assert root.is_dir() and not any(root.iterdir()), "test tmpfs must be empty"
    package = workspace / "quant_platform"
    package.mkdir()
    lease = workspace / "producer.lease"
    lease.write_bytes(b"\0")
    lease.chmod(0o444)
    (package / "__init__.py").write_text("", encoding="utf-8")
    identity = {}
    for name in ("model_data_handler.py", "model_data_request.py", "model_prepared_data.py",
                 "upstream_versions.py"):
        source = Path(handler_module.__file__).with_name(name)
        destination = package / name
        shutil.copy2(source, destination)
        identity[name] = file_digest(destination)
    source = Path(__file__).resolve().parents[1] / "scripts" / "prepare_model_data.py"
    script = workspace / "prepare_model_data.py"
    shutil.copy2(source, script)
    identity[script.name] = file_digest(script)

    calendar = pd.bdate_range("2020-01-02", periods=100)
    (provider / "calendars").mkdir()
    (provider / "instruments").mkdir()
    (provider / "calendars" / "day.txt").write_text(
        "\n".join(calendar.strftime("%Y-%m-%d")), encoding="utf-8",
    )
    symbols = [f"SH{600000 + index}" for index in range(8)]
    (provider / "instruments" / "cn_all.txt").write_text(
        "\n".join(f"{symbol}\t{calendar[30 if i == 7 else 0]:%Y-%m-%d}"
                  f"\t{calendar[-1]:%Y-%m-%d}" for i, symbol in enumerate(symbols)),
        encoding="utf-8",
    )
    time_axis = np.arange(len(calendar))
    for number, symbol in enumerate(symbols):
        directory = provider / "features" / symbol.lower()
        directory.mkdir(parents=True)
        close = (10 + number + time_axis * 0.03 + np.sin(time_axis / (number + 2))).astype("f4")
        if number == 2:
            close[44] = np.nan
        fields = {"close": close, "open": close * np.float32(0.999),
                  "volume": (1000 + number * 70 + time_axis * 11).astype("f4")}
        for field, values in fields.items():
            np.concatenate([np.array([0], dtype="f4"), values]).tofile(
                directory / f"{field}.day.bin",
            )
    digest = hashlib.sha256()
    for path in sorted(provider.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(provider)).encode())
            digest.update(path.read_bytes())
    start, train_end, load_end = [calendar[i].strftime("%Y-%m-%d") for i in (20, 70, 90)]
    features = {"momentum": "$close/Ref($close,5)-1", "close": "$close",
                "activity": "$volume/Mean($volume,5)", "gap": "$open/$close-1"}
    label_contract = {"label_expression": "Ref($close,-2)/Ref($close,-1)-1",
                      "purge_sessions": 2, "embargo_sessions": 2}
    manifest = {
        "dataset_identity_sha256": digest.hexdigest(), "feature_set": {"features": features},
        "periods": {
            "train_start": start, "train_end": train_end,
            "valid_start": calendar[75].strftime("%Y-%m-%d"), "valid_end": load_end,
            "test_start": calendar[95].strftime("%Y-%m-%d"),
            "test_end": calendar[99].strftime("%Y-%m-%d"),
        },
        "prediction_segment": "valid", "universe": "cn_all", "final_oos_opened": False,
    }
    request = prepared_data_request(
        manifest, provider=provider, label_contract=label_contract, producer_identity=identity,
    )
    request_path = workspace / "request.json"
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    return {"script": script, "request": request, "request_path": request_path,
            "features": features, "label_contract": label_contract, "start": start,
            "train_end": train_end, "load_end": load_end, "symbols": symbols, "lease": lease}


def test_real_preparation_script_serializes_the_same_synthetic_qlib_data(producer_fixture):
    import qlib
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import QlibDataLoader

    fixture = producer_fixture
    audit = Path("/output/success-progress.jsonl")
    completed = subprocess.run(
        [sys.executable, "-B", str(fixture["script"]), "--lease", str(fixture["lease"]),
         "--audit", str(audit)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    events = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert events[-1]["status"] == "prepared"
    assert events[-1]["contract_version"] == "model-prepared-data-summary-v1"
    assert events[-1]["request_sha256"] == canonical_key(fixture["request"])
    assert events[-1]["elapsed_seconds"] >= events[-1]["prepare_seconds"] >= 0
    assert events[-1]["write_seconds"] >= 0
    memory = events[-1]["memory"]
    assert memory["process_peak_rss_bytes"] >= memory["process_rss_bytes"] > 0
    assert memory["cgroup_limit_bytes"] > 0
    assert memory["observed_memory_peak_bytes"] >= memory["process_peak_rss_bytes"]
    assert events[-1]["manifest_sha256"] == manifest_sha256(Path("/output/entry"))
    progress = _assert_progress_audit(audit, "completed")
    stages = [record["stage"] for record in progress]
    loaded = next(i for i, name in enumerate(stages) if name.startswith("handler_loaded_features_"))
    normalized = next(i for i, name in enumerate(stages)
                      if name.startswith("handler_normalized_features_"))
    assert stages.index("qlib_initialized") < loaded < normalized
    assert normalized < stages.index("handler_labels_ready") < stages.index("prepared_data_writing")
    assert stages.index("prepared_data_writing") < len(stages) - 1
    data = load_prepared_data(
        Path("/output/entry"), expected_contract=fixture["request"],
        expected_manifest_sha256=events[-1]["manifest_sha256"],
    )
    assert len(data.index) == 558  # 7 * 71 sessions plus the late-listed stock's 61.
    assert list(data.features) == [("feature", name) for name in fixture["features"]]
    assert set(data.index.get_level_values("instrument")) == set(fixture["symbols"])
    assert len(data.learn_labels) < len(data.infer_labels)
    assert all(not values.flags.writeable for values in data.features.values())

    qlib.init(provider_uri="/qlib", region="cn", kernels=1,
              expression_cache=None, dataset_cache=None,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": "/tmp/producer-reference-tracking",
                                      "default_exp_name": "reference"}})
    reference = DataHandlerLP(
        data_loader=QlibDataLoader(config={
            "feature": [list(fixture["features"].values()), list(fixture["features"])],
            "label": [[fixture["label_contract"]["label_expression"]], ["LABEL0"]],
        }),
        instruments="cn_all", start_time=fixture["start"], end_time=fixture["load_end"],
        drop_raw=True, process_type=DataHandlerLP.PTYPE_A,
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {
                "fields_group": "feature", "clip_outlier": True,
                "fit_start_time": fixture["start"], "fit_end_time": fixture["train_end"],
            }},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[{"class": "DropnaLabel"},
                          {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}],
    )
    cached = build_model_handler_from_prepared_data(data)
    for data_key in (DataHandlerLP.DK_I, DataHandlerLP.DK_L):
        expected = reference.fetch(data_key=data_key, col_set=DataHandlerLP.CS_RAW)
        actual = cached.fetch(data_key=data_key, col_set=DataHandlerLP.CS_RAW)
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        for position in range(len(expected.columns)):
            assert expected.iloc[:, position].to_numpy().tobytes() == (
                actual.iloc[:, position].to_numpy().tobytes()
            )


def test_real_preparation_script_rejects_wrong_source_identity(producer_fixture):
    request = json.loads(producer_fixture["request_path"].read_text(encoding="utf-8"))
    request["producer_identity"]["model_data_handler.py"] = "0" * 64
    request_path, output = Path("/work/wrong-identity.json"), Path("/output/wrong-identity")
    audit = Path("/output/wrong-identity-progress.jsonl")
    request_path.write_text(json.dumps(request), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-B", str(producer_fixture["script"]),
         "--request", str(request_path), "--output", str(output),
         "--lease", str(producer_fixture["lease"]), "--audit", str(audit)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode != 0
    assert "prepared data producer source changed" in completed.stderr
    assert not output.exists()
    records = _assert_progress_audit(audit, "failed")
    assert [record["stage"] for record in records] == ["producer_started", "producer_failed"]
    assert '"status": "prepared"' not in completed.stdout


def test_real_preparation_script_requires_a_live_readable_producer_lease(producer_fixture):
    output = Path("/output/missing-lease")
    audit = Path("/output/missing-lease-progress.jsonl")
    missing = Path("/work/absent-producer.lease")
    completed = subprocess.run(
        [sys.executable, "-B", str(producer_fixture["script"]),
         "--output", str(output), "--lease", str(missing), "--audit", str(audit)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode != 0
    assert str(missing) in completed.stderr
    assert not output.exists()
    records = _assert_progress_audit(audit, "failed")
    assert [record["stage"] for record in records] == ["producer_started", "producer_failed"]
    assert '"status": "prepared"' not in completed.stdout


def test_producer_progress_is_visible_before_execution_finishes(producer_fixture):
    import fcntl

    fixture = producer_fixture
    audit, output = Path("/output/live-progress.jsonl"), Path("/output/live-observed-entry")
    with fixture["lease"].open("rb") as blocker:
        fcntl.flock(blocker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen(
            [sys.executable, "-B", str(fixture["script"]), "--lease", str(fixture["lease"]),
             "--audit", str(audit), "--output", str(output)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if audit.exists() and audit.read_bytes().endswith(b"\n"):
                    break
                assert child.poll() is None, "producer exited before publishing its started audit"
                time.sleep(0.01)
            records = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            assert records[0]["stage"] == "producer_started"
            assert records[0]["status"] == "running"
            assert records[0]["memory"]["process_rss_bytes"] > 0
            assert child.poll() is None
            assert not output.exists()
            fcntl.flock(blocker, fcntl.LOCK_UN)
            stdout, stderr = child.communicate(timeout=60)
            assert child.returncode == 0, stdout + stderr
            _assert_progress_audit(audit, "completed")
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)


def test_producer_never_overwrites_a_previous_attempt_audit(producer_fixture):
    audit, output = Path("/output/existing-progress.jsonl"), Path("/output/existing-audit-entry")
    previous = b'{"stage":"previous-attempt","status":"failed"}\n'
    audit.write_bytes(previous)
    completed = subprocess.run(
        [sys.executable, "-B", str(producer_fixture["script"]),
         "--lease", str(producer_fixture["lease"]), "--audit", str(audit),
         "--output", str(output)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode != 0
    assert audit.read_bytes() == previous
    assert not output.exists()
