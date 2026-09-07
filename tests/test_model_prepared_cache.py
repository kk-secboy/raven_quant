from __future__ import annotations

import errno
import os
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd
import pytest

from quant_platform import model_prepared_cache as cache
from quant_platform.model_prepared_data import (
    PreparedModelData,
    canonical_key,
    manifest_sha256,
    write_prepared_data,
)

pytestmark = pytest.mark.no_database


def _build(contract, *, count=12):
    def build(path):
        index = pd.MultiIndex.from_product(
            [pd.date_range("2020-01-01", periods=count // 2), ["SH001", "SZ002"]],
            names=["datetime", "instrument"],
        )
        values = np.arange(len(index), dtype=np.float32)
        labels = pd.DataFrame({("label", "LABEL0"): values}, index=index)
        write_prepared_data(
            path, contract=contract,
            data=PreparedModelData(index, {("feature", "F0"): values}, labels, labels.copy()),
        )
    return build


def _cache(root, contract, *, build=None, max_bytes=10**8, min_free_bytes=0, deadline=None):
    return cache.prepared_data_cache(
        root, contract=contract, build=build or _build(contract),
        max_bytes=max_bytes, min_free_bytes=min_free_bytes,
        deadline=deadline,
    )


def test_cold_build_then_verified_hit_preserves_manifest_and_lease_inode(tmp_path):
    contract = {"dataset": "frozen", "window": 1}
    calls = []

    def build(path):
        calls.append(path)
        _build(contract)(path)

    with _cache(tmp_path, contract, build=build) as cold:
        assert cold is not None and not cold["cache_hit"]
        assert cold["entry"] == tmp_path / "entries" / canonical_key(contract)
        assert cold["lease_path"].parent == tmp_path / "locks"
        assert cold["elapsed_seconds"] >= 0
        seal = manifest_sha256(cold["entry"])
        inode = cold["lease_path"].stat().st_ino
    with _cache(tmp_path, contract, build=build) as warm:
        assert warm["cache_hit"] and warm["manifest_sha256"] == seal
        assert warm["lease_path"].stat().st_ino == inode
    assert len(calls) == 1
    assert not list((tmp_path / "building").iterdir())


@pytest.mark.parametrize("kwargs", [{"max_bytes": 0}, {"min_free_bytes": 2**80}])
def test_no_capacity_does_not_build(tmp_path, kwargs):
    def forbidden(_path):
        raise AssertionError("capacity must be checked before preparing data")
    with _cache(tmp_path, {"window": 1}, build=forbidden, **kwargs) as result:
        assert result is None


def test_existing_entry_over_new_budget_falls_back_without_deleting_lease(tmp_path):
    contract = {"window": 1}
    with _cache(tmp_path, contract) as result:
        entry = result["entry"]
    with _cache(tmp_path, contract, max_bytes=0) as result:
        assert result is None
    assert entry.exists()


def test_cache_hit_does_not_wait_for_an_unrelated_preparation_budget_lock(tmp_path):
    contract = {"window": 1}
    with _cache(tmp_path, contract):
        pass
    with cache._locked(tmp_path / "locks" / "budget.lock", exclusive=True):
        with _cache(tmp_path, contract) as result:
            assert result is not None and result["cache_hit"]


@pytest.mark.parametrize("kwargs", [{"max_bytes": -1}, {"min_free_bytes": True}])
def test_bad_capacity_configuration_is_an_error(tmp_path, kwargs):
    with pytest.raises(ValueError, match="nonnegative integer"):
        with _cache(tmp_path, {}, **kwargs):
            pass


def test_oversized_new_entry_is_discarded_without_overwriting_old_entry(tmp_path):
    old, new = {"window": 1}, {"window": 2}
    with _cache(tmp_path, old) as result:
        old_entry = result["entry"]
        seal = result["manifest_sha256"]
    budget = cache._size(old_entry) + 100
    with _cache(tmp_path, new, build=_build(new, count=10000), max_bytes=budget) as result:
        assert result is None
    assert manifest_sha256(old_entry) == seal
    assert not (tmp_path / "entries" / canonical_key(new)).exists()
    assert not list((tmp_path / "building").iterdir())


def test_lru_eviction_happens_only_when_space_is_needed(tmp_path):
    first, second, third = ({"window": number} for number in (1, 2, 3))
    for contract in (first, second):
        with _cache(tmp_path, contract):
            pass
    first_entry = tmp_path / "entries" / canonical_key(first)
    second_entry = tmp_path / "entries" / canonical_key(second)
    size = cache._size(first_entry)
    (tmp_path / "access" / canonical_key(first)).write_text("2")
    (tmp_path / "access" / canonical_key(second)).write_text("1")
    with _cache(tmp_path, third, max_bytes=2 * size) as result:
        assert result is not None
        assert first_entry.exists() and not second_entry.exists()
    assert (tmp_path / "locks" / f"{canonical_key(second)}.lock").exists()


def test_live_lease_prevents_eviction_and_capacity_falls_back(tmp_path):
    first, second = {"window": 1}, {"window": 2}
    with _cache(tmp_path, first) as result:
        size = cache._size(result["entry"])
        with _cache(tmp_path, second, max_bytes=size) as blocked:
            assert blocked is None
        assert result["entry"].exists()
    with _cache(tmp_path, second, max_bytes=size) as admitted:
        assert admitted is not None
        assert not (tmp_path / "entries" / canonical_key(first)).exists()


def test_corrupt_cached_content_raises_without_calling_builder(tmp_path):
    contract = {"window": 1}
    with _cache(tmp_path, contract) as result:
        entry = result["entry"]
    feature = entry / "feature-0000.npy"
    raw = bytearray(feature.read_bytes())
    raw[-1] ^= 1
    feature.write_bytes(raw)

    def forbidden(_path):
        raise AssertionError("corrupt cache is not a miss")
    with pytest.raises(ValueError):
        with _cache(tmp_path, contract, build=forbidden):
            pass
    assert entry.exists()


def test_invalid_builder_output_raises_and_only_own_staging_is_cleaned(tmp_path):
    other = tmp_path / "building" / "another-active-builder"
    other.mkdir(parents=True)
    (other / "keep").write_text("keep")

    def bad(path):
        path.mkdir()
        (path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        with _cache(tmp_path, {}, build=bad):
            pass
    assert (other / "keep").read_text() == "keep"
    assert list((tmp_path / "building").iterdir()) == [other]


def test_builder_running_out_of_disk_falls_back_and_cleans_own_staging(tmp_path):
    def full(path):
        path.mkdir()
        (path / "partial").write_bytes(b"partial")
        raise OSError(errno.ENOSPC, "no space")
    with _cache(tmp_path, {}, build=full) as result:
        assert result is None
    assert not list((tmp_path / "building").iterdir())


def test_protected_staging_counts_against_budget_and_is_never_deleted(tmp_path):
    other = tmp_path / "building" / "another-active-builder"
    other.mkdir(parents=True)
    (other / "keep").write_bytes(b"x" * 1024)

    def forbidden(_path):
        raise AssertionError("protected staging already occupies the byte budget")
    with _cache(tmp_path, {}, build=forbidden, max_bytes=512) as result:
        assert result is None
    assert (other / "keep").stat().st_size == 1024


def test_symlinked_root_is_rejected_without_touching_target(tmp_path):
    destination = tmp_path / "protected"
    destination.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(destination, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit symlink creation")
    with pytest.raises(ValueError, match="link or reparse"):
        with _cache(link, {}):
            pass
    assert list(destination.iterdir()) == []


def test_two_processes_build_the_same_key_only_once(tmp_path):
    source = """
import json,runpy,sys,time
from pathlib import Path
from quant_platform.model_prepared_cache import prepared_data_cache
helpers=runpy.run_path(sys.argv[2])
root=Path(sys.argv[1]); contract={'window':1}
def build(path):
    with (root/'build-calls').open('a') as stream: stream.write('build\\n')
    time.sleep(0.2)
    helpers['_build'](contract)(path)
with prepared_data_cache(root/'cache',contract=contract,build=build,
                         max_bytes=10**8,min_free_bytes=0) as result:
    print(json.dumps({'cache_hit':result['cache_hit']}))
"""
    children = [subprocess.Popen(
        [sys.executable, "-c", source, str(tmp_path), os.path.abspath(__file__)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for _ in range(2)]
    outcomes = []
    try:
        for child in children:
            stdout, stderr = child.communicate(timeout=20)
            assert child.returncode == 0, stderr
            outcomes.append(stdout.strip())
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)
    assert sorted(outcomes) == ['{"cache_hit": false}', '{"cache_hit": true}']
    assert (tmp_path / "build-calls").read_text().splitlines() == ["build"]


@pytest.mark.parametrize("lock_kind", [
    "entry", "budget",
    pytest.param("shared_entry", marks=pytest.mark.skipif(
        os.name == "nt", reason="Windows intentionally uses exclusive reader locks",
    )),
])
def test_cross_process_lock_deadline_raises_and_releases_acquired_locks(tmp_path, lock_kind):
    contract = {"window": 1}
    cache._mkdir(tmp_path / "locks")
    key_lock = tmp_path / "locks" / f"{canonical_key(contract)}.lock"
    held_lock = tmp_path / "locks" / "budget.lock" if lock_kind == "budget" else key_lock
    source = """
import sys
from pathlib import Path
from quant_platform.model_prepared_cache import _locked
with _locked(Path(sys.argv[1]),exclusive=sys.argv[2]=='1'):
    print('leased',flush=True)
    sys.stdin.read()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", source, str(held_lock), "0" if lock_kind == "shared_entry" else "1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "leased"
        start = time.monotonic()
        with pytest.raises(TimeoutError, match="execution deadline"):
            with _cache(tmp_path, contract, deadline=start + 0.15):
                pytest.fail("a held cross-process lock must not be acquired")
        assert time.monotonic() - start < 3
        if lock_kind == "budget":
            with cache._locked(key_lock, exclusive=True, blocking=False) as released:
                assert released is not None
    finally:
        child.communicate(timeout=10)
    with _cache(tmp_path, contract, deadline=time.monotonic() + 10) as result:
        assert result is not None


def test_build_exceeding_deadline_raises_instead_of_capacity_fallback(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: clock[0])
    contract = {"window": 1}

    def slow(path):
        _build(contract)(path)
        clock[0] = 12.0
    with pytest.raises(TimeoutError, match="execution deadline"):
        with _cache(tmp_path, contract, build=slow, deadline=11.0):
            pass
    assert not list((tmp_path / "building").iterdir())
    with _cache(tmp_path, contract) as result:
        assert result is not None


def test_hash_verification_exceeding_deadline_releases_warm_lease(tmp_path, monkeypatch):
    contract = {"window": 1}
    with _cache(tmp_path, contract):
        pass
    clock = [10.0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: clock[0])
    original = cache.load_prepared_data

    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 12.0
        return result
    monkeypatch.setattr(cache, "load_prepared_data", slow)
    with pytest.raises(TimeoutError, match="execution deadline"):
        with _cache(tmp_path, contract, deadline=11.0):
            pass
    with _cache(tmp_path, contract) as result:
        assert result["cache_hit"]


@pytest.mark.skipif(os.name == "nt", reason="Linux shared flock contract")
def test_independent_readonly_sandbox_lease_survives_parent_context(tmp_path):
    first, second = {"window": 1}, {"window": 2}
    with _cache(tmp_path, first) as result:
        size = cache._size(result["entry"])
        child = subprocess.Popen(
            [sys.executable, "-c", (
                "import fcntl,sys; f=open(sys.argv[1],'rb'); "
                "fcntl.flock(f,fcntl.LOCK_SH); print('leased',flush=True); sys.stdin.read()"
            ), str(result["lease_path"])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        assert child.stdout.readline().strip() == "leased"
    try:
        with _cache(tmp_path, second, max_bytes=size) as blocked:
            assert blocked is None
    finally:
        child.communicate(timeout=10)
    with _cache(tmp_path, second, max_bytes=size) as admitted:
        assert admitted is not None


def _orphan_stage(root, contract, *, with_lease=True, name=None):
    stage = root / "building" / (name or f"{canonical_key(contract)}.{uuid.uuid4().hex}")
    stage.mkdir(parents=True)
    (stage / "partial").write_bytes(b"x" * 65536)
    if with_lease:
        with cache._locked(stage / "producer.lease", exclusive=True):
            pass
    return stage


@pytest.mark.parametrize("same_key", [False, True])
def test_idle_orphan_staging_is_reclaimed_before_capacity_fallback(tmp_path, same_key):
    contract = {"window": 1}
    old = contract if same_key else {"window": 0}
    stage = _orphan_stage(tmp_path, old)
    with _cache(tmp_path, contract, max_bytes=10000) as result:
        assert result is not None
        assert not stage.exists()
    assert not list((tmp_path / "building").iterdir())


@pytest.mark.parametrize("invalid", ["missing_lease", "unknown_name"])
def test_unverifiable_staging_is_preserved_and_still_consumes_budget(tmp_path, invalid):
    stage = _orphan_stage(
        tmp_path, {}, with_lease=invalid != "missing_lease",
        name="unrecognized-build" if invalid == "unknown_name" else None,
    )
    with _cache(tmp_path, {"window": 1}, max_bytes=10000) as result:
        assert result is None
    assert stage.exists() and (stage / "partial").stat().st_size == 65536


@pytest.mark.parametrize("held", ["key", "producer"])
def test_staging_gc_requires_both_key_and_producer_lease_to_be_idle(tmp_path, held):
    old, new = {"window": 0}, {"window": 1}
    stage = _orphan_stage(tmp_path, old)
    cache._mkdir(tmp_path / "locks")
    lease = (
        tmp_path / "locks" / f"{canonical_key(old)}.lock"
        if held == "key" else stage / "producer.lease"
    )
    with cache._locked(lease, exclusive=False):
        with _cache(tmp_path, new, max_bytes=10000) as result:
            assert result is None
        assert stage.exists()
    with _cache(tmp_path, new, max_bytes=10000) as result:
        assert result is not None
        assert not stage.exists()


def test_builder_parent_holds_producer_lease_through_publish(tmp_path, monkeypatch):
    contract = {"window": 1}
    stages = []
    original = cache.publish_prepared_data

    def check(path):
        lease = path.parent / "producer.lease"
        assert lease.is_file()
        with cache._locked(lease, exclusive=True, blocking=False) as owner:
            assert owner is None

    def build(path):
        stages.append(path.parent)
        check(path)
        _build(contract)(path)

    def publish(path, *args, **kwargs):
        check(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(cache, "publish_prepared_data", publish)
    with _cache(tmp_path, contract, build=build) as result:
        assert result is not None
        assert not stages[0].exists()


@pytest.mark.skipif(os.name == "nt", reason="Linux producer shared flock and orphan process")
def test_killed_parent_does_not_allow_gc_of_surviving_readonly_preparer(tmp_path):
    producer_source = (
        "import fcntl,sys,time; from pathlib import Path; "
        "f=open(sys.argv[1],'rb'); fcntl.flock(f,fcntl.LOCK_SH); "
        "Path(sys.argv[2]).touch(); "
        "exec('while not Path(sys.argv[3]).exists(): time.sleep(0.02)')"
    )
    parent_source = """
import subprocess,sys,time
from pathlib import Path
from quant_platform.model_prepared_cache import prepared_data_cache
root=Path(sys.argv[1])
def build(path):
    (path.parent/'partial').write_bytes(b'x'*65536)
    child=subprocess.Popen([sys.executable,'-c',sys.argv[2],
        str(path.parent/'producer.lease'),str(root/'ready'),str(root/'stop')],
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    while not (root/'ready').exists(): time.sleep(0.02)
    print('preparer-leased',flush=True)
    sys.stdin.read()
with prepared_data_cache(root/'cache',contract={'window':0},build=build,
                         max_bytes=10**8,min_free_bytes=0): pass
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_source, str(tmp_path), producer_source],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert parent.stdout.readline().strip() == "preparer-leased"
        parent.kill()
        parent.communicate(timeout=10)
        root = tmp_path / "cache"
        stage, = (root / "building").iterdir()
        with _cache(root, {"window": 1}, max_bytes=10000) as blocked:
            assert blocked is None
        assert (stage / "partial").stat().st_size == 65536
        (tmp_path / "stop").touch()
        with cache._locked(
            stage / "producer.lease", exclusive=True, deadline=time.monotonic() + 5,
        ):
            pass
        with _cache(root, {"window": 1}, max_bytes=10000) as admitted:
            assert admitted is not None
        assert not stage.exists()
    finally:
        (tmp_path / "stop").touch()
        if parent.poll() is None:
            parent.kill()
            parent.communicate(timeout=10)
