from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

import pytest

from quant_platform.runtime_source_closure import (
    POSITION_RISK_SOURCE_ENTRY_PATHS,
    closure_paths,
    local_python_source_closure_inventory,
    local_python_source_closure_sha256,
    position_risk_source_closure_inventory,
    position_risk_source_closure_sha256,
)

pytestmark = pytest.mark.no_database


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8", newline="")


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def test_transitive_local_import_is_automatically_added_to_the_seal(tmp_path: Path) -> None:
    _write(tmp_path / "src/quant_platform/__init__.py", "")
    _write(
        tmp_path / "scripts/entry.py",
        "from quant_platform.alpha import value\nprint(value)\n",
    )
    _write(tmp_path / "src/quant_platform/alpha.py", "value = 1\n")
    _write(tmp_path / "src/quant_platform/beta.py", "value = 2\n")
    before = local_python_source_closure_inventory(
        tmp_path,
        entry_paths=("scripts/entry.py",),
    )
    before_sha256 = local_python_source_closure_sha256(
        tmp_path,
        entry_paths=("scripts/entry.py",),
    )

    _write(
        tmp_path / "src/quant_platform/alpha.py",
        "from quant_platform.beta import value\n",
    )
    after = local_python_source_closure_inventory(
        tmp_path,
        entry_paths=("scripts/entry.py",),
    )
    after_sha256 = local_python_source_closure_sha256(
        tmp_path,
        entry_paths=("scripts/entry.py",),
    )

    assert "src/quant_platform/beta.py" not in closure_paths(before)
    assert "src/quant_platform/beta.py" in closure_paths(after)
    assert before_sha256 != after_sha256


@pytest.mark.parametrize(
    "source",
    [
        "import importlib\nimportlib.import_module('quant_platform.beta')\n",
        "import importlib as loader\nloader.import_module('quant_platform.beta')\n",
        "import importlib\ngetattr(importlib, 'import_module')('quant_platform.beta')\n",
        "import importlib\nload = importlib.import_module\nload('quant_platform.beta')\n",
        "from importlib import import_module as load\nload('quant_platform.beta')\n",
        "import builtins\nbuiltins.__import__('quant_platform.beta')\n",
        "from builtins import __import__ as load\nload('quant_platform.beta')\n",
        "load = __import__\nload('quant_platform.beta')\n",
        "globals()['__import__']('quant_platform.beta')\n",
        "__import__('quant_platform.beta')\n",
        "exec(\"import quant_platform.beta\")\n",
    ],
)
def test_unprovable_dynamic_import_fails_closed(tmp_path: Path, source: str) -> None:
    _write(tmp_path / "scripts/entry.py", source)
    _write(tmp_path / "src/quant_platform/__init__.py", "")
    _write(tmp_path / "src/quant_platform/beta.py", "value = 1\n")

    with pytest.raises(ValueError, match="unsealed dynamic import/eval site"):
        local_python_source_closure_inventory(
            tmp_path,
            entry_paths=("scripts/entry.py",),
        )


def test_source_hash_normalizes_lf_crlf_and_bom(tmp_path: Path) -> None:
    entry = tmp_path / "scripts/entry.py"
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"\xef\xbb\xbfvalue = 1\r\n")
    crlf = local_python_source_closure_sha256(
        tmp_path,
        entry_paths=("scripts/entry.py",),
    )
    entry.write_bytes(b"value = 1\n")
    lf = local_python_source_closure_sha256(
        tmp_path,
        entry_paths=("scripts/entry.py",),
    )

    assert crlf == lf


def test_fragment_local_dependency_requires_an_explicit_supplement(tmp_path: Path) -> None:
    _write(tmp_path / "src/quant_platform/__init__.py", "")
    _write(
        tmp_path / "scripts/entry.py",
        "from quant_platform.api import StrategyConfigRequest\n",
    )
    _write(
        tmp_path / "src/quant_platform/api.py",
        "from quant_platform.dependency import DEFAULT\n\n"
        "class StrategyConfigRequest:\n"
        "    value = DEFAULT\n",
    )
    _write(tmp_path / "src/quant_platform/dependency.py", "DEFAULT = 1\n")

    with pytest.raises(ValueError, match="require explicit closure supplements"):
        local_python_source_closure_inventory(
            tmp_path,
            entry_paths=("scripts/entry.py",),
        )

    manifest = local_python_source_closure_inventory(
        tmp_path,
        entry_paths=("scripts/entry.py",),
        supplement_modules=("quant_platform.dependency",),
    )
    assert {
        "src/quant_platform/api.py#StrategyConfigRequest",
        "src/quant_platform/api.py#StrategyConfigRequest.imports",
        "src/quant_platform/dependency.py",
    } <= set(closure_paths(manifest))


def test_current_v14_closure_covers_execution_research_and_construction() -> None:
    root = Path(__file__).parents[1]
    manifest = position_risk_source_closure_inventory(root)
    paths = set(closure_paths(manifest))

    assert len(paths) >= 90
    assert set(POSITION_RISK_SOURCE_ENTRY_PATHS) <= paths
    assert {
        "src/quant_platform/eligibility.py",
        "src/quant_platform/discrete_constraints.py",
        "src/quant_platform/horizon_review.py",
        "src/quant_platform/strategy_recipes.py",
        "src/quant_platform/strategy_rule_compiler.py",
        "src/quant_platform/transparent_baseline_governance.py",
        "src/quant_platform/transparent_baseline_runner.py",
        "src/quant_data/qlib_builder.py",
        "src/quant_platform/runtime_source_closure.py",
        "src/quant_data/database.py",
    } <= paths
    assert "src/quant_platform/api.py" not in paths
    assert not any(path.startswith(("tests/", "web/")) for path in paths)


def test_current_v14_closure_survives_git_archive_and_autocrlf(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    manifest = position_risk_source_closure_inventory(root)
    source_paths = {
        str(item["path"])
        for item in manifest["inventory"]
        if item["kind"] == "module"
    }
    source_paths.add("src/quant_platform/api.py")
    repo = tmp_path / "source-closure-repo"
    archive_root = tmp_path / "source-closure-archive"
    _write(repo / ".gitattributes", (root / ".gitattributes").read_text(encoding="utf-8"))
    for relative in sorted(source_paths):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())
    _git(repo, "init", "--quiet")
    _git(repo, "config", "core.autocrlf", "true")
    _git(repo, "config", "user.name", "QuantLab Tests")
    _git(repo, "config", "user.email", "tests@quantlab.invalid")
    _git(repo, "add", ".gitattributes", *sorted(source_paths))
    _git(repo, "commit", "--quiet", "-m", "Seal local source closure")
    archive = _git(repo, "archive", "--format=tar", "HEAD")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for relative in sorted(source_paths):
            archived = tar.extractfile(relative)
            assert archived is not None
            destination = archive_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archived.read())

    root_sha256 = position_risk_source_closure_sha256(root)
    assert position_risk_source_closure_sha256(repo) == root_sha256
    assert position_risk_source_closure_sha256(archive_root) == root_sha256
