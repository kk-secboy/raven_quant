from __future__ import annotations

import ast
import hashlib
import json
import sysconfig
from pathlib import Path

import pytest

from quant_platform import release_upgrade as release

pytestmark = pytest.mark.no_database
IMAGE = "rdagent-registry:5000/quantlab/model-sandbox@sha256:" + "d" * 64
IMAGE_ID = "sha256:" + "e" * 64


def source_tree(root: Path) -> Path:
    paths = [
        *release._MODEL_REUSE_ENTRYPOINTS,
        "pyproject.toml", ".dockerignore", "deploy/Dockerfile.worker",
        "deploy/Dockerfile.governed-full-source-overlay", "deploy/model-sandbox/Dockerfile",
        *("src/quant_platform/" + name for name in release._MODEL_REUSE_EXTRA_MODULES),
        "src/quant_platform/model_recompute.py", "src/quant_platform/model_prepared_data.py",
        "src/quant_platform/model_prepared_execution.py", "src/quant_data/__init__.py",
    ]
    for name in paths:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# " + name + "\n", encoding="utf-8")
    (root / "deploy/.env").write_text("MODEL_SANDBOX_IMAGE=" + IMAGE + "\n", encoding="utf-8")
    return root


class ProbeContext:
    def __init__(self, root: Path) -> None:
        self.env_file = root / "deploy/.env"
        self.calls: list[tuple[str, ...]] = []
        self.image_id = IMAGE_ID
        self.incomplete = False

    def run(self, *args: str, **_kwargs) -> str:
        self.calls.append(args)
        if args[3:6] == ("docker", "image", "inspect"):
            return self.image_id
        if self.incomplete:
            return "{}"
        expected = next(
            node.value for node in ast.parse(args[-1]).body
            if isinstance(node, ast.Assign) and node.targets[0].id == "expected"
        )
        return json.dumps(ast.literal_eval(expected))


def test_reuse_verifies_both_runtime_views_and_pins_source_manifest(tmp_path: Path) -> None:
    old, new = source_tree(tmp_path / "old"), source_tree(tmp_path / "new")
    context = ProbeContext(new)
    proof = release._verify_model_sandbox_reuse(context, new, old, IMAGE, IMAGE_ID)
    assert proof["image"] == IMAGE and proof["image_id"] == IMAGE_ID
    assert proof["sources"] == release._model_reuse_source_inventory(new)
    assert proof["live_worker_sources_verified"] is True
    assert proof["sandbox_installed_sources_verified"] is True
    assert any(call[:3] == ("exec", "-T", "worker") for call in context.calls)
    private = next(call for call in context.calls if "--network" in call)
    assert private[private.index("--network") + 1] == "none"
    assert "--read-only" in private and "--cap-drop" in private
    assert "--volume" not in private and "-v" not in private


@pytest.mark.parametrize("name", [
    "scripts/model_sandbox_runner.py", "scripts/prepare_model_data.py",
    "src/quant_platform/model_prepared_data.py", "src/quant_platform/model_recompute.py",
    "src/quant_platform/research_horizon.py", "src/quant_data/__init__.py",
    "pyproject.toml", "deploy/model-sandbox/Dockerfile", "deploy/Dockerfile.worker",
])
def test_changed_numeric_or_build_input_refuses_reuse_before_docker(
    tmp_path: Path, name: str,
) -> None:
    old, new = source_tree(tmp_path / "old"), source_tree(tmp_path / "new")
    (new / name).write_text("changed\n", encoding="utf-8")
    context = ProbeContext(new)
    with pytest.raises(RuntimeError, match="sources changed"):
        release._verify_model_sandbox_reuse(context, new, old, IMAGE, IMAGE_ID)
    assert context.calls == []


def test_model_module_set_change_refuses_reuse(tmp_path: Path) -> None:
    old, new = source_tree(tmp_path / "old"), source_tree(tmp_path / "new")
    (old / "src/quant_platform/model_removed.py").write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sources changed"):
        release._verify_model_sandbox_reuse(ProbeContext(new), new, old, IMAGE, IMAGE_ID)


def test_transitive_runner_dependency_change_refuses_reuse(tmp_path: Path) -> None:
    old, new = source_tree(tmp_path / "old"), source_tree(tmp_path / "new")
    for root in (old, new):
        (root / "scripts/evaluate_model_batch.py").write_text(
            "from quant_platform.feature_set_registry import REGISTRY\n", encoding="utf-8",
        )
        (root / "src/quant_platform/feature_set_registry.py").write_text(
            "from .statistical_validation import SCORE\nREGISTRY = SCORE\n", encoding="utf-8",
        )
        (root / "src/quant_platform/statistical_validation.py").write_text(
            "SCORE = 1\n", encoding="utf-8",
        )
    (new / "src/quant_platform/statistical_validation.py").write_text("SCORE = 2\n")
    context = ProbeContext(new)
    with pytest.raises(RuntimeError, match="sources changed"):
        release._verify_model_sandbox_reuse(context, new, old, IMAGE, IMAGE_ID)
    assert context.calls == []


def test_unsealed_dynamic_runner_import_cannot_preserve_model_image(tmp_path: Path) -> None:
    root = source_tree(tmp_path / "new")
    (root / "scripts/model_sandbox_runner.py").write_text("__import__('arbitrary_runtime')\n")
    with pytest.raises(ValueError, match="unsealed dynamic import"):
        release._model_reuse_source_inventory(root)


def test_missing_private_image_and_incomplete_runtime_evidence_refuse_reuse(tmp_path: Path) -> None:
    old, new = source_tree(tmp_path / "old"), source_tree(tmp_path / "new")
    context = ProbeContext(new)
    context.image_id = "sha256:" + "a" * 64
    with pytest.raises(RuntimeError, match="image ID changed"):
        release._verify_model_sandbox_reuse(context, new, old, IMAGE, IMAGE_ID)
    context.image_id, context.incomplete = IMAGE_ID, True
    with pytest.raises(RuntimeError, match="incomplete evidence"):
        release._verify_model_sandbox_reuse(context, new, old, IMAGE, IMAGE_ID)


@pytest.mark.parametrize("candidate", [True, False])
def test_reuse_cannot_select_an_image_other_than_existing_configuration(
    tmp_path: Path, candidate: bool,
) -> None:
    old, new = source_tree(tmp_path / "old"), source_tree(tmp_path / "new")
    ((new if candidate else old) / "deploy/.env").write_text("MODEL_SANDBOX_IMAGE=other")
    context = ProbeContext(new)
    with pytest.raises(RuntimeError, match="configured model sandbox|existing release"):
        release._verify_model_sandbox_reuse(context, new, old, IMAGE, IMAGE_ID)
    assert context.calls == []


def test_real_installed_byte_probe_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    installed = tmp_path / "quant_platform/model_recompute.py"
    installed.parent.mkdir()
    installed.write_bytes(b"original\n")
    sources = {"src/quant_platform/model_recompute.py": hashlib.sha256(b"original\n").hexdigest()}
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    probe = release._model_reuse_runtime_probe(sources, installed_only=True)
    exec(compile(probe, "reuse-probe", "exec"), {})
    assert json.loads(capsys.readouterr().out) == sources
    installed.write_bytes(b"changed\n")
    with pytest.raises(RuntimeError, match="source differs"):
        exec(compile(probe, "reuse-probe", "exec"), {})


def test_preservation_rechecks_source_after_candidate_build(tmp_path: Path) -> None:
    root = source_tree(tmp_path / "new")
    sources = release._model_reuse_source_inventory(root)
    proof = {"contract_version": "model-sandbox-reuse-v1", "status": "verified",
             "live_worker_sources_verified": True, "sandbox_installed_sources_verified": True,
             "sources": sources, "image": IMAGE, "image_id": IMAGE_ID}
    (root / "scripts/model_sandbox_runner.py").write_text("changed")
    context = ProbeContext(root)
    with pytest.raises(RuntimeError, match="proof is missing or changed"):
        release._prepare_sandbox_images(
            context, root, "20260912T120000Z", wait_timeout=45, preserved_model_sandbox=proof,
        )
    assert context.calls == []


def test_preserved_model_skips_only_model_build_and_keeps_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    old, root = source_tree(tmp_path / "old"), source_tree(tmp_path / "new")

    class SealingContext(ProbeContext):
        def run(self, *args: str, **kwargs) -> str:
            if args == ("config", "--format", "json"):
                return json.dumps({"services": {
                    "worker": {"image": "worker:new"}, "rdagent-worker": {"image": "rd:new"},
                }})
            if any(str(arg).startswith("find ") for arg in args):
                return "/data/qlib/daily/calendars/day.txt\n"
            if args[-2:-1] == ("-c",) or args[3:6] == ("docker", "image", "inspect"):
                return super().run(*args, **kwargs)
            self.calls.append(args)
            return ""

        def docker(self, *args: str, **_kwargs) -> str:
            self.calls.append(("docker", *args))
            if args[:4] == ("image", "inspect", "--format", "{{.Id}}"):
                return "sha256:" + "b" * 64
            if args[:4] == ("image", "inspect", "--format", "{{json .RepoDigests}}"):
                return json.dumps([args[4].rsplit(":", 1)[0] + "@sha256:" + "c" * 64])
            return ""

    context = SealingContext(root)
    proof = release._verify_model_sandbox_reuse(context, root, old, IMAGE, IMAGE_ID)
    builds, smokes = [], []
    monkeypatch.setattr(release, "_registry_port", lambda _context: 55000)
    monkeypatch.setattr(release, "_wait_for_registry", lambda *_args: None)
    monkeypatch.setattr(release, "_governed_sandbox_evidence", lambda _root: {})

    def build(_context, **kwargs):
        builds.append(kwargs["context_root"].name)
        return kwargs["dind_repository"] + "@sha256:" + "f" * 64

    monkeypatch.setattr(release, "_build_and_publish_host_image", build)
    monkeypatch.setattr(
        release, "_dind_smoke", lambda _context, image, *_args, **_kw: smokes.append(image),
    )
    sealed = release._prepare_sandbox_images(
        context, root, "20260912T120000Z", wait_timeout=45, preserved_model_sandbox=proof,
    )
    assert builds == ["qlib-sandbox"]
    assert IMAGE in smokes and any("qlib-sandbox@" in image for image in smokes)
    assert sealed["MODEL_SANDBOX_IMAGE"] == IMAGE
    assert sealed["model_sandbox_reuse"] == proof
    assert "MODEL_SANDBOX_IMAGE=" + IMAGE in context.env_file.read_text()
