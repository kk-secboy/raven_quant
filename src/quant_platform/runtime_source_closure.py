"""Deterministic local-source closure for governed transparent quant runtimes.

The closure is resolved from Python syntax without importing project modules.
Only ``quant_data`` and ``quant_platform`` are local namespaces.  External
libraries are image/runtime dependencies and remain outside this source seal.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCE_CLOSURE_CONTRACT_VERSION = "quantlab-local-python-source-closure-v1"
LOCAL_SOURCE_NAMESPACES = ("quant_data", "quant_platform")
POSITION_RISK_SOURCE_ENTRY_PATHS = (
    "scripts/evaluate_quant_bundle.py",
    "scripts/run_multifactor_backtest.py",
    "scripts/run_recommendation_refresh.py",
)
# ``StrategyConfigRequest`` is deliberately still owned by api.py.  Sealing
# the whole API would pull authentication, web projections and unrelated
# control-plane stores into an economic-runtime identity.  The fragment plus
# these explicit semantic dependencies closes that narrow construction edge.
POSITION_RISK_SOURCE_SUPPLEMENT_MODULES = (
    "quant_data.execution_contract",
    "quant_platform.cost_model",
    "quant_platform.research_horizon",
    "quant_platform.runtime_source_closure",
    "quant_platform.strategy_recipes",
    "quant_platform.strategy_rule_compiler",
)
LOCAL_IMPORT_FRAGMENT_BOUNDARIES: Mapping[str, tuple[str, ...]] = {
    "quant_platform.api": ("StrategyConfigRequest",),
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUNNER_SEAL_ASSIGNMENT = re.compile(
    r"(?ms)^((?:POSITION_RISK|FAIL_CLOSED_EXECUTION|FILL_AWARE_HOLDING_AGE|"
    r"SINGLE_MEMBER_PRE_RESULT_REPAIR|DISCRETE_MAX_POSITION_REPAIR|"
    r"TOPK_INDUSTRY_CAPACITY_REPAIR|FORWARD_ONLY_REHABILITATION|"
    r"STRATEGY_RESEARCH_V20|STRATEGY_RESEARCH_V21|STRATEGY_RESEARCH_V22|"
    r"STRATEGY_RESEARCH_V23|STRATEGY_RESEARCH_V24|STRATEGY_RESEARCH_V25|"
    r"STRATEGY_RESEARCH_V26|STRATEGY_RESEARCH_V27|STRATEGY_RESEARCH_V28|"
    r"STRATEGY_RESEARCH_V29|STRATEGY_RESEARCH_V30|STRATEGY_RESEARCH_V31|"
    r"STRATEGY_RESEARCH_V32|STRATEGY_RESEARCH_V33|STRATEGY_RESEARCH_V34|"
    r"STRATEGY_RESEARCH_V35|STRATEGY_RESEARCH_V36|"
    r"STRATEGY_RESEARCH)_TARGET_"
    r"(?:RUNNER|RUNTIME_BUNDLE)_SHA256\s*=\s*\(\s*)"
    r'"[0-9a-f]{64}"(\s*\))'
)
_DATABASE_RUNTIME_IDENTITY_CONSTRAINTS = (
    "ck_strategy_versions_v12_runtime_identity",
    "ck_strategy_versions_v13_runtime_identity",
    "ck_strategy_versions_v14_runtime_identity",
    "ck_strategy_versions_v15_runtime_identity",
    "ck_strategy_versions_v16_runtime_identity",
    "ck_strategy_versions_v17_runtime_identity",
    "ck_strategy_versions_v18_runtime_identity",
    "ck_strategy_versions_v19_runtime_identity",
    "ck_strategy_versions_v20_runtime_identity",
    "ck_strategy_versions_v21_runtime_identity",
    "ck_strategy_versions_v22_runtime_identity",
    "ck_strategy_versions_v23_runtime_identity",
    "ck_strategy_versions_v24_runtime_identity",
    "ck_strategy_versions_v25_runtime_identity",
    "ck_strategy_versions_v26_runtime_identity",
    "ck_strategy_versions_v27_runtime_identity",
    "ck_strategy_versions_v28_runtime_identity",
    "ck_strategy_versions_v29_runtime_identity",
    "ck_strategy_versions_v30_runtime_identity",
    "ck_strategy_versions_v31_runtime_identity",
    "ck_strategy_versions_v32_runtime_identity",
    "ck_strategy_versions_v33_runtime_identity",
    "ck_strategy_versions_v34_runtime_identity",
    "ck_strategy_versions_v35_runtime_identity",
    "ck_strategy_versions_v36_runtime_identity",
    "ck_strategy_versions_v37_runtime_identity",
)
_DYNAMIC_IMPORT_CALLS = frozenset(
    {
        "__import__",
        "builtins.__import__",
        "import_module",
        "importlib.import_module",
        "run_module",
        "run_path",
        "runpy.run_module",
        "runpy.run_path",
        "spec_from_file_location",
        "importlib.util.spec_from_file_location",
        "SourceFileLoader",
        "machinery.SourceFileLoader",
        "importlib.machinery.SourceFileLoader",
        "pkgutil.resolve_name",
        "pydoc.locate",
        "exec",
        "eval",
    }
)


def _normalized_source(path: Path) -> str:
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"runtime source cannot be read as UTF-8: {path}") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _module_name_for_path(relative: str) -> str | None:
    path = Path(relative)
    parts = path.parts
    if len(parts) < 3 or parts[0] != "src" or path.suffix != ".py":
        return None
    module_parts = list(parts[1:])
    if module_parts[-1] == "__init__.py":
        module_parts.pop()
    else:
        module_parts[-1] = Path(module_parts[-1]).stem
    return ".".join(module_parts)


def _module_relative_path(project_root: Path, module: str) -> str | None:
    if not module.startswith(tuple(f"{name}." for name in LOCAL_SOURCE_NAMESPACES)) and (
        module not in LOCAL_SOURCE_NAMESPACES
    ):
        return None
    stem = Path("src", *module.split("."))
    candidates = (stem.with_suffix(".py"), stem / "__init__.py")
    existing = [path for path in candidates if (project_root / path).is_file()]
    if len(existing) > 1:
        raise ValueError(f"local import is ambiguous between module and package: {module}")
    if not existing:
        return None
    return existing[0].as_posix()


def _package_ancestors(project_root: Path, module: str) -> list[str]:
    parts = module.split(".")
    result: list[str] = []
    for length in range(1, len(parts)):
        relative = Path("src", *parts[:length], "__init__.py")
        if (project_root / relative).is_file():
            result.append(relative.as_posix())
    return result


def _absolute_from_import(
    module_name: str | None,
    *,
    is_package: bool,
    node: ast.ImportFrom,
) -> str:
    if node.level == 0:
        return str(node.module or "")
    if not module_name:
        raise ValueError("relative local import appears outside a source package")
    package_parts = module_name.split(".")
    if not is_package:
        package_parts = package_parts[:-1]
    climb = node.level - 1
    if climb > len(package_parts):
        raise ValueError("relative local import escapes its package")
    base = package_parts[: len(package_parts) - climb]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if not isinstance(node, ast.Attribute):
        return None
    parts = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
        return ".".join(reversed(parts))
    return None


def _dynamic_import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    dynamic_modules = {
        "builtins",
        "importlib",
        "importlib.machinery",
        "importlib.util",
        "pydoc",
        "pkgutil",
        "runpy",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in dynamic_modules:
                    aliases[str(alias.asname or alias.name)] = str(alias.name)
        elif isinstance(node, ast.ImportFrom) and str(node.module or "") in dynamic_modules:
            for alias in node.names:
                qualified = f"{node.module}.{alias.name}"
                if qualified in _DYNAMIC_IMPORT_CALLS or alias.name in _DYNAMIC_IMPORT_CALLS:
                    aliases[str(alias.asname or alias.name)] = qualified
    return aliases


def _dynamic_import_sites(tree: ast.AST) -> list[tuple[int, str]]:
    aliases = _dynamic_import_aliases(tree)
    result: set[tuple[int, str]] = set()

    def resolve(node: ast.AST) -> str:
        name = str(_call_name(node) or "")
        if not name:
            return ""
        head, separator, tail = name.partition(".")
        if head in aliases:
            return aliases[head] + (f".{tail}" if separator else "")
        return name

    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(
            getattr(node, "ctx", None), ast.Load
        ):
            resolved = resolve(node)
            if resolved in _DYNAMIC_IMPORT_CALLS or resolved.rsplit(".", 1)[-1] in {
                "exec_module",
                "load_module",
            }:
                result.add((int(node.lineno), resolved))
        if isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and str(key.value) in _DYNAMIC_IMPORT_CALLS:
                result.add((int(node.lineno), str(key.value)))
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        resolved = resolve(node.func)
        if (
            str(name or "") == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and isinstance(node.args[1], ast.Constant)
        ):
            owner = aliases.get(node.args[0].id, node.args[0].id)
            target = f"{owner}.{node.args[1].value}"
            if target in _DYNAMIC_IMPORT_CALLS:
                result.add((int(node.lineno), target))
    return sorted(result)


def _fragment_source(
    source: str,
    *,
    path: str,
    names: Sequence[str],
    project_root: Path,
) -> tuple[list[dict[str, Any]], set[str]]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise ValueError(f"runtime fragment source is not valid Python: {path}") from exc
    by_name = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    fragments: list[dict[str, Any]] = []
    imported_modules: set[str] = set()
    lines = source.splitlines(keepends=True)
    for name in sorted(names):
        node = by_name.get(name)
        if node is None or node.end_lineno is None:
            raise ValueError(f"sealed runtime fragment is missing: {path}#{name}")
        start = min(
            [node.lineno]
            + [decorator.lineno for decorator in getattr(node, "decorator_list", [])]
        )
        payload = "".join(lines[start - 1 : node.end_lineno]).encode("utf-8")
        fragments.append(
            {
                "path": f"{path}#{name}",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "kind": "source_fragment",
            }
        )
        loaded_names = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        relevant_imports: list[ast.Import | ast.ImportFrom] = []
        for candidate in tree.body:
            if not isinstance(candidate, (ast.Import, ast.ImportFrom)):
                continue
            bindings = {
                str(alias.asname or alias.name.split(".")[0])
                for alias in candidate.names
            }
            if bindings & loaded_names:
                relevant_imports.append(candidate)
        import_payload = "".join(
            "".join(lines[item.lineno - 1 : item.end_lineno])
            for item in relevant_imports
            if item.end_lineno is not None
        ).encode("utf-8")
        if not import_payload:
            raise ValueError(f"sealed runtime fragment imports are missing: {path}#{name}")
        fragments.append(
            {
                "path": f"{path}#{name}.imports",
                "bytes": len(import_payload),
                "sha256": hashlib.sha256(import_payload).hexdigest(),
                "kind": "source_fragment_imports",
            }
        )
        fragment_tree = ast.Module(body=relevant_imports, type_ignores=[])
        modules, nested_fragments = _imported_local_modules(
            fragment_tree,
            module_name=_module_name_for_path(path),
            project_root=project_root,
            relative=f"{path}#{name}.imports",
        )
        if nested_fragments:
            raise ValueError("nested fragment-boundary import is not supported")
        imported_modules.update(modules)
    return fragments, imported_modules


def _normalized_seal_payload(relative: str, source: str) -> bytes:
    if relative == "src/quant_platform/transparent_baseline_runner.py":
        source, replacements = _RUNNER_SEAL_ASSIGNMENT.subn(
            lambda match: f'{match.group(1)}"<sealed-at-release>"{match.group(2)}',
            source,
        )
        if replacements != 50:
            raise ValueError("transparent runner seal constants cannot be normalized")
    elif relative == "src/quant_data/database.py":
        for constraint in _DATABASE_RUNTIME_IDENTITY_CONSTRAINTS:
            marker = f'name="{constraint}"'
            marker_at = source.find(marker)
            if marker_at < 0:
                raise ValueError(
                    f"database runtime identity constraint is missing: {constraint}"
                )
            block_start = source.rfind("    CheckConstraint(", 0, marker_at)
            block_end = source.find("    ),", marker_at)
            if block_start < 0 or block_end < 0:
                raise ValueError(
                    f"database runtime identity constraint is malformed: {constraint}"
                )
            block_end += len("    ),")
            block = source[block_start:block_end]
            block, replacements = _SHA256.subn("<sealed-at-release>", block)
            if replacements != 2:
                raise ValueError(
                    f"database runtime hashes cannot be normalized: {constraint}"
                )
            source = source[:block_start] + block + source[block_end:]
    return source.encode("utf-8")


def _imported_local_modules(
    tree: ast.AST,
    *,
    module_name: str | None,
    project_root: Path,
    relative: str,
) -> tuple[set[str], dict[str, set[str]]]:
    modules: set[str] = set()
    fragments: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = str(alias.name)
                if name in LOCAL_IMPORT_FRAGMENT_BOUNDARIES:
                    raise ValueError(
                        f"fragment-boundary module must use from-import: {relative}:{node.lineno}"
                    )
                if name in LOCAL_SOURCE_NAMESPACES or name.startswith(
                    tuple(f"{value}." for value in LOCAL_SOURCE_NAMESPACES)
                ):
                    if _module_relative_path(project_root, name) is None:
                        raise ValueError(f"local import cannot be resolved: {name}")
                    modules.add(name)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_from_import(
                module_name,
                is_package=relative.endswith("/__init__.py"),
                node=node,
            )
            if not base:
                continue
            if base in LOCAL_IMPORT_FRAGMENT_BOUNDARIES:
                allowed = set(LOCAL_IMPORT_FRAGMENT_BOUNDARIES[base])
                requested = {str(alias.name) for alias in node.names}
                if "*" in requested or not requested <= allowed:
                    raise ValueError(
                        f"local fragment import is outside its sealed boundary: "
                        f"{relative}:{node.lineno}"
                    )
                fragments.setdefault(base, set()).update(requested)
                continue
            is_local = base in LOCAL_SOURCE_NAMESPACES or base.startswith(
                tuple(f"{value}." for value in LOCAL_SOURCE_NAMESPACES)
            )
            base_path = _module_relative_path(project_root, base) if is_local else None
            if is_local and base_path is None:
                raise ValueError(f"local from-import cannot be resolved: {base}")
            if base_path is not None:
                modules.add(base)
                for alias in node.names:
                    child = f"{base}.{alias.name}"
                    if _module_relative_path(project_root, child) is not None:
                        modules.add(child)
    return modules, fragments


def local_python_source_closure_inventory(
    project_root: Path,
    *,
    entry_paths: Sequence[str],
    supplement_modules: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve and hash one fail-closed local Python source closure."""

    root = project_root.resolve()
    queue: deque[str] = deque()
    queued: set[str] = set()

    def enqueue(relative: str) -> None:
        normalized = Path(relative).as_posix()
        try:
            candidate = (root / normalized).resolve()
            candidate.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"runtime source escapes the project root: {relative}") from exc
        if not candidate.is_file() or candidate.suffix != ".py":
            raise ValueError(f"runtime source path is missing or not Python: {relative}")
        if normalized not in queued:
            queue.append(normalized)
            queued.add(normalized)

    for relative in entry_paths:
        enqueue(relative)
    for module in supplement_modules:
        relative = _module_relative_path(root, str(module))
        if relative is None:
            raise ValueError(f"supplemental local module cannot be resolved: {module}")
        enqueue(relative)
        for ancestor in _package_ancestors(root, str(module)):
            enqueue(ancestor)

    records: dict[str, dict[str, Any]] = {}
    requested_fragments: dict[str, set[str]] = {}
    while queue:
        relative = queue.popleft()
        source = _normalized_source(root / relative)
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            raise ValueError(f"runtime source is not valid Python: {relative}") from exc
        dynamic_sites = _dynamic_import_sites(tree)
        if dynamic_sites:
            rendered = ", ".join(f"{name}@{line}" for line, name in dynamic_sites)
            raise ValueError(
                f"runtime source has an unsealed dynamic import/eval site: "
                f"{relative}: {rendered}"
            )
        payload = _normalized_seal_payload(relative, source)
        records[relative] = {
            "path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "kind": "module",
        }
        module_name = _module_name_for_path(relative)
        modules, fragments = _imported_local_modules(
            tree,
            module_name=module_name,
            project_root=root,
            relative=relative,
        )
        for imported in sorted(modules):
            resolved = _module_relative_path(root, imported)
            if resolved is None:
                raise ValueError(f"local import disappeared during closure: {imported}")
            enqueue(resolved)
            for ancestor in _package_ancestors(root, imported):
                enqueue(ancestor)
        for fragment_module, names in fragments.items():
            requested_fragments.setdefault(fragment_module, set()).update(names)

    fragment_records: list[dict[str, Any]] = []
    for module, names in sorted(requested_fragments.items()):
        relative = _module_relative_path(root, module)
        if relative is None:
            raise ValueError(f"fragment-boundary module cannot be resolved: {module}")
        records_for_fragment, imported_modules = _fragment_source(
            _normalized_source(root / relative),
            path=relative,
            names=sorted(names),
            project_root=root,
        )
        fragment_records.extend(records_for_fragment)
        missing_from_closure = []
        for imported in sorted(imported_modules):
            imported_path = _module_relative_path(root, imported)
            if imported_path is None:
                raise ValueError(f"fragment local import cannot be resolved: {imported}")
            if imported_path not in records:
                missing_from_closure.append(imported)
        if missing_from_closure:
            raise ValueError(
                "fragment local imports require explicit closure supplements: "
                + ", ".join(missing_from_closure)
            )

    inventory = sorted([*records.values(), *fragment_records], key=lambda item: item["path"])
    return {
        "contract_version": SOURCE_CLOSURE_CONTRACT_VERSION,
        "entry_paths": sorted(Path(value).as_posix() for value in entry_paths),
        "supplement_modules": sorted(str(value) for value in supplement_modules),
        "fragment_boundaries": {
            module: sorted(names) for module, names in sorted(requested_fragments.items())
        },
        "inventory": inventory,
    }


def local_python_source_closure_sha256(
    project_root: Path,
    *,
    entry_paths: Sequence[str],
    supplement_modules: Sequence[str] = (),
) -> str:
    manifest = local_python_source_closure_inventory(
        project_root,
        entry_paths=entry_paths,
        supplement_modules=supplement_modules,
    )
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def position_risk_source_closure_inventory(project_root: Path) -> dict[str, Any]:
    return local_python_source_closure_inventory(
        project_root,
        entry_paths=POSITION_RISK_SOURCE_ENTRY_PATHS,
        supplement_modules=POSITION_RISK_SOURCE_SUPPLEMENT_MODULES,
    )


def position_risk_source_closure_sha256(project_root: Path) -> str:
    return local_python_source_closure_sha256(
        project_root,
        entry_paths=POSITION_RISK_SOURCE_ENTRY_PATHS,
        supplement_modules=POSITION_RISK_SOURCE_SUPPLEMENT_MODULES,
    )


def closure_paths(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or any(
        not isinstance(item, Mapping) for item in inventory
    ):
        raise ValueError("runtime source closure inventory is missing")
    return tuple(sorted(str(dict(item)["path"]) for item in inventory))
