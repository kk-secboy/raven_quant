"""Bind RD-Agent's neural model templates to the features they actually load."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

BASELINE_MODEL_CONFIG = "conf_baseline_factors_model.yaml"
MODEL_CONFIGS = frozenset({
    BASELINE_MODEL_CONFIG,
    "conf_sota_factors_model.yaml",
    "conf_combined_factors_sota_model.yaml",
})
_LEGACY_DIMENSION = '"num_features": 20,'
_BOUND_DIMENSION = '"num_features": {{ num_features }}'


def _feature_list(value: Any, name: str) -> list[str]:
    try:
        result = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"fin_quant invalid {name}") from exc
    if not isinstance(result, list) or not result or any(
        not isinstance(item, str) or not item.strip() for item in result
    ):
        raise ValueError(f"fin_quant requires a nonempty {name} list")
    return result


def _parquet_feature_names(path: Path) -> list[str]:
    # Read only schema metadata, never load the full research matrix a second time.
    import pyarrow.parquet as pq

    metadata = pq.read_schema(path).pandas_metadata
    if metadata is None:
        raise ValueError("fin_quant combined factors are missing pandas schema metadata")
    index_fields = {item for item in metadata["index_columns"] if isinstance(item, str)}
    names = []
    for column in metadata["columns"]:
        if column["field_name"] in index_fields:
            continue
        name = ast.literal_eval(column["name"])
        if (not isinstance(name, tuple) or len(name) != 2 or name[0] != "feature"
                or not isinstance(name[1], str) or not name[1]):
            raise ValueError("fin_quant combined factor column is not a named feature")
        names.append(name[1])
    if not names or len(names) != len(set(names)):
        raise ValueError("fin_quant combined factors must have unique, nonempty columns")
    return names


def bind_model_dimensions(
    workspace: Any, config_name: str, run_env: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Validate the feature contract before Qlib starts, preserving the model itself."""
    if config_name not in MODEL_CONFIGS:
        raise ValueError("fin_quant unsupported neural model configuration")
    names = _feature_list(run_env.get("feature_names"), "feature_names")
    expressions = _feature_list(run_env.get("feature_expressions"), "feature_expressions")
    if len(names) != len(expressions) or len(names) != len(set(names)):
        raise ValueError("fin_quant base feature names and expressions are inconsistent")
    expected = len(names)
    if config_name != BASELINE_MODEL_CONFIG:
        additional = _parquet_feature_names(
            Path(workspace.workspace_path) / "combined_factors_df.parquet")
        if set(names) & set(additional):
            raise ValueError("fin_quant base and combined factor names overlap")
        expected += len(additional)
    declared = run_env.get("num_features")
    if declared is not None and str(declared) != str(expected):
        raise ValueError(
            f"fin_quant model input mismatch: declared {declared}, loaded features {expected}")
    template = workspace.file_dict[config_name]
    if config_name == BASELINE_MODEL_CONFIG and template.count(_LEGACY_DIMENSION) == 1:
        template = template.replace(_LEGACY_DIMENSION, _BOUND_DIMENSION + ",")
    if template.count('"num_features":') != 1 or template.count(_BOUND_DIMENSION) != 1:
        raise ValueError("fin_quant model template does not bind the actual feature count")
    return template, {**run_env, "num_features": str(expected)}


def enable_fin_quant_model_dimensions() -> None:
    from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace

    if getattr(QlibFBWorkspace, "_quantlab_model_dimensions_enabled", False):
        return
    original_execute = QlibFBWorkspace.execute

    def execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = inspect.signature(original_execute).bind(self, *args, **kwargs)
        config_name = bound.arguments.get("qlib_config_name", "conf.yaml")
        if config_name in MODEL_CONFIGS:
            template, env = bind_model_dimensions(
                self, config_name, dict(bound.arguments.get("run_env") or {}))
            if template != self.file_dict[config_name]:
                self.inject_files(**{config_name: template})
            bound.arguments["run_env"] = env
        return original_execute(*bound.args, **bound.kwargs)

    QlibFBWorkspace.execute = execute
    QlibFBWorkspace._quantlab_model_dimensions_enabled = True
