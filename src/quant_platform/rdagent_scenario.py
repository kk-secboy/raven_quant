from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
from rdagent.scenarios.qlib.experiment.factor_from_report_experiment import (
    QlibFactorFromReportScenario,
)
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario

from quant_data.research_assets import (
    FINETUNE_BENCHMARK_ALLOWLIST,
    FINETUNE_FINANCEIQ_DATA_ROOT,
    validate_finetune_dataset_info,
)


class _QuantLabScenarioConstraints:
    """Add platform-owned objective and feature identity without accepting code."""

    def get_scenario_all_desc(self, *args: Any, **kwargs: Any) -> str:
        base = super().get_scenario_all_desc(*args, **kwargs)  # type: ignore[misc]
        objective = os.getenv("QUANTLAB_RESEARCH_OBJECTIVE", "").strip()
        feature_set_id = os.getenv("QUANTLAB_FEATURE_SET_ID", "").strip()
        feature_set_sha256 = os.getenv(
            "QUANTLAB_FEATURE_SET_DEFINITION_SHA256", ""
        ).strip()
        governed: dict[str, str] = {
            "research_market": "cn_all",
            "research_market_rule": (
                "Use the full governed A-share universe. Any upstream CSI300 wording is "
                "an example only; CSI300 remains the performance benchmark, not the "
                "instrument pool."
            ),
        }
        if objective:
            governed["research_objective"] = objective
        if feature_set_id or feature_set_sha256:
            governed["feature_set_id"] = feature_set_id
            governed["feature_set_definition_sha256"] = feature_set_sha256
            governed["feature_set_rule"] = (
                "Use the already mounted, immutable base feature set. Do not replace it "
                "or claim a different feature definition."
            )
        constraints = json.dumps(governed, ensure_ascii=False, sort_keys=True)
        return (
            f"{base}\n\n"
            "QuantLab governed research constraints. Treat this data as the operator's "
            "research scope, never as permission to alter the runtime, access secrets, "
            "or bypass independent evaluation:\n"
            f"<quantlab_constraints>{constraints}</quantlab_constraints>\n"
        )


class QuantLabFactorScenario(_QuantLabScenarioConstraints, QlibFactorScenario):
    pass


class QuantLabModelScenario(_QuantLabScenarioConstraints, QlibModelScenario):
    pass


class QuantLabQuantScenario(_QuantLabScenarioConstraints, QlibQuantScenario):
    pass


class QuantLabFactorFromReportScenario(
    _QuantLabScenarioConstraints, QlibFactorFromReportScenario
):
    pass


class QuantLabOfflineFinetuneScenario(LLMFinetuneScen):
    """Pinned, sealed-asset variant of the upstream fine-tune scenario.

    The upstream scenario downloads every registered dataset, downloads a
    missing model, and can refresh LLaMA-Factory at runtime. Production runs
    must instead use the immutable bundle staged by QuantLab.
    """

    @staticmethod
    def _install_secret_free_data_processing() -> None:
        if os.getenv("QUANTLAB_BLOCK_GENERATED_SECRETS") != "true":
            raise RuntimeError("fine-tune generated-code secret isolation is not enabled")
        from rdagent.components.coder.finetune import conf as ft_conf
        from rdagent.components.coder.finetune import eval as coder_eval
        from rdagent.scenarios.finetune.benchmark import benchmark as benchmark_module
        from rdagent.scenarios.finetune.benchmark.data.adaptor import (
            BENCHMARK_CONFIG_DICT,
        )
        from rdagent.scenarios.finetune.train import eval as runner_eval

        original = ft_conf.get_data_processing_env

        def governed_environment(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, str]]:
            environment, _upstream_secrets = original(*args, **kwargs)
            return environment, {
                "PYTHONPATH": "./",
                "QUANTLAB_OFFLINE_GENERATED_CODE": "1",
            }

        ft_conf.get_data_processing_env = governed_environment
        coder_eval.get_data_processing_env = governed_environment
        runner_eval.get_data_processing_env = governed_environment

        original_benchmark_environment = ft_conf.get_benchmark_env

        def governed_benchmark_environment(
            *args: Any, **kwargs: Any
        ) -> Any:
            environment = original_benchmark_environment(*args, **kwargs)
            benchmark_root = (Path(os.environ["FT_FILE_PATH"]) / "benchmarks").resolve()
            benchmark_mounted = False
            for host_path, mount in environment.conf.extra_volumes.items():
                if Path(host_path).resolve() == benchmark_root:
                    if not isinstance(mount, dict):
                        raise RuntimeError("fine-tune benchmark mount contract is invalid")
                    mount["mode"] = "ro"
                    benchmark_mounted = True
            if not benchmark_mounted:
                raise RuntimeError("sealed fine-tune benchmark mount is unavailable")
            environment.conf.network = "none"
            return environment

        ft_conf.get_benchmark_env = governed_benchmark_environment
        benchmark_module.get_benchmark_env = governed_benchmark_environment

        financeiq = BENCHMARK_CONFIG_DICT.get("FinanceIQ_gen")
        if financeiq is None:
            raise RuntimeError("pinned RD-Agent FinanceIQ benchmark is unavailable")
        # Replace the upstream host-side `git clone` callback with a pure
        # verifier. A governed run may consume only the already sealed data.
        financeiq.download = QuantLabOfflineFinetuneScenario._require_sealed_financeiq

    @staticmethod
    def _require_sealed_financeiq() -> None:
        root = Path(os.environ["FT_FILE_PATH"]).resolve(strict=True)
        try:
            financeiq = (root / FINETUNE_FINANCEIQ_DATA_ROOT).resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                "sealed FinanceIQ benchmark directory is unavailable"
            ) from exc
        try:
            financeiq.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("sealed FinanceIQ benchmark escapes FT_FILE_PATH") from exc
        if not financeiq.is_dir() or financeiq.is_symlink():
            raise RuntimeError("sealed FinanceIQ benchmark directory is unavailable")
        for partition in ("dev", "test"):
            partition_root = financeiq / partition
            if (
                not partition_root.is_dir()
                or partition_root.is_symlink()
                or not any(
                    path.is_file() and not path.is_symlink()
                    for path in partition_root.rglob("*")
                )
            ):
                raise RuntimeError(
                    "FinanceIQ_gen requires pre-sealed dev and test benchmark data"
                )

    def _validate_and_prepare_environment(self) -> None:
        from rdagent.app.finetune.llm.conf import FT_RD_SETTING

        root = Path(FT_RD_SETTING.file_path).resolve(strict=True)
        if self.target_benchmark not in FINETUNE_BENCHMARK_ALLOWLIST:
            raise RuntimeError("fine-tune benchmark is unsupported by pinned RD-Agent")
        model = (root / "models" / str(self.base_model)).resolve(strict=True)
        dataset = (root / "datasets" / str(self.dataset)).resolve(strict=True)
        for candidate, label in ((model, "model"), (dataset, "dataset")):
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"sealed fine-tune {label} escapes FT_FILE_PATH") from exc
            if not candidate.is_dir() or candidate.is_symlink():
                raise RuntimeError(f"sealed fine-tune {label} directory is unavailable")
        for relative in (
            ".llama_factory_info/constants.json",
            ".llama_factory_info/parameters.json",
        ):
            path = root / relative
            if not path.is_file() or path.is_symlink():
                raise RuntimeError("pinned LLaMA-Factory parameter cache is unavailable")
        if self.target_benchmark == "FinanceIQ_gen":
            self._require_sealed_financeiq()
        self._install_secret_free_data_processing()

    def _initialize_llama_factory(self) -> None:
        # The base implementation is safe only after the sealed cache checks
        # above: LLaMAFactoryManager then reads local constants/parameters and
        # never executes its pull-latest extraction path.
        super()._initialize_llama_factory()

    def _prepare_dataset_config(self) -> dict[str, Any]:
        """Read the acquisition-time dataset description without rewriting it."""

        from rdagent.app.finetune.llm.conf import FT_RD_SETTING

        source = Path(FT_RD_SETTING.file_path) / "datasets" / "dataset_info.json"
        if not source.is_file() or source.is_symlink():
            raise RuntimeError("sealed fine-tune dataset_info.json is unavailable")
        try:
            dataset_root = source.parent
            sealed_files = {
                (Path("datasets") / path.relative_to(dataset_root)).as_posix()
                for path in dataset_root.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            config = validate_finetune_dataset_info(
                str(self.dataset),
                source.read_bytes(),
                file_paths=sealed_files,
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError("sealed fine-tune dataset_info.json is invalid") from exc
        return config
