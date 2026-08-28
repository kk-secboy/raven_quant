from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from rdagent.scenarios.finetune.scen.scenario import LLMFinetuneScen
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorScenario
from rdagent.scenarios.qlib.experiment.factor_from_report_experiment import (
    QlibFactorFromReportScenario,
)
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelScenario
from rdagent.scenarios.qlib.experiment.quant_experiment import QlibQuantScenario
from rdagent.scenarios.shared.get_runtime_info import get_runtime_environment_by_env
from rdagent.utils.env import QlibDockerConf, QTDockerEnv

from quant_data.research_assets import (
    FINETUNE_BENCHMARK_ALLOWLIST,
    FINETUNE_FINANCEIQ_DATA_ROOT,
    validate_finetune_dataset_info,
)
from quant_platform.factor_library import (
    library_release_definition,
    qlib_expression_contract,
)


def _factor_library_prompt_reference() -> dict[str, Any]:
    """Return the auditable library identity without prompt-useless member hashes."""

    release = library_release_definition()
    return {
        key: value
        for key, value in release.items()
        if key != "member_definition_sha256"
    }


def _governed_qlib_runtime_environment() -> str:
    """Describe the preloaded Qlib sandbox without upstream's empty-volume bug."""

    conf = QlibDockerConf()
    if not conf.extra_volumes:
        raise RuntimeError("governed Qlib sandbox has no mounted dataset")
    env = QTDockerEnv(conf=conf)
    env.prepare()
    return get_runtime_environment_by_env(env=env)


def _install_shared_factor_data_generator() -> None:
    """Make RD-Agent's Qlib template visible to the remote Docker daemon."""

    from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
    from rdagent.scenarios.qlib.experiment import utils as experiment_utils

    shared_root = Path(
        os.environ.get("RDAGENT_DOCKER_SHARED_ROOT", "/data")
    ).resolve(strict=True)
    workspace = Path(os.environ["WORKSPACE_PATH"]).resolve(strict=False)
    try:
        workspace.relative_to(shared_root)
    except ValueError as exc:
        raise RuntimeError("RD-Agent workspace is outside the Docker shared root") from exc
    workspace.mkdir(parents=True, exist_ok=True)
    workspace = workspace.resolve(strict=True)
    source = Path(experiment_utils.__file__).resolve().parent / "factor_data_template"
    target = workspace / "rdagent-factor-data-template"

    def generate_data_folder_from_governed_qlib() -> None:
        if target.exists():
            if target.is_symlink() or not target.is_dir():
                raise RuntimeError("governed factor template staging path is invalid")
            shutil.rmtree(target)
        shutil.copytree(source, target)
        shutil.copyfile(
            Path(__file__).with_name("rdagent_factor_data_generator.py"),
            target / "generate.py",
        )
        environment = QTDockerEnv()
        environment.prepare()
        execute_log = environment.check_output(
            local_path=str(target),
            entry="python generate.py",
        )
        full = target / "daily_pv_all.h5"
        debug = target / "daily_pv_debug.h5"
        if not full.is_file() or not debug.is_file():
            raise RuntimeError(
                "governed Qlib factor data generation failed: " + execute_log[-2000:]
            )
        for destination, source_file in (
            (Path(FACTOR_COSTEER_SETTINGS.data_folder), full),
            (Path(FACTOR_COSTEER_SETTINGS.data_folder_debug), debug),
        ):
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, destination / "daily_pv.h5")
            shutil.copyfile(target / "README.md", destination / "README.md")

    experiment_utils.generate_data_folder_from_qlib = (
        generate_data_folder_from_governed_qlib
    )


class _QuantLabScenarioConstraints:
    """Add platform-owned objective and feature identity without accepting code."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if isinstance(
            self,
            (QlibFactorScenario, QlibFactorFromReportScenario, QlibQuantScenario),
        ):
            _install_shared_factor_data_generator()
        super().__init__(*args, **kwargs)  # type: ignore[misc]

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
        if isinstance(
            self,
            (QlibFactorScenario, QlibFactorFromReportScenario, QlibQuantScenario),
        ):
            period_prefix = "QLIB_QUANT" if isinstance(self, QlibQuantScenario) else "QLIB_FACTOR"
            governed["research_periods"] = json.dumps(
                {
                    name.lower(): os.getenv(f"{period_prefix}_{name}", "")
                    for name in (
                        "TRAIN_START",
                        "TRAIN_END",
                        "VALID_START",
                        "VALID_END",
                        "TEST_START",
                        "TEST_END",
                    )
                },
                sort_keys=True,
            )
            governed["factor_expression_contract"] = json.dumps(
                qlib_expression_contract(), ensure_ascii=False, sort_keys=True
            )
            governed["factor_library_reference"] = json.dumps(
                {
                    **_factor_library_prompt_reference(),
                    "active_library_version_id": os.getenv(
                        "QUANTLAB_FACTOR_LIBRARY_VERSION_ID", ""
                    ),
                    "active_library_definition_sha256": os.getenv(
                        "QUANTLAB_FACTOR_LIBRARY_DEFINITION_SHA256", ""
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            governed["factor_submission_rule"] = (
                "Return a testable hypothesis, one governed Qlib expression, its required "
                "fields, and a suggested economic family. Platform validation overrides the "
                "suggested family and compares every proposal with the frozen full library. "
                "If the upstream loop also emits Python, keep the immutable input/output row "
                "set, never call dropna, use literal daily_pv.h5 and result.h5 names, prefer "
                "groupby operations on the existing MultiIndex, and do not invoke filesystem "
                "enumeration, mutation, subprocess, network, or dynamic-code capabilities."
            )
        if feature_set_id or feature_set_sha256:
            governed["feature_set_id"] = feature_set_id
            governed["feature_set_definition_sha256"] = feature_set_sha256
            governed["feature_set_rule"] = (
                "Use the already mounted, immutable base feature set. Do not replace it "
                "or claim a different feature definition."
            )
        if isinstance(self, (QlibModelScenario, QlibQuantScenario)):
            governed["model_resource_rule"] = (
                "The governed production queue is CPU-only unless the platform explicitly "
                "declares a GPU capability. Prefer a compact tabular baseline before proposing "
                "GRU, LSTM, Transformer, or other sequence models. Keep architecture width, "
                "depth, epochs, and lookback modest; independent validation applies fixed "
                "resource caps without shortening the registered dates or stock universe."
            )
        if isinstance(self, QlibQuantScenario):
            raw_champion = os.getenv(
                "QUANTLAB_PREDICTION_CHAMPION_JSON", ""
            ).strip()
            if not raw_champion:
                raise RuntimeError(
                    "governed fin_quant has no frozen prediction champion"
                )
            try:
                champion = json.loads(raw_champion)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "governed fin_quant prediction champion is invalid"
                ) from exc
            if (
                not isinstance(champion, dict)
                or champion.get("kind") not in {"model", "ensemble"}
                or not str(champion.get("candidate_id") or "")
                or not str(champion.get("candidate_manifest_sha256") or "")
                or not str(champion.get("admission_evidence_sha256") or "")
                or not str(champion.get("selection_evidence_sha256") or "")
            ):
                raise RuntimeError(
                    "governed fin_quant prediction champion contract is incomplete"
                )
            governed["prediction_champion"] = json.dumps(
                champion, ensure_ascii=False, sort_keys=True
            )
            governed["quant_ablation_rule"] = (
                "Treat the frozen prediction champion as the incumbent. Propose a "
                "challenger model and/or new factors, but never rename the incumbent. "
                "The platform independently evaluates factor-only as incumbent plus new "
                "factors, model-only as challenger plus baseline factors, and joint as "
                "challenger plus new factors. An ensemble incumbent is fail-closed until "
                "every member can be independently retrained; never substitute one member "
                "or a fixed LightGBM baseline."
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
    def get_runtime_environment(self) -> str:
        return _governed_qlib_runtime_environment()


class QuantLabModelScenario(_QuantLabScenarioConstraints, QlibModelScenario):
    def get_runtime_environment(self) -> str:
        return _governed_qlib_runtime_environment()


class QuantLabQuantScenario(_QuantLabScenarioConstraints, QlibQuantScenario):
    def get_runtime_environment(self, tag: str | None = None) -> str:
        if tag not in {None, "factor", "model"}:
            raise ValueError("unsupported Qlib quant runtime tag")
        description = _governed_qlib_runtime_environment()
        if tag is not None:
            return description
        return (
            "=== [Environment to generate the factors] ===\n"
            f"{description.strip()}\n\n"
            "=== [Environment to train the models] ===\n"
            f"{description.strip()}"
        )


class QuantLabFactorFromReportScenario(
    _QuantLabScenarioConstraints, QlibFactorFromReportScenario
):
    def get_runtime_environment(self) -> str:
        return _governed_qlib_runtime_environment()


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
