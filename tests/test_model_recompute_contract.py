from pathlib import Path

import pytest

from quant_platform.model_recompute import execute_model_candidate, validate_model_code

VALID_MODEL = """
import torch
from torch import nn

class SafeModel(nn.Module):
    def __init__(self, num_features=20):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)

model_cls = SafeModel
"""


def test_model_code_is_limited_to_pure_torch_definition() -> None:
    validate_model_code(VALID_MODEL)
    with pytest.raises(ValueError, match="forbidden module"):
        validate_model_code("import os\nclass X: pass\nmodel_cls = X\n")
    with pytest.raises(ValueError, match="forbidden builtin"):
        validate_model_code(
            "class X:\n    def forward(self):\n        return open('/etc/passwd')\nmodel_cls = X\n"
        )


def test_model_recompute_fails_closed_without_immutable_inputs(tmp_path: Path) -> None:
    code = tmp_path / "model.py"
    code.write_text(VALID_MODEL, encoding="utf-8")
    provider = tmp_path / "qlib"
    provider.mkdir()
    runner = tmp_path / "runner.py"
    runner.write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="code hash"):
        execute_model_candidate(
            code_path=code,
            provider_path=provider,
            manifest={
                "candidate_id": "model-1",
                "code_sha256": "0" * 64,
                "final_oos_opened": False,
            },
            workspace=tmp_path / "work",
            runner_path=runner,
        )
