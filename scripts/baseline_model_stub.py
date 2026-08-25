from torch import nn


class GovernedBaselineStub(nn.Module):
    """Identity-only module; the sandbox selects the governed LightGBM engine."""

    def __init__(self, num_features: int = 20) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, values):
        return self.linear(values)


model_cls = GovernedBaselineStub
