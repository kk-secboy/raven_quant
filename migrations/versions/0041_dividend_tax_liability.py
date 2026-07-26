"""Conservative dividend tax liability ledger columns (design draft 5.6).

Under the at-sale dividend tax regime the exact tax is unknown until the
shares are sold, so the ex-date receivable must be paired with a conservative
tax liability: NAV = cash + market value + receivables - tax liabilities, and
the sale later recognizes only the difference between the actual tax and the
accrued liability (over-accrual reversal / under-accrual top-up).

- simulation_dividend_entitlements.liability_per_share: per-share liability
  accrued at the ex-date holding-period bracket (a per-lot conservative upper
  bound, since holding periods only lengthen and rates only fall); released
  proportionally as untaxed_quantity is consumed. Existing rows default to 0
  (nothing accrued; the full tax is then recognized at sale).
- simulation_dividend_actions.tax_liability_amount: total liability accrued
  by one ex-date application (cash dividend and bonus-par entitlements).
- simulation_nav.corporate_tax_liabilities: unsettled liability total carried
  as a NAV deduction per day.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0041_dividend_tax_liability"
down_revision: str | None = "0040_oos_vintage_sealing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "simulation_dividend_entitlements",
        sa.Column(
            "liability_per_share",
            sa.Numeric(20, 8),
            nullable=False,
            server_default="0",
        ),
        schema="quantlab",
    )
    op.add_column(
        "simulation_dividend_actions",
        sa.Column(
            "tax_liability_amount",
            sa.Numeric(20, 6),
            nullable=False,
            server_default="0",
        ),
        schema="quantlab",
    )
    op.add_column(
        "simulation_nav",
        sa.Column(
            "corporate_tax_liabilities",
            sa.Numeric(20, 6),
            nullable=False,
            server_default="0",
        ),
        schema="quantlab",
    )


def downgrade() -> None:
    op.drop_column("simulation_nav", "corporate_tax_liabilities", schema="quantlab")
    op.drop_column("simulation_dividend_actions", "tax_liability_amount", schema="quantlab")
    op.drop_column(
        "simulation_dividend_entitlements", "liability_per_share", schema="quantlab"
    )
