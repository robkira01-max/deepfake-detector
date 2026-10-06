"""audit_chain

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-04 00:00:00.000000

Ajoute la colonne previous_entry_hash sur audit_logs pour le chaînage SHA-256.
entry_hash existait déjà (nullable) ; previous_entry_hash est nullable pour
les entrées historiques antérieures à cette migration.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "audit_logs",
        sa.Column("previous_entry_hash", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("audit_logs", "previous_entry_hash")
