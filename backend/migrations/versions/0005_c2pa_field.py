"""c2pa_field

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-04 00:00:00.000000

Ajoute la colonne c2pa_result (JSON nullable) sur analyses.
Stocke le résultat structuré de la vérification C2PA (provenance cryptographique).
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "analyses",
        sa.Column("c2pa_result", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("analyses", "c2pa_result")
