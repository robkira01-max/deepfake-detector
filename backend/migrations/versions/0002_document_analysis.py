"""document_analysis

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-23 00:00:00.000000

v2.0 — Ajoute :
  - Valeur 'document' au type enum MediaType (media_files.media_type)
  - Table document_analyses
  - Nouvelles valeurs enum AuditAction (v2.0)
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Étendre l'enum MediaType ───────────────────────────────────────────────
    # PostgreSQL : ALTER TYPE ... ADD VALUE (idempotent avec IF NOT EXISTS)
    # SQLite (dev/test) : l'enum est stocké comme VARCHAR — pas besoin de migration
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TYPE mediatype ADD VALUE IF NOT EXISTS 'document'")

    # ── Table document_analyses ───────────────────────────────────────────────
    op.create_table(
        "document_analyses",
        sa.Column("id",                  sa.Integer(),     nullable=False),
        sa.Column("case_id",             sa.Integer(),     nullable=False),
        sa.Column("media_file_id",       sa.Integer(),     nullable=False),
        sa.Column("status",              sa.String(32),    nullable=False, server_default="pending"),
        sa.Column("celery_task_id",      sa.String(256),   nullable=True),
        # Verdict
        sa.Column("verdict",             sa.String(64),    nullable=True),
        sa.Column("verdict_label",       sa.String(64),    nullable=True),
        sa.Column("final_score",         sa.Float(),       nullable=True),
        # Scores composantes
        sa.Column("score_ela",           sa.Float(),       nullable=True),
        sa.Column("score_clone",         sa.Float(),       nullable=True),
        sa.Column("score_metadata_doc",  sa.Float(),       nullable=True),
        sa.Column("score_font",          sa.Float(),       nullable=True),
        sa.Column("score_text_ai",       sa.Float(),       nullable=True),
        # XAI
        sa.Column("anomalies",           sa.JSON(),        nullable=True),
        sa.Column("xai_plain_explanation", sa.Text(),      nullable=True),
        sa.Column("text_preview",        sa.Text(),        nullable=True),
        sa.Column("models_used",         sa.JSON(),        nullable=True),
        # Audit
        sa.Column("requested_by_id",     sa.Integer(),     nullable=False),
        sa.Column("started_at",          sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at",        sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds",    sa.Integer(),     nullable=True),
        sa.Column("error_message",       sa.Text(),        nullable=True),
        # PK + FK
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["case_id"],         ["cases.id"]),
        sa.ForeignKeyConstraint(["media_file_id"],   ["media_files.id"]),
        sa.ForeignKeyConstraint(["requested_by_id"], ["users.id"]),
    )
    op.create_index("ix_document_analyses_case_id",       "document_analyses", ["case_id"])
    op.create_index("ix_document_analyses_media_file_id", "document_analyses", ["media_file_id"])


def downgrade() -> None:
    op.drop_index("ix_document_analyses_media_file_id", table_name="document_analyses")
    op.drop_index("ix_document_analyses_case_id",       table_name="document_analyses")
    op.drop_table("document_analyses")
    # Note : PostgreSQL ne supporte pas DROP VALUE sur les enums
    # La valeur 'document' reste dans l'enum après downgrade
