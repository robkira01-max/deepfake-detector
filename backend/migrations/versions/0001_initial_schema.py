"""initial_schema

Revision ID: 0001
Revises:
Create Date: 2026-09-22 00:00:00.000000

Crée les 6 tables du schéma DeepfakeDetector Canada :
  users, cases, media_files, audit_logs, analyses, reports

Toutes les contraintes FK, index, et valeurs par défaut sont incluses.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# ── Revision identifiers ──────────────────────────────────────────────────────
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── users ──────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",              sa.Integer(),      nullable=False),
        sa.Column("username",        sa.String(64),     nullable=False),
        sa.Column("email",           sa.String(128),    nullable=False),
        sa.Column("hashed_password", sa.String(256),    nullable=False),
        sa.Column("role",            sa.String(8),      nullable=False, server_default="readonly"),
        sa.Column("is_active",       sa.Boolean(),      nullable=False, server_default=sa.text("true")),
        sa.Column("mfa_secret",      sa.String(64),     nullable=True),
        sa.Column("mfa_enabled",     sa.Boolean(),      nullable=False, server_default=sa.text("false")),
        sa.Column("created_at",      sa.DateTime(),     nullable=False, server_default=sa.text("NOW()")),
        sa.Column("last_login",      sa.DateTime(),     nullable=True),
        sa.Column("login_count",     sa.Integer(),      nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_id",       "users", ["id"],       unique=False)
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email",    "users", ["email"],    unique=True)

    # ── cases ──────────────────────────────────────────────────────────────────
    op.create_table(
        "cases",
        sa.Column("id",             sa.Integer(),      nullable=False),
        sa.Column("case_number",    sa.String(64),     nullable=False),
        sa.Column("title",          sa.String(256),    nullable=False),
        sa.Column("description",    sa.Text(),         nullable=True),
        sa.Column("jurisdiction",   sa.String(16),     nullable=False, server_default="federal"),
        sa.Column("status",         sa.String(11),     nullable=False, server_default="open"),
        sa.Column("plaintiff",      sa.String(256),    nullable=True),
        sa.Column("defendant",      sa.String(256),    nullable=True),
        sa.Column("counsel",        sa.String(256),    nullable=True),
        sa.Column("created_by_id",  sa.Integer(),      nullable=False),
        sa.Column("created_at",     sa.DateTime(),     nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at",     sa.DateTime(),     nullable=False, server_default=sa.text("NOW()")),
        sa.Column("closed_at",      sa.DateTime(),     nullable=True),
        sa.Column("retain_until",   sa.DateTime(),     nullable=True),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_cases_id",          "cases", ["id"],          unique=False)
    op.create_index("ix_cases_case_number", "cases", ["case_number"], unique=True)

    # ── media_files ────────────────────────────────────────────────────────────
    op.create_table(
        "media_files",
        sa.Column("id",                sa.Integer(),        nullable=False),
        sa.Column("uuid",              sa.String(36),       nullable=False),
        sa.Column("case_id",           sa.Integer(),        nullable=False),
        sa.Column("original_filename", sa.String(512),      nullable=False),
        sa.Column("media_type",        sa.String(5),        nullable=False),
        sa.Column("mime_type",         sa.String(128),      nullable=False),
        sa.Column("file_size_bytes",   sa.BigInteger(),     nullable=False),
        sa.Column("status",            sa.String(10),       nullable=False, server_default="quarantine"),
        sa.Column("hash_sha256",       sa.String(64),       nullable=False),
        sa.Column("hash_blake3",       sa.String(64),       nullable=False),
        sa.Column("hash_md5",          sa.String(32),       nullable=False),
        sa.Column("tsa_token_b64",     sa.Text(),           nullable=True),
        sa.Column("tsa_authority",     sa.String(256),      nullable=True),
        sa.Column("tsa_timestamp",     sa.DateTime(),       nullable=True),
        sa.Column("storage_bucket",    sa.String(128),      nullable=True),
        sa.Column("storage_key",       sa.String(512),      nullable=True),
        sa.Column("media_metadata",    sa.JSON(),           nullable=True),
        sa.Column("ingested_at",       sa.DateTime(),       nullable=False, server_default=sa.text("NOW()")),
        sa.Column("ingested_by_id",    sa.Integer(),        nullable=False),
        sa.ForeignKeyConstraint(["case_id"],        ["cases.id"],  ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["ingested_by_id"], ["users.id"],  ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_media_files_id",        "media_files", ["id"],         unique=False)
    op.create_index("ix_media_files_uuid",      "media_files", ["uuid"],       unique=True)
    op.create_index("ix_media_files_hash_sha256","media_files",["hash_sha256"],unique=False)

    # ── audit_logs ─────────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id",             sa.Integer(),      nullable=False),
        sa.Column("user_id",        sa.Integer(),      nullable=True),
        sa.Column("user_username",  sa.String(64),     nullable=True),
        sa.Column("action",         sa.String(18),     nullable=False),
        sa.Column("resource_type",  sa.String(64),     nullable=True),
        sa.Column("resource_id",    sa.String(64),     nullable=True),
        sa.Column("details",        sa.JSON(),         nullable=True),
        sa.Column("ip_address",     sa.String(45),     nullable=True),
        sa.Column("user_agent",     sa.String(512),    nullable=True),
        sa.Column("entry_hash",     sa.String(64),     nullable=True),
        sa.Column("signature_b64",  sa.Text(),         nullable=True),
        sa.Column("timestamp",      sa.DateTime(),     nullable=False, server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_id",        "audit_logs", ["id"],        unique=False)
    op.create_index("ix_audit_logs_user_id",   "audit_logs", ["user_id"],   unique=False)
    op.create_index("ix_audit_logs_action",    "audit_logs", ["action"],    unique=False)
    op.create_index("ix_audit_logs_timestamp", "audit_logs", ["timestamp"], unique=False)

    # ── analyses ───────────────────────────────────────────────────────────────
    op.create_table(
        "analyses",
        sa.Column("id",                       sa.Integer(),   nullable=False),
        sa.Column("case_id",                  sa.Integer(),   nullable=False),
        sa.Column("media_file_id",            sa.Integer(),   nullable=False),
        sa.Column("status",                   sa.String(9),   nullable=False, server_default="pending"),
        sa.Column("celery_task_id",           sa.String(256), nullable=True),
        sa.Column("verdict",                  sa.String(12),  nullable=True),
        sa.Column("final_score",              sa.Float(),     nullable=True),
        sa.Column("confidence_low",           sa.Float(),     nullable=True),
        sa.Column("confidence_high",          sa.Float(),     nullable=True),
        sa.Column("score_video_texture",      sa.Float(),     nullable=True),
        sa.Column("score_video_temporal",     sa.Float(),     nullable=True),
        sa.Column("score_rppg",               sa.Float(),     nullable=True),
        sa.Column("score_biometrics",         sa.Float(),     nullable=True),
        sa.Column("score_audio_model",        sa.Float(),     nullable=True),
        sa.Column("score_audio_phase",        sa.Float(),     nullable=True),
        sa.Column("score_metadata",           sa.Float(),     nullable=True),
        sa.Column("model_far",                sa.Float(),     nullable=True),
        sa.Column("model_frr",                sa.Float(),     nullable=True),
        sa.Column("model_eer",                sa.Float(),     nullable=True),
        sa.Column("model_auc",                sa.Float(),     nullable=True),
        sa.Column("models_used",              sa.JSON(),      nullable=True),
        sa.Column("xai_shap_values",          sa.JSON(),      nullable=True),
        sa.Column("xai_heatmap_path",         sa.String(512), nullable=True),
        sa.Column("xai_spectrogram_path",     sa.String(512), nullable=True),
        sa.Column("xai_suspicious_timecodes", sa.JSON(),      nullable=True),
        sa.Column("xai_plain_explanation",    sa.Text(),      nullable=True),
        sa.Column("error_message",            sa.Text(),      nullable=True),
        sa.Column("duration_seconds",         sa.Integer(),   nullable=True),
        sa.Column("started_at",               sa.DateTime(),  nullable=False, server_default=sa.text("NOW()")),
        sa.Column("completed_at",             sa.DateTime(),  nullable=True),
        sa.Column("requested_by_id",          sa.Integer(),   nullable=False),
        sa.ForeignKeyConstraint(["case_id"],       ["cases.id"],       ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["media_file_id"], ["media_files.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["requested_by_id"],["users.id"],      ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analyses_id",            "analyses", ["id"],            unique=False)
    op.create_index("ix_analyses_case_id",        "analyses", ["case_id"],       unique=False)
    op.create_index("ix_analyses_media_file_id",  "analyses", ["media_file_id"], unique=False)

    # ── reports ────────────────────────────────────────────────────────────────
    op.create_table(
        "reports",
        sa.Column("id",                    sa.Integer(),   nullable=False),
        sa.Column("case_id",               sa.Integer(),   nullable=False),
        sa.Column("analysis_id",           sa.Integer(),   nullable=False),
        sa.Column("report_number",         sa.String(64),  nullable=False),
        sa.Column("pdf_path",              sa.String(512), nullable=True),
        sa.Column("storage_bucket",        sa.String(128), nullable=True),
        sa.Column("storage_key",           sa.String(512), nullable=True),
        sa.Column("report_hash_sha256",    sa.String(64),  nullable=True),
        sa.Column("is_signed",             sa.Boolean(),   nullable=False, server_default=sa.text("false")),
        sa.Column("signature_b64",         sa.Text(),      nullable=True),
        sa.Column("signer_cert_fingerprint",sa.String(128),nullable=True),
        sa.Column("tsa_token_b64",         sa.Text(),      nullable=True),
        sa.Column("tsa_timestamp",         sa.DateTime(),  nullable=True),
        sa.Column("expert_id",             sa.Integer(),   nullable=False),
        sa.Column("expert_username",       sa.String(64),  nullable=False),
        sa.Column("generated_at",          sa.DateTime(),  nullable=False, server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["case_id"],     ["cases.id"],    ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["expert_id"],   ["users.id"],    ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_reports_id",            "reports", ["id"],            unique=False)
    op.create_index("ix_reports_case_id",       "reports", ["case_id"],       unique=False)
    op.create_index("ix_reports_report_number", "reports", ["report_number"], unique=True)


def downgrade() -> None:
    # Drop in reverse FK order
    op.drop_table("reports")
    op.drop_table("analyses")
    op.drop_table("audit_logs")
    op.drop_table("media_files")
    op.drop_table("cases")
    op.drop_table("users")
