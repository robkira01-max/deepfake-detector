"""phases_3_4_5

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-26 00:00:00.000000

v3.0/v3.1 — Ajoute :
  - Table training_protocols     (Phase 3 — Brief v3 §0)
  - Table model_versions         (Phase 3 — Model Registry)
  - Table analysis_feedbacks     (Phase 4 — Active Learning)
  - Table kyc_verifications      (v3.1 — KYC Engine)
  - Colonnes C2PA sur media_files (Phase 5 — Provenance)
  - Nouveaux enums PostgreSQL : validationstatus, protocolstatus,
                                 feedbacktype, trueverdict, kycverdict
  - Nouvelles valeurs AuditAction (v3.0 + v3.1)
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"

    # ── Nouveaux types enum PostgreSQL ────────────────────────────────────────
    if is_pg:
        op.execute(
            "DO $$ BEGIN "
            "  CREATE TYPE validationstatus AS ENUM "
            "    ('experimental','validated','disabled'); "
            "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        )
        op.execute(
            "DO $$ BEGIN "
            "  CREATE TYPE protocolstatus AS ENUM "
            "    ('draft','frozen','consumed'); "
            "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        )
        op.execute(
            "DO $$ BEGIN "
            "  CREATE TYPE feedbacktype AS ENUM "
            "    ('correct','incorrect','uncertain'); "
            "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        )
        op.execute(
            "DO $$ BEGIN "
            "  CREATE TYPE trueverdict AS ENUM "
            "    ('authentic','deepfake','indeterminate'); "
            "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        )
        op.execute(
            "DO $$ BEGIN "
            "  CREATE TYPE kycverdict AS ENUM "
            "    ('PASS','FAIL','REVIEW'); "
            "EXCEPTION WHEN duplicate_object THEN NULL; END $$"
        )

    # ── Table training_protocols ──────────────────────────────────────────────
    op.create_table(
        "training_protocols",
        sa.Column("id",                          sa.Integer(),   nullable=False),
        sa.Column("name",                        sa.String(256), nullable=False),
        sa.Column("engine_name",                 sa.String(128), nullable=False),
        sa.Column("test_set_definition",         sa.JSON(),      nullable=False),
        sa.Column("metrics_targets",             sa.JSON(),      nullable=False),
        sa.Column("notes",                       sa.Text(),      nullable=True),
        sa.Column("protocol_hash",               sa.String(64),  nullable=True),
        sa.Column(
            "status",
            sa.Enum("draft", "frozen", "consumed", name="protocolstatus"),
            nullable=False,
            server_default="draft",
        ),
        sa.Column("created_by_id",               sa.Integer(),   nullable=False),
        sa.Column("consumed_by_model_version_id", sa.Integer(),  nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()" if is_pg else "CURRENT_TIMESTAMP")),
        sa.Column("frozen_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"]),
    )
    op.create_index("ix_training_protocols_id",            "training_protocols", ["id"])
    op.create_index("ix_training_protocols_engine_name",   "training_protocols", ["engine_name"])
    op.create_index("ix_training_protocols_status",        "training_protocols", ["status"])
    op.create_index(
        "ix_training_protocols_engine_status",
        "training_protocols", ["engine_name", "status"],
    )

    # ── Table model_versions ──────────────────────────────────────────────────
    op.create_table(
        "model_versions",
        sa.Column("id",                    sa.Integer(),    nullable=False),
        sa.Column("model_name",            sa.String(128),  nullable=False),
        sa.Column("version",               sa.String(64),   nullable=False),
        sa.Column("hf_repo_id",            sa.String(256),  nullable=True),
        sa.Column("hf_revision",           sa.String(128),  nullable=True),
        sa.Column("local_path",            sa.String(512),  nullable=True),
        sa.Column("sha256",                sa.String(64),   nullable=True),
        sa.Column("metrics",               sa.JSON(),       nullable=True),
        sa.Column("metrics_source_hash",   sa.String(64),   nullable=True),
        sa.Column("metrics_source_path",   sa.String(512),  nullable=True),
        sa.Column(
            "validation_status",
            sa.Enum("experimental", "validated", "disabled", name="validationstatus"),
            nullable=False,
            server_default="experimental",
        ),
        sa.Column("training_protocol_id",  sa.Integer(),    nullable=True),
        sa.Column("is_active",             sa.Boolean(),    nullable=False, server_default=sa.text("false")),
        sa.Column("downloaded_at",         sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at",            sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()" if is_pg else "CURRENT_TIMESTAMP")),
        sa.Column("registered_by_id",      sa.Integer(),    nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["training_protocol_id"], ["training_protocols.id"]),
        sa.ForeignKeyConstraint(["registered_by_id"],     ["users.id"]),
    )
    op.create_index("ix_model_versions_id",             "model_versions", ["id"])
    op.create_index("ix_model_versions_model_name",     "model_versions", ["model_name"])
    op.create_index("ix_model_versions_validation_status", "model_versions", ["validation_status"])
    op.create_index("ix_model_versions_is_active",      "model_versions", ["is_active"])
    op.create_index(
        "ix_model_versions_name_active",
        "model_versions", ["model_name", "is_active"],
    )
    op.create_index(
        "ix_model_versions_name_status",
        "model_versions", ["model_name", "validation_status"],
    )

    # Rétro-FK training_protocols → model_versions (consumed_by_model_version_id)
    op.create_foreign_key(
        "fk_training_protocols_consumed_by",
        "training_protocols", "model_versions",
        ["consumed_by_model_version_id"], ["id"],
        use_alter=True,   # Évite la circularité pendant la création
    )

    # ── Table analysis_feedbacks ──────────────────────────────────────────────
    op.create_table(
        "analysis_feedbacks",
        sa.Column("id",                    sa.Integer(),   nullable=False),
        sa.Column("analysis_id",           sa.Integer(),   nullable=True),
        sa.Column("document_analysis_id",  sa.Integer(),   nullable=True),
        sa.Column(
            "feedback_type",
            sa.Enum("correct", "incorrect", "uncertain", name="feedbacktype"),
            nullable=False,
        ),
        sa.Column(
            "true_verdict",
            sa.Enum("authentic", "deepfake", "indeterminate", name="trueverdict"),
            nullable=True,
        ),
        sa.Column("analyst_notes",     sa.Text(),    nullable=True),
        sa.Column("confidence_rating", sa.Integer(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()" if is_pg else "CURRENT_TIMESTAMP")),
        sa.Column("submitted_by_id",   sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["analysis_id"],          ["analyses.id"]),
        sa.ForeignKeyConstraint(["document_analysis_id"], ["document_analyses.id"]),
        sa.ForeignKeyConstraint(["submitted_by_id"],      ["users.id"]),
        sa.CheckConstraint(
            "analysis_id IS NOT NULL OR document_analysis_id IS NOT NULL",
            name="ck_feedback_target_not_null",
        ),
    )
    op.create_index("ix_analysis_feedbacks_id",                   "analysis_feedbacks", ["id"])
    op.create_index("ix_analysis_feedbacks_analysis_id",          "analysis_feedbacks", ["analysis_id"])
    op.create_index("ix_analysis_feedbacks_document_analysis_id", "analysis_feedbacks", ["document_analysis_id"])
    op.create_index("ix_analysis_feedbacks_submitted_by_id",      "analysis_feedbacks", ["submitted_by_id"])

    # ── Table kyc_verifications ───────────────────────────────────────────────
    op.create_table(
        "kyc_verifications",
        sa.Column("id",                       sa.Integer(),  nullable=False),
        sa.Column("case_id",                  sa.Integer(),  nullable=True),
        sa.Column("doc_media_file_id",        sa.Integer(),  nullable=False),
        sa.Column("selfie_media_file_id",     sa.Integer(),  nullable=True),
        # Face matching
        sa.Column("face_match_score",         sa.Float(),    nullable=True),
        sa.Column("face_match_verdict",       sa.String(16), nullable=True),
        # OCR fields
        sa.Column("ocr_document_type",        sa.String(8),  nullable=True),
        sa.Column("ocr_country",              sa.String(8),  nullable=True),
        sa.Column("ocr_surname",              sa.String(128), nullable=True),
        sa.Column("ocr_given_names",          sa.String(128), nullable=True),
        sa.Column("ocr_document_number",      sa.String(32), nullable=True),
        sa.Column("ocr_birth_date",           sa.String(8),  nullable=True),
        sa.Column("ocr_expiry_date",          sa.String(8),  nullable=True),
        sa.Column("ocr_nationality",          sa.String(8),  nullable=True),
        sa.Column("ocr_mrz_valid",            sa.Boolean(),  nullable=True),
        sa.Column("ocr_raw_fields",           sa.JSON(),     nullable=True),
        # Document deepfake
        sa.Column("document_deepfake_score",   sa.Float(),   nullable=True),
        sa.Column("document_deepfake_verdict", sa.String(32), nullable=True),
        # Global verdict
        sa.Column(
            "kyc_verdict",
            sa.Enum("PASS", "FAIL", "REVIEW", name="kycverdict"),
            nullable=False,
            server_default="REVIEW",
        ),
        sa.Column("kyc_verdict_reasons",      sa.JSON(),     nullable=True),
        # Meta
        sa.Column("requested_by_id",          sa.Integer(),  nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("NOW()" if is_pg else "CURRENT_TIMESTAMP")),
        sa.Column("completed_at",             sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms",              sa.Integer(),  nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["case_id"],              ["cases.id"]),
        sa.ForeignKeyConstraint(["doc_media_file_id"],    ["media_files.id"]),
        sa.ForeignKeyConstraint(["selfie_media_file_id"], ["media_files.id"]),
        sa.ForeignKeyConstraint(["requested_by_id"],      ["users.id"]),
    )
    op.create_index("ix_kyc_verifications_id",      "kyc_verifications", ["id"])
    op.create_index("ix_kyc_verifications_case_id", "kyc_verifications", ["case_id"])

    # ── Colonnes C2PA sur media_files ─────────────────────────────────────────
    op.add_column("media_files",
        sa.Column("c2pa_status",       sa.String(16),  nullable=True))
    op.add_column("media_files",
        sa.Column("c2pa_producer",     sa.String(256), nullable=True))
    op.add_column("media_files",
        sa.Column("c2pa_manifest_json", sa.JSON(),     nullable=True))

    # ── Nouvelles valeurs AuditAction ─────────────────────────────────────────
    # PostgreSQL : ALTER TYPE ... ADD VALUE (idempotent)
    # SQLite : VARCHAR — aucune action requise
    if is_pg:
        new_audit_values = [
            "FEEDBACK_SUBMITTED",
            "MODEL_REGISTERED",
            "MODEL_PULLED",
            "MODEL_ACTIVATED",
            "RETRAIN_TRIGGERED",
            "KYC_VERIFIED",
            "PROTOCOL_CREATED",
            "PROTOCOL_FROZEN",
            "PROTOCOL_CONSUMED",
        ]
        for val in new_audit_values:
            op.execute(f"ALTER TYPE auditaction ADD VALUE IF NOT EXISTS '{val}'")


def downgrade() -> None:
    # Colonnes C2PA
    op.drop_column("media_files", "c2pa_manifest_json")
    op.drop_column("media_files", "c2pa_producer")
    op.drop_column("media_files", "c2pa_status")

    # Tables (ordre inverse des FK)
    op.drop_index("ix_kyc_verifications_case_id", table_name="kyc_verifications")
    op.drop_index("ix_kyc_verifications_id",      table_name="kyc_verifications")
    op.drop_table("kyc_verifications")

    op.drop_index("ix_analysis_feedbacks_submitted_by_id",      table_name="analysis_feedbacks")
    op.drop_index("ix_analysis_feedbacks_document_analysis_id", table_name="analysis_feedbacks")
    op.drop_index("ix_analysis_feedbacks_analysis_id",          table_name="analysis_feedbacks")
    op.drop_index("ix_analysis_feedbacks_id",                   table_name="analysis_feedbacks")
    op.drop_table("analysis_feedbacks")

    # Supprimer FK circulaire avant drop training_protocols
    op.drop_constraint("fk_training_protocols_consumed_by", "training_protocols", type_="foreignkey")

    op.drop_index("ix_model_versions_name_status",       table_name="model_versions")
    op.drop_index("ix_model_versions_name_active",       table_name="model_versions")
    op.drop_index("ix_model_versions_is_active",         table_name="model_versions")
    op.drop_index("ix_model_versions_validation_status", table_name="model_versions")
    op.drop_index("ix_model_versions_model_name",        table_name="model_versions")
    op.drop_index("ix_model_versions_id",                table_name="model_versions")
    op.drop_table("model_versions")

    op.drop_index("ix_training_protocols_engine_status", table_name="training_protocols")
    op.drop_index("ix_training_protocols_status",        table_name="training_protocols")
    op.drop_index("ix_training_protocols_engine_name",   table_name="training_protocols")
    op.drop_index("ix_training_protocols_id",            table_name="training_protocols")
    op.drop_table("training_protocols")

    # PostgreSQL : supprimer les types (après drop des tables qui les utilisent)
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP TYPE IF EXISTS kycverdict")
        op.execute("DROP TYPE IF EXISTS trueverdict")
        op.execute("DROP TYPE IF EXISTS feedbacktype")
        op.execute("DROP TYPE IF EXISTS protocolstatus")
        op.execute("DROP TYPE IF EXISTS validationstatus")
        # Note : les valeurs AuditAction ajoutées ne peuvent pas être supprimées sur PostgreSQL
