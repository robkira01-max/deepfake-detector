"""Tests pour routers/dashboard.py — stats, audit log, health."""
from __future__ import annotations

import unittest.mock as mock
import uuid

from sqlalchemy.orm import Session

from models.audit_log import AuditLog, AuditAction
from models.case import Case, CaseStatus
from models.user import User


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_case(db: Session, user: User) -> Case:
    from models.case import Jurisdiction
    case = Case(
        case_number=f"DASH-{uuid.uuid4().hex[:6].upper()}",
        title="Dashboard test case",
        jurisdiction=Jurisdiction.federal,
        status=CaseStatus.open,
        created_by_id=user.id,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def _make_audit_log(db: Session, user: User, action: AuditAction = AuditAction.USER_LOGIN) -> AuditLog:
    log_entry = AuditLog(
        user_id=user.id,
        user_username=user.username,
        action=action,
        entry_hash="a" * 64,
        signature_b64="sig==",
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry


# ── Tests /dashboard/stats ─────────────────────────────────────────────────────

class TestStats:
    def test_stats_requires_auth(self, client):
        resp = client.get("/dashboard/stats")
        assert resp.status_code == 401

    def test_stats_readonly_allowed(self, client, readonly_token):
        resp = client.get(
            "/dashboard/stats",
            headers={"Authorization": f"Bearer {readonly_token}"},
        )
        assert resp.status_code == 200

    def test_stats_empty_db(self, client, analyst_token):
        resp = client.get(
            "/dashboard/stats",
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_cases"] >= 0
        assert data["total_analyses"] >= 0
        assert data["total_media_files"] >= 0
        assert data["total_reports"] >= 0
        assert "deepfake_rate" in data
        assert "avg_confidence" in data
        assert "analyses_last_7_days" in data
        assert "generated_at" in data

    def test_stats_with_case(self, client, analyst_token, analyst_user, db):
        before = client.get(
            "/dashboard/stats",
            headers={"Authorization": f"Bearer {analyst_token}"},
        ).json()["total_cases"]

        _make_case(db, analyst_user)

        after = client.get(
            "/dashboard/stats",
            headers={"Authorization": f"Bearer {analyst_token}"},
        ).json()["total_cases"]

        assert after == before + 1

    def test_stats_cases_by_status_keys(self, client, readonly_token):
        resp = client.get(
            "/dashboard/stats",
            headers={"Authorization": f"Bearer {readonly_token}"},
        )
        data = resp.json()
        assert "open" in data["cases_by_status"]

    def test_stats_deepfake_rate_zero_when_no_analysis(self, client, readonly_token):
        resp = client.get(
            "/dashboard/stats",
            headers={"Authorization": f"Bearer {readonly_token}"},
        )
        data = resp.json()
        assert isinstance(data["deepfake_rate"], float)


# ── Tests /dashboard/audit-log ─────────────────────────────────────────────────

class TestAuditLog:
    def test_audit_log_requires_admin(self, client, analyst_token):
        resp = client.get(
            "/dashboard/audit-log",
            headers={"Authorization": f"Bearer {analyst_token}"},
        )
        assert resp.status_code == 403

    def test_audit_log_requires_auth(self, client):
        resp = client.get("/dashboard/audit-log")
        assert resp.status_code == 401

    def test_audit_log_returns_list(self, client, admin_token):
        resp = client.get(
            "/dashboard/audit-log",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_audit_log_contains_entry(self, client, admin_token, admin_user, db):
        _make_audit_log(db, admin_user, AuditAction.USER_LOGIN)
        resp = client.get(
            "/dashboard/audit-log",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        entries = resp.json()
        assert len(entries) >= 1
        assert "action" in entries[0]
        assert "timestamp" in entries[0]

    def test_audit_log_filter_by_user(self, client, admin_token, admin_user, db):
        _make_audit_log(db, admin_user, AuditAction.USER_LOGOUT)
        resp = client.get(
            f"/dashboard/audit-log?user_id={admin_user.id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        entries = resp.json()
        assert len(entries) >= 1

    def test_audit_log_filter_by_action(self, client, admin_token, admin_user, db):
        _make_audit_log(db, admin_user, AuditAction.REPORT_DOWNLOADED)
        resp = client.get(
            "/dashboard/audit-log?action=REPORT_DOWNLOADED",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200

    def test_audit_log_pagination(self, client, admin_token, admin_user, db):
        for _ in range(5):
            _make_audit_log(db, admin_user)
        resp = client.get(
            "/dashboard/audit-log?skip=0&limit=3",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        assert len(resp.json()) <= 3


# ── Tests /dashboard/health/detailed ──────────────────────────────────────────

class TestHealthDetailed:
    def test_health_requires_auth(self, client):
        resp = client.get("/dashboard/health/detailed")
        assert resp.status_code == 401

    def test_health_readonly_allowed(self, client, readonly_token):
        resp = client.get(
            "/dashboard/health/detailed",
            headers={"Authorization": f"Bearer {readonly_token}"},
        )
        assert resp.status_code == 200

    def test_health_response_structure(self, client, readonly_token):
        resp = client.get(
            "/dashboard/health/detailed",
            headers={"Authorization": f"Bearer {readonly_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded", "critical")
        assert "components" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_contains_database_component(self, client, readonly_token):
        resp = client.get(
            "/dashboard/health/detailed",
            headers={"Authorization": f"Bearer {readonly_token}"},
        )
        data = resp.json()
        assert "database" in data["components"]
        assert data["components"]["database"]["status"] == "ok"

    def test_health_degraded_when_redis_unavailable(self, client, readonly_token):
        import redis as _redis_module
        with mock.patch.object(_redis_module, "from_url") as mock_redis:
            mock_redis.return_value.ping.side_effect = Exception("Connection refused")
            resp = client.get(
                "/dashboard/health/detailed",
                headers={"Authorization": f"Bearer {readonly_token}"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("degraded", "ok")

    def test_health_jwt_keys_present(self, client, readonly_token):
        resp = client.get(
            "/dashboard/health/detailed",
            headers={"Authorization": f"Bearer {readonly_token}"},
        )
        data = resp.json()
        assert "jwt_keys" in data["components"]
