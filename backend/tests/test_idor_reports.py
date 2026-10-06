"""Tests IDOR sur les endpoints /reports.

Vérifie qu'un utilisateur ne peut pas accéder aux rapports d'un dossier
qui ne lui appartient pas.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from models.case import Case, CaseStatus, Jurisdiction
from models.analysis import Analysis, AnalysisStatus
from models.report import Report
from models.user import User


def _make_case(db: Session, owner: User, suffix: str) -> Case:
    case = Case(
        case_number=f"IDOR-RPT-{suffix}",
        title=f"Case {suffix}",
        jurisdiction=Jurisdiction.federal,
        status=CaseStatus.open,
        created_by_id=owner.id,
        retain_until=datetime.now(timezone.utc) + timedelta(days=3650),
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def _make_analysis(db: Session, case: Case, owner: User) -> Analysis:
    analysis = Analysis(
        case_id=case.id,
        media_file_id=1,
        status=AnalysisStatus.completed,
        requested_by_id=owner.id,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def _make_report(db: Session, case: Case, analysis: Analysis, expert: User, suffix: str) -> Report:
    report = Report(
        case_id=case.id,
        analysis_id=analysis.id,
        report_number=f"RPT-IDOR-{suffix}",
        expert_id=expert.id,
        expert_username=expert.username,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


class TestIDORReportsDownload:
    """GET /reports/{report_id}/download — accès refusé si le dossier appartient à autrui."""

    def test_owner_gets_own_report_not_found_without_pdf(
        self, client: TestClient, db: Session, admin_user: User, auth_admin: dict
    ):
        """Le propriétaire obtient 404 (pas de PDF sur disque) — pas 403."""
        case = _make_case(db, admin_user, "DL-01")
        analysis = _make_analysis(db, case, admin_user)
        report = _make_report(db, case, analysis, admin_user, "DL-01")

        resp = client.get(f"/reports/{report.id}/download", headers=auth_admin)
        assert resp.status_code == 404  # pas de fichier PDF, mais pas 403

    def test_other_analyst_cannot_download(
        self,
        client: TestClient,
        db: Session,
        admin_user: User,
        analyst_user: User,
        auth_analyst: dict,
    ):
        """Un analyst ne peut pas télécharger le rapport d'un dossier d'un autre utilisateur."""
        case = _make_case(db, admin_user, "DL-02")
        analysis = _make_analysis(db, case, admin_user)
        report = _make_report(db, case, analysis, admin_user, "DL-02")

        resp = client.get(f"/reports/{report.id}/download", headers=auth_analyst)
        assert resp.status_code in (403, 404)

    def test_unauthenticated_cannot_download(
        self, client: TestClient, db: Session, admin_user: User
    ):
        case = _make_case(db, admin_user, "DL-03")
        analysis = _make_analysis(db, case, admin_user)
        report = _make_report(db, case, analysis, admin_user, "DL-03")

        resp = client.get(f"/reports/{report.id}/download")
        assert resp.status_code == 401

    def test_nonexistent_report_returns_404(
        self, client: TestClient, auth_analyst: dict
    ):
        resp = client.get("/reports/99999/download", headers=auth_analyst)
        assert resp.status_code == 404


class TestIDORReportsList:
    """GET /reports/case/{case_id} — accès refusé si le dossier appartient à autrui."""

    def test_owner_can_list_own_case_reports(
        self, client: TestClient, db: Session, admin_user: User, auth_admin: dict
    ):
        case = _make_case(db, admin_user, "LS-01")
        analysis = _make_analysis(db, case, admin_user)
        _make_report(db, case, analysis, admin_user, "LS-01")

        resp = client.get(f"/reports/case/{case.id}", headers=auth_admin)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_other_analyst_cannot_list_reports(
        self,
        client: TestClient,
        db: Session,
        admin_user: User,
        analyst_user: User,
        auth_analyst: dict,
    ):
        """Un analyst ne peut pas lister les rapports d'un dossier qui ne lui appartient pas."""
        case = _make_case(db, admin_user, "LS-02")
        analysis = _make_analysis(db, case, admin_user)
        _make_report(db, case, analysis, admin_user, "LS-02")

        resp = client.get(f"/reports/case/{case.id}", headers=auth_analyst)
        assert resp.status_code in (403, 404)

    def test_unauthenticated_cannot_list_reports(
        self, client: TestClient, db: Session, admin_user: User
    ):
        case = _make_case(db, admin_user, "LS-03")
        resp = client.get(f"/reports/case/{case.id}")
        assert resp.status_code == 401

    def test_nonexistent_case_returns_empty_or_403(
        self, client: TestClient, auth_analyst: dict
    ):
        resp = client.get("/reports/case/99999", headers=auth_analyst)
        assert resp.status_code in (200, 403, 404)


class TestIDORReportsGenerate:
    """POST /reports/generate/{analysis_id} — accès refusé si l'analyse appartient à autrui."""

    def test_other_analyst_cannot_generate_report(
        self,
        client: TestClient,
        db: Session,
        admin_user: User,
        analyst_user: User,
        auth_analyst: dict,
    ):
        """Un analyst ne peut pas générer un rapport pour une analyse d'un autre utilisateur."""
        case = _make_case(db, admin_user, "GEN-01")
        analysis = _make_analysis(db, case, admin_user)

        resp = client.post(f"/reports/generate/{analysis.id}", headers=auth_analyst, json={})
        assert resp.status_code in (403, 404)

    def test_unauthenticated_cannot_generate(
        self, client: TestClient, db: Session, admin_user: User
    ):
        case = _make_case(db, admin_user, "GEN-02")
        analysis = _make_analysis(db, case, admin_user)

        resp = client.post(f"/reports/generate/{analysis.id}", json={})
        assert resp.status_code == 401

    def test_nonexistent_analysis_returns_404(
        self, client: TestClient, auth_analyst: dict
    ):
        resp = client.post("/reports/generate/99999", headers=auth_analyst, json={})
        assert resp.status_code == 404


class TestIDORCourtroom:
    """POST /reports/{analysis_id}/courtroom — accès refusé si l'analyse appartient à autrui."""

    def test_other_analyst_cannot_generate_courtroom_report(
        self,
        client: TestClient,
        db: Session,
        admin_user: User,
        analyst_user: User,
        auth_analyst: dict,
    ):
        case = _make_case(db, admin_user, "CT-01")
        analysis = _make_analysis(db, case, admin_user)

        resp = client.post(f"/reports/{analysis.id}/courtroom", headers=auth_analyst, json={})
        assert resp.status_code in (403, 404)

    def test_unauthenticated_cannot_generate_courtroom(
        self, client: TestClient, db: Session, admin_user: User
    ):
        case = _make_case(db, admin_user, "CT-02")
        analysis = _make_analysis(db, case, admin_user)

        resp = client.post(f"/reports/{analysis.id}/courtroom", json={})
        assert resp.status_code == 401
