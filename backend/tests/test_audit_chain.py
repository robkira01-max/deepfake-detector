"""Tests du chaînage SHA-256 du journal d'audit."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base
from models.audit_log import AuditLog, AuditAction, verify_audit_chain, _compute_chain_hash


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


def _make_entry(action: AuditAction = AuditAction.USER_LOGIN, **kwargs) -> AuditLog:
    ts = kwargs.pop("timestamp", datetime.now(timezone.utc))
    return AuditLog(action=action, timestamp=ts, **kwargs)


class TestChainBuilding:
    def test_first_entry_has_no_previous_hash(self, db):
        entry = _make_entry()
        db.add(entry)
        db.commit()
        db.refresh(entry)

        assert entry.previous_entry_hash is None
        assert entry.entry_hash is not None
        assert len(entry.entry_hash) == 64

    def test_second_entry_links_to_first(self, db):
        e1 = _make_entry(AuditAction.USER_LOGIN)
        db.add(e1)
        db.commit()
        db.refresh(e1)

        e2 = _make_entry(AuditAction.CASE_CREATED)
        db.add(e2)
        db.commit()
        db.refresh(e2)

        assert e2.previous_entry_hash == e1.entry_hash

    def test_chain_of_three(self, db):
        entries = [_make_entry(AuditAction.USER_LOGIN) for _ in range(3)]
        for e in entries:
            db.add(e)
            db.commit()
            db.refresh(e)

        assert entries[0].previous_entry_hash is None
        assert entries[1].previous_entry_hash == entries[0].entry_hash
        assert entries[2].previous_entry_hash == entries[1].entry_hash

    def test_entry_hash_is_deterministic(self, db):
        ts = datetime(2026, 10, 4, 12, 0, 0, tzinfo=timezone.utc)
        entry = _make_entry(
            AuditAction.FILE_UPLOADED,
            timestamp=ts,
            user_id=42,
            user_username="alice",
            resource_type="media_file",
            resource_id="abc123",
            details={"filename": "test.mp4"},
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)

        expected = _compute_chain_hash(
            previous_hash=None,
            timestamp=ts,
            action=AuditAction.FILE_UPLOADED,
            user_id=42,
            user_username="alice",
            resource_type="media_file",
            resource_id="abc123",
            details={"filename": "test.mp4"},
        )
        assert entry.entry_hash == expected

    def test_details_order_irrelevant(self, db):
        ts = datetime(2026, 10, 4, 12, 0, 0, tzinfo=timezone.utc)
        h1 = _compute_chain_hash(None, ts, AuditAction.USER_LOGIN, None, None, None, None,
                                  {"b": 2, "a": 1})
        h2 = _compute_chain_hash(None, ts, AuditAction.USER_LOGIN, None, None, None, None,
                                  {"a": 1, "b": 2})
        assert h1 == h2  # json.dumps avec sort_keys=True

    def test_timestamp_set_automatically_if_missing(self, db):
        entry = AuditLog(action=AuditAction.USER_LOGIN)
        assert entry.timestamp is None
        db.add(entry)
        db.commit()
        db.refresh(entry)
        assert entry.timestamp is not None
        assert entry.entry_hash is not None


class TestVerifyChain:
    def test_empty_chain_is_valid(self, db):
        ok, errors = verify_audit_chain(db)
        assert ok is True
        assert errors == []

    def test_intact_chain_is_valid(self, db):
        for action in [AuditAction.USER_LOGIN, AuditAction.CASE_CREATED, AuditAction.FILE_UPLOADED]:
            db.add(_make_entry(action))
            db.commit()

        ok, errors = verify_audit_chain(db)
        assert ok is True
        assert errors == []

    def test_tampered_entry_hash_detected(self, db):
        e1 = _make_entry(AuditAction.USER_LOGIN)
        e2 = _make_entry(AuditAction.CASE_CREATED)
        db.add(e1)
        db.commit()
        db.add(e2)
        db.commit()
        db.refresh(e1)
        db.refresh(e2)

        # Altérer entry_hash de e1 directement en SQL (contourne le listener)
        from sqlalchemy import text
        db.execute(
            text("UPDATE audit_logs SET entry_hash = 'deadbeef' || substr(entry_hash, 9) WHERE id = :id"),
            {"id": e1.id},
        )
        db.commit()

        ok, errors = verify_audit_chain(db)
        assert ok is False
        assert any(str(e1.id) in err for err in errors)

    def test_broken_link_detected(self, db):
        e1 = _make_entry(AuditAction.USER_LOGIN)
        e2 = _make_entry(AuditAction.CASE_CREATED)
        db.add(e1)
        db.commit()
        db.add(e2)
        db.commit()
        db.refresh(e2)

        # Casser le lien de e2 en SQL
        from sqlalchemy import text
        db.execute(
            text("UPDATE audit_logs SET previous_entry_hash = 'fakehash' WHERE id = :id"),
            {"id": e2.id},
        )
        db.commit()

        ok, errors = verify_audit_chain(db)
        assert ok is False
        assert any("lien rompu" in err for err in errors)

    def test_multiple_tampering_all_reported(self, db):
        for _ in range(5):
            db.add(_make_entry(AuditAction.USER_LOGIN))
            db.commit()

        from sqlalchemy import text
        # Altérer les entrées 2 et 4
        rows = db.execute(text("SELECT id FROM audit_logs ORDER BY id")).fetchall()
        ids = [r[0] for r in rows]
        db.execute(
            text("UPDATE audit_logs SET entry_hash = 'aabbcc' || substr(entry_hash, 7) WHERE id = :id"),
            {"id": ids[1]},
        )
        db.execute(
            text("UPDATE audit_logs SET entry_hash = 'aabbcc' || substr(entry_hash, 7) WHERE id = :id"),
            {"id": ids[3]},
        )
        db.commit()

        ok, errors = verify_audit_chain(db)
        assert ok is False
        assert len(errors) >= 2
