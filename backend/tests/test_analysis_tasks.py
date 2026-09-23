"""Tests pour tasks/analysis_tasks.py — pipeline Celery sans broker."""
from __future__ import annotations

import uuid
import unittest.mock as mock
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from models.analysis import Analysis, AnalysisStatus, Verdict
from models.case import Case, CaseStatus
from models.media_file import MediaFile, MediaType, MediaStatus
from models.user import User


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_case(db: Session, user: User) -> Case:
    from models.case import Jurisdiction
    case = Case(
        case_number=f"TASK-{uuid.uuid4().hex[:6].upper()}",
        title="Task test case",
        jurisdiction=Jurisdiction.federal,
        status=CaseStatus.open,
        created_by_id=user.id,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def _make_media(
    db: Session, case: Case, user: User,
    media_type: MediaType = MediaType.video,
    with_storage_key: bool = True,
) -> MediaFile:
    mf = MediaFile(
        uuid=str(uuid.uuid4()),
        case_id=case.id,
        original_filename="test_evidence.mp4",
        media_type=media_type,
        mime_type="video/mp4",
        file_size_bytes=4096,
        status=MediaStatus.quarantine,
        hash_sha256="a" * 64,
        hash_blake3="b" * 64,
        hash_md5="c" * 32,
        ingested_by_id=user.id,
        storage_key="/tmp/test_evidence.mp4" if with_storage_key else None,
    )
    db.add(mf)
    db.commit()
    db.refresh(mf)
    return mf


def _make_analysis(db: Session, case: Case, media: MediaFile, user: User) -> Analysis:
    analysis = Analysis(
        case_id=case.id,
        media_file_id=media.id,
        status=AnalysisStatus.pending,
        requested_by_id=user.id,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def _call_task(task_fn, analysis_id: int, media_file_id: int):
    """Appelle la tâche Celery de manière synchrone via apply().

    `apply()` exécute la tâche dans le processus courant sans broker.
    Avec `bind=True`, Celery injecte automatiquement `self` = l'instance de tâche.
    """
    return task_fn.apply(args=[analysis_id, media_file_id])


def _make_mock_video_result() -> mock.MagicMock:
    r = mock.MagicMock()
    r.score_texture = 0.80
    r.score_temporal = 0.75
    r.score_rppg = 0.60
    r.score_biometrics = 0.70
    r.suspicious_timecodes = [1.5, 3.2]
    r.heatmap_path = None
    r.models_used = {"texture": "efficientnet-b4", "temporal": "resnet50-lstm"}
    return r


def _make_mock_audio_result() -> mock.MagicMock:
    r = mock.MagicMock()
    r.score_model = 0.85
    r.score_phase = 0.60
    r.spectrogram_path = None
    r.models_used = {"audio": "wav2vec2"}
    return r


def _make_mock_fusion_result() -> mock.MagicMock:
    r = mock.MagicMock()
    r.final_score = 0.78
    r.verdict = Verdict.deepfake
    r.confidence_low = 0.70
    r.confidence_high = 0.86
    r.component_scores = {
        "texture": 0.80, "temporal": 0.75, "rppg": 0.60,
        "biometrics": 0.70, "audio": 0.85, "phase": 0.60, "metadata": 0.50,
    }
    r.model_far = 0.05
    r.model_frr = 0.08
    r.model_eer = 0.065
    r.model_auc = 0.94
    r.shap_ranking = [{"feature": "audio", "weighted_score": 0.85}]
    r.plain_explanation = "Deepfake détecté avec haute confiance."
    return r


def _session_factory_that_returns(session: Session):
    """Retourne une SessionLocal mockée qui donne la session de test sans la fermer.

    Returns: (mock_factory, mock_session) - mock_session wraps the real session
    but intercepts close() pour ne pas fermer la session de test.
    """
    mock_session = mock.MagicMock(wraps=session)
    mock_session.close = mock.MagicMock()
    return mock.MagicMock(return_value=mock_session), mock_session


# ── Tests : chemins de succès ──────────────────────────────────────────────────

class TestRunDeepfakeAnalysisSuccess:
    def test_analysis_not_found_returns_error(self, db):
        """Analysis introuvable → retourne {"error": "analysis_not_found"}."""
        from tasks.analysis_tasks import run_deepfake_analysis

        mock_sl = mock.MagicMock()
        mock_inner_session = mock.MagicMock()
        mock_inner_session.query.return_value.filter.return_value.first.return_value = None
        mock_sl.return_value = mock_inner_session

        with mock.patch("database.SessionLocal", mock_sl):
            cel_result = _call_task(run_deepfake_analysis, 99999, 99999)

        result = cel_result.result
        assert result == {"error": "analysis_not_found"}

    def test_success_video_file(self, db, analyst_user):
        """Pipeline complet sur un fichier vidéo."""
        from tasks.analysis_tasks import run_deepfake_analysis
        case = _make_case(db, analyst_user)
        media = _make_media(db, case, analyst_user, MediaType.video)
        analysis = _make_analysis(db, case, media, analyst_user)

        mock_sl, _ = _session_factory_that_returns(db)
        video_result = _make_mock_video_result()
        audio_result = _make_mock_audio_result()
        fusion_result = _make_mock_fusion_result()

        with mock.patch("database.SessionLocal", mock_sl), \
             mock.patch("engines.video_engine.VideoEngine.analyze", return_value=video_result), \
             mock.patch("engines.audio_engine.AudioEngine.analyze", return_value=audio_result), \
             mock.patch("engines.metadata_engine.MetadataEngine.analyze") as mock_meta, \
             mock.patch("engines.fusion.fuse_scores", return_value=fusion_result), \
             mock.patch("core.chain_of_custody.sign_audit_entry", return_value=("fakehash64", "fakesig==")):

            mock_meta_result = mock.MagicMock()
            mock_meta_result.score = 0.5
            mock_meta_result.error = None
            mock_meta.return_value = mock_meta_result

            cel_result = _call_task(run_deepfake_analysis, analysis.id, media.id)

        result = cel_result.result
        assert result["analysis_id"] == analysis.id
        assert result["verdict"] == Verdict.deepfake.value
        assert result["final_score"] == pytest.approx(0.78)
        assert "duration_seconds" in result

    def test_success_audio_only_file(self, db, analyst_user):
        """Pipeline sur un fichier audio (VideoEngine non invoqué)."""
        from tasks.analysis_tasks import run_deepfake_analysis
        case = _make_case(db, analyst_user)
        media = _make_media(db, case, analyst_user, MediaType.audio)
        analysis = _make_analysis(db, case, media, analyst_user)

        mock_sl, _ = _session_factory_that_returns(db)
        audio_result = _make_mock_audio_result()
        fusion_result = _make_mock_fusion_result()

        mock_video_analyze = mock.MagicMock()

        with mock.patch("database.SessionLocal", mock_sl), \
             mock.patch("engines.video_engine.VideoEngine.analyze", mock_video_analyze), \
             mock.patch("engines.audio_engine.AudioEngine.analyze", return_value=audio_result), \
             mock.patch("engines.metadata_engine.MetadataEngine.analyze") as mock_meta, \
             mock.patch("engines.fusion.fuse_scores", return_value=fusion_result), \
             mock.patch("core.chain_of_custody.sign_audit_entry", return_value=("fakehash64", "fakesig==")):

            mock_meta_result = mock.MagicMock()
            mock_meta_result.score = 0.3
            mock_meta_result.error = None
            mock_meta.return_value = mock_meta_result

            cel_result = _call_task(run_deepfake_analysis, analysis.id, media.id)

        # VideoEngine ne doit pas avoir été appelé pour un fichier audio
        mock_video_analyze.assert_not_called()
        assert cel_result.result["analysis_id"] == analysis.id

    def test_analysis_status_set_to_running(self, db, analyst_user):
        """Vérifie que le statut passe à 'running' avant les moteurs."""
        from tasks.analysis_tasks import run_deepfake_analysis
        case = _make_case(db, analyst_user)
        media = _make_media(db, case, analyst_user)
        analysis = _make_analysis(db, case, media, analyst_user)
        assert analysis.status == AnalysisStatus.pending

        status_during_run = []
        fusion_result = _make_mock_fusion_result()
        audio_result = _make_mock_audio_result()

        def capture_status(file_path):
            db.refresh(analysis)
            status_during_run.append(analysis.status)
            return _make_mock_video_result()

        mock_sl, _ = _session_factory_that_returns(db)

        with mock.patch("database.SessionLocal", mock_sl), \
             mock.patch("engines.video_engine.VideoEngine.analyze", side_effect=capture_status), \
             mock.patch("engines.audio_engine.AudioEngine.analyze", return_value=audio_result), \
             mock.patch("engines.metadata_engine.MetadataEngine.analyze") as mock_meta, \
             mock.patch("engines.fusion.fuse_scores", return_value=fusion_result), \
             mock.patch("core.chain_of_custody.sign_audit_entry", return_value=("fakehash64", "fakesig==")):

            mock_meta_result = mock.MagicMock()
            mock_meta_result.score = 0.5
            mock_meta_result.error = None
            mock_meta.return_value = mock_meta_result

            _call_task(run_deepfake_analysis, analysis.id, media.id)

        assert AnalysisStatus.running in status_during_run

    def test_analysis_status_set_to_completed(self, db, analyst_user):
        """Après succès, le statut est 'completed'."""
        from tasks.analysis_tasks import run_deepfake_analysis
        case = _make_case(db, analyst_user)
        media = _make_media(db, case, analyst_user)
        analysis = _make_analysis(db, case, media, analyst_user)

        mock_sl, _ = _session_factory_that_returns(db)
        fusion_result = _make_mock_fusion_result()
        audio_result = _make_mock_audio_result()

        with mock.patch("database.SessionLocal", mock_sl), \
             mock.patch("engines.video_engine.VideoEngine.analyze", return_value=_make_mock_video_result()), \
             mock.patch("engines.audio_engine.AudioEngine.analyze", return_value=audio_result), \
             mock.patch("engines.metadata_engine.MetadataEngine.analyze") as mock_meta, \
             mock.patch("engines.fusion.fuse_scores", return_value=fusion_result), \
             mock.patch("core.chain_of_custody.sign_audit_entry", return_value=("fakehash64", "fakesig==")):

            mock_meta_result = mock.MagicMock()
            mock_meta_result.score = 0.5
            mock_meta_result.error = None
            mock_meta.return_value = mock_meta_result

            _call_task(run_deepfake_analysis, analysis.id, media.id)

        db.refresh(analysis)
        assert analysis.status == AnalysisStatus.completed
        assert analysis.verdict == Verdict.deepfake
        assert analysis.final_score == pytest.approx(0.78)


# ── Tests : chemins d'erreur ──────────────────────────────────────────────────

class TestRunDeepfakeAnalysisErrors:
    def test_media_file_missing_storage_key_sets_failed(self, db, analyst_user):
        """Fichier sans storage_key → status=failed, retry appelé."""
        from tasks.analysis_tasks import run_deepfake_analysis
        case = _make_case(db, analyst_user)
        media = _make_media(db, case, analyst_user, with_storage_key=False)
        analysis = _make_analysis(db, case, media, analyst_user)

        mock_sl, _ = _session_factory_that_returns(db)

        with mock.patch("database.SessionLocal", mock_sl), \
             mock.patch("core.chain_of_custody.sign_audit_entry", return_value=("h", "s")), \
             mock.patch.object(run_deepfake_analysis, "retry", side_effect=Exception("MaxRetry")):
            try:
                _call_task(run_deepfake_analysis, analysis.id, media.id)
            except Exception:
                pass

        db.refresh(analysis)
        assert analysis.status == AnalysisStatus.failed
        assert analysis.error_message is not None

    def test_engine_exception_sets_failed(self, db, analyst_user):
        """Si VideoEngine lève, status=failed."""
        from tasks.analysis_tasks import run_deepfake_analysis
        case = _make_case(db, analyst_user)
        media = _make_media(db, case, analyst_user)
        analysis = _make_analysis(db, case, media, analyst_user)

        mock_sl, _ = _session_factory_that_returns(db)

        with mock.patch("database.SessionLocal", mock_sl), \
             mock.patch("engines.video_engine.VideoEngine.analyze",
                        side_effect=RuntimeError("GPU OOM")), \
             mock.patch("core.chain_of_custody.sign_audit_entry", return_value=("h", "s")), \
             mock.patch.object(run_deepfake_analysis, "retry", side_effect=Exception("MaxRetry")):
            try:
                _call_task(run_deepfake_analysis, analysis.id, media.id)
            except Exception:
                pass

        db.refresh(analysis)
        assert analysis.status == AnalysisStatus.failed
        assert "GPU OOM" in (analysis.error_message or "")

    def test_metadata_error_does_not_abort(self, db, analyst_user):
        """MetadataEngine avec error != None → score=0.0, pipeline continue."""
        from tasks.analysis_tasks import run_deepfake_analysis
        case = _make_case(db, analyst_user)
        media = _make_media(db, case, analyst_user)
        analysis = _make_analysis(db, case, media, analyst_user)

        mock_sl, _ = _session_factory_that_returns(db)
        fusion_result = _make_mock_fusion_result()
        audio_result = _make_mock_audio_result()

        with mock.patch("database.SessionLocal", mock_sl), \
             mock.patch("engines.video_engine.VideoEngine.analyze", return_value=_make_mock_video_result()), \
             mock.patch("engines.audio_engine.AudioEngine.analyze", return_value=audio_result), \
             mock.patch("engines.metadata_engine.MetadataEngine.analyze") as mock_meta, \
             mock.patch("engines.fusion.fuse_scores", return_value=fusion_result), \
             mock.patch("core.chain_of_custody.sign_audit_entry", return_value=("fakehash", "fakesig")):

            meta_err_result = mock.MagicMock()
            meta_err_result.score = 0.0
            meta_err_result.error = "ffprobe not found"
            mock_meta.return_value = meta_err_result

            cel_result = _call_task(run_deepfake_analysis, analysis.id, media.id)

        assert cel_result.result is not None
        assert cel_result.result.get("analysis_id") == analysis.id


# ── Tests : fusion de scores ──────────────────────────────────────────────────

class TestFuseScores:
    """Tests directs sur la fonction fuse_scores (engines/fusion.py)."""

    def test_fuse_all_zeros(self):
        from engines.fusion import fuse_scores
        result = fuse_scores(
            score_texture=0.0, score_temporal=0.0, score_rppg=0.0,
            score_biometrics=0.0, score_audio=0.0, score_phase=0.0,
            score_metadata=0.0,
        )
        assert result.final_score == pytest.approx(0.0, abs=0.05)
        assert result.verdict is not None

    def test_fuse_all_ones_returns_deepfake(self):
        from engines.fusion import fuse_scores
        result = fuse_scores(
            score_texture=1.0, score_temporal=1.0, score_rppg=1.0,
            score_biometrics=1.0, score_audio=1.0, score_phase=1.0,
            score_metadata=1.0,
        )
        assert result.final_score > 0.8
        assert result.verdict == Verdict.deepfake

    def test_fuse_all_zero_returns_authentic(self):
        from engines.fusion import fuse_scores
        result = fuse_scores(
            score_texture=0.0, score_temporal=0.0, score_rppg=0.0,
            score_biometrics=0.0, score_audio=0.0, score_phase=0.0,
            score_metadata=0.0,
        )
        assert result.final_score < 0.4

    def test_fuse_returns_confidence_interval(self):
        from engines.fusion import fuse_scores
        result = fuse_scores(
            score_texture=0.5, score_temporal=0.5, score_rppg=0.5,
            score_biometrics=0.5, score_audio=0.5, score_phase=0.5,
            score_metadata=0.5,
        )
        assert result.confidence_low <= result.final_score
        assert result.final_score <= result.confidence_high

    def test_fuse_component_scores_present(self):
        from engines.fusion import fuse_scores
        result = fuse_scores(
            score_texture=0.6, score_temporal=0.4, score_rppg=0.3,
            score_biometrics=0.5, score_audio=0.7, score_phase=0.2,
            score_metadata=0.1,
        )
        assert "texture" in result.component_scores
        assert "audio" in result.component_scores
        assert "metadata" in result.component_scores

    def test_fuse_shap_ranking_sorted(self):
        from engines.fusion import fuse_scores
        result = fuse_scores(
            score_texture=0.1, score_temporal=0.1, score_rppg=0.1,
            score_biometrics=0.1, score_audio=0.9, score_phase=0.1,
            score_metadata=0.1,
        )
        if result.shap_ranking:
            contributions = [s["contribution"] for s in result.shap_ranking]
            assert contributions == sorted(contributions, reverse=True)

    def test_fuse_plain_explanation_not_empty(self):
        from engines.fusion import fuse_scores
        result = fuse_scores(
            score_texture=0.8, score_temporal=0.8, score_rppg=0.8,
            score_biometrics=0.8, score_audio=0.8, score_phase=0.8,
            score_metadata=0.8,
        )
        assert result.plain_explanation != ""
        assert len(result.plain_explanation) > 10

    def test_fuse_score_clamped_between_0_and_1(self):
        from engines.fusion import fuse_scores
        result = fuse_scores(
            score_texture=2.0, score_temporal=-0.5, score_rppg=1.5,
            score_biometrics=0.0, score_audio=0.5, score_phase=0.5,
            score_metadata=0.5,
        )
        assert 0.0 <= result.final_score <= 1.0


# ── Tests : MetadataEngine ────────────────────────────────────────────────────

class TestMetadataEngine:
    """Tests sur engines/metadata_engine.py — couverture des branches principales."""

    def test_file_not_found(self, tmp_path):
        from engines.metadata_engine import MetadataEngine
        engine = MetadataEngine()
        result = engine.analyze(tmp_path / "nonexistent.mp4")
        assert result.error is not None
        assert "introuvable" in result.error.lower()

    def test_analyze_with_ffprobe_mock(self, tmp_path):
        """Simule ffprobe retournant des métadonnées propres."""
        from engines.metadata_engine import MetadataEngine
        import json
        fake_video = tmp_path / "clean.mp4"
        fake_video.write_bytes(b"\x00" * 100)

        ffprobe_output = json.dumps({
            "format": {"duration": "60.0", "bit_rate": "1000000", "tags": {}},
            "streams": [{"codec_type": "video", "nb_frames": "1500", "duration": "60.0", "r_frame_rate": "25/1", "tags": {}}],
        })

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ffprobe_output
            engine = MetadataEngine()
            result = engine.analyze(fake_video)

        assert result.error is None
        assert 0.0 <= result.score <= 1.0

    def test_analyze_deepfake_tool_detected(self, tmp_path):
        """Détection de 'DeepFaceLab' dans les métadonnées."""
        from engines.metadata_engine import MetadataEngine
        import json
        fake_video = tmp_path / "deepfake.mp4"
        fake_video.write_bytes(b"\x00" * 100)

        ffprobe_output = json.dumps({
            "format": {
                "duration": "10.0",
                "bit_rate": "500000",
                "tags": {"encoder": "DeepFaceLab v3.0", "title": "generated"},
            },
            "streams": [],
        })

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ffprobe_output
            engine = MetadataEngine()
            result = engine.analyze(fake_video)

        assert result.score > 0.3
        assert result.tool_detected is not None
        finding_types = [f["type"] for f in result.findings]
        assert "deepfake_tool_signature" in finding_types

    def test_analyze_legitimate_tool(self, tmp_path):
        """Adobe Premiere dans les tags → score faible."""
        from engines.metadata_engine import MetadataEngine
        import json
        fake_video = tmp_path / "legit.mp4"
        fake_video.write_bytes(b"\x00" * 100)

        ffprobe_output = json.dumps({
            "format": {"duration": "30.0", "bit_rate": "2000000", "tags": {"encoder": "Adobe Premiere Pro CC"}},
            "streams": [],
        })

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ffprobe_output
            engine = MetadataEngine()
            result = engine.analyze(fake_video)

        assert result.score < 0.3

    def test_analyze_fps_mismatch(self, tmp_path):
        """FPS déclaré ≠ FPS réel → finding fps_mismatch."""
        from engines.metadata_engine import MetadataEngine
        import json
        fake_video = tmp_path / "fps_mismatch.mp4"
        fake_video.write_bytes(b"\x00" * 100)

        ffprobe_output = json.dumps({
            "format": {"duration": "60.0", "bit_rate": "1000000", "tags": {}},
            "streams": [{
                "codec_type": "video",
                "nb_frames": "3000",  # → 50 fps réel
                "duration": "60.0",
                "r_frame_rate": "25/1",  # 25 fps déclaré — écart >10%
                "tags": {},
            }],
        })

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ffprobe_output
            engine = MetadataEngine()
            result = engine.analyze(fake_video)

        finding_types = [f["type"] for f in result.findings]
        assert "fps_mismatch" in finding_types

    def test_analyze_ffprobe_not_available_fallback(self, tmp_path):
        """ffprobe absent → fallback lecture binaire."""
        from engines.metadata_engine import MetadataEngine
        fake_video = tmp_path / "noffprobe.mp4"
        fake_video.write_bytes(b"ftyp isom " + b"\x00" * 100)

        with mock.patch("subprocess.run", side_effect=FileNotFoundError("ffprobe not found")):
            engine = MetadataEngine()
            result = engine.analyze(fake_video)

        # Avec le fallback, on peut ne pas avoir d'erreur mais score minimal
        assert result is not None

    def test_flatten_to_text(self):
        """_flatten_to_text récursif."""
        from engines.metadata_engine import MetadataEngine
        engine = MetadataEngine()
        obj = {"key": {"nested": "DeepFaceLab"}, "list": ["wav2lip", 42]}
        text = engine._flatten_to_text(obj)
        assert "DeepFaceLab" in text
        assert "wav2lip" in text

    def test_temporal_anomaly_future_date(self, tmp_path):
        """Date de création dans le futur (2099) → temporal_anomaly."""
        from engines.metadata_engine import MetadataEngine
        import json
        fake_video = tmp_path / "future.mp4"
        fake_video.write_bytes(b"\x00" * 100)

        ffprobe_output = json.dumps({
            "format": {"duration": "10.0", "bit_rate": "500000", "tags": {"creation_time": "2099-01-01T00:00:00Z"}},
            "streams": [],
        })

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ffprobe_output
            engine = MetadataEngine()
            result = engine.analyze(fake_video)

        assert result.temporal_anomaly is True
        temporal_types = [f["type"] for f in result.findings]
        assert "temporal_anomaly" in temporal_types
