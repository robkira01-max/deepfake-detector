"""Tests pour training/eval_metadata_engine.py — Règle 10, condition 1."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "training"))
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

from eval_metadata_engine import (  # noqa: E402
    TEST_CASES,
    THRESHOLD,
    _compute_auc,
    _compute_eer,
    _compute_metrics_at_threshold,
    _run_engine_on_case,
    build_metrics_report,
    evaluate,
)


# ── Intégrité du jeu de test ──────────────────────────────────────────────────

def test_test_cases_count():
    assert len(TEST_CASES) == 100

def test_test_cases_balanced():
    assert sum(c.label for c in TEST_CASES) == 50
    assert sum(1 - c.label for c in TEST_CASES) == 50

def test_test_cases_descriptions_unique():
    descs = [c.description for c in TEST_CASES]
    assert len(descs) == len(set(descs)), "Descriptions dupliquées détectées"

def test_test_cases_labels_binary():
    for c in TEST_CASES:
        assert c.label in (0, 1)

def test_deepfake_cases_have_detectable_signals():
    deepfakes = [c for c in TEST_CASES if c.label == 1]
    # Les Groupes A-E (30 cas) contiennent des signatures d'outils → attendu détectable
    with_tool_sig = [
        c for c in deepfakes
        if any(
            kw in str(c.ffprobe_output).lower()
            for kw in ["deepfacelab", "facefusion", "roop", "simswap", "faceswap",
                       "wav2lip", "sadtalker", "elevenlabs", "xtts", "gfpgan",
                       "codeformer", "real-esrgan", "diffusion", "insightface",
                       "tortoise", "codeformer"]
        )
    ]
    assert len(with_tool_sig) >= 25


# ── Métriques scalaires ───────────────────────────────────────────────────────

def test_auc_perfect_classifier():
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = [1,   1,   0,   0]
    assert _compute_auc(scores, labels) == pytest.approx(1.0)

def test_auc_random_classifier():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1,   0,   1,   0]
    auc = _compute_auc(scores, labels)
    assert 0.0 <= auc <= 1.0

def test_auc_no_positive():
    assert _compute_auc([0.1, 0.2], [0, 0]) == 0.0

def test_auc_no_negative():
    assert _compute_auc([0.1, 0.2], [1, 1]) == 0.0

def test_eer_perfect_separation():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1,   1,   0,   0]
    eer, _ = _compute_eer(scores, labels)
    assert eer < 0.1

def test_eer_range():
    scores = [0.4, 0.6, 0.3, 0.7]
    labels = [1,   1,   0,   0]
    eer, t = _compute_eer(scores, labels)
    assert 0.0 <= eer <= 1.0
    assert 0.0 <= t <= 1.0

def test_metrics_at_threshold_perfect():
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = [1,   1,   0,   0]
    m = _compute_metrics_at_threshold(scores, labels, 0.5)
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["tn"] == 2
    assert m["fn"] == 0
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["far"] == pytest.approx(0.0)
    assert m["frr"] == pytest.approx(0.0)

def test_metrics_at_threshold_no_tp():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [1,   1,   0,   0]
    m = _compute_metrics_at_threshold(scores, labels, 0.5)
    assert m["tp"] == 0
    assert m["fn"] == 2
    assert m["frr"] == pytest.approx(1.0)


# ── Exécution sur le jeu de test réel ─────────────────────────────────────────

def test_evaluate_returns_correct_length():
    scores, labels, descs = evaluate()
    assert len(scores) == len(labels) == len(descs) == 100

def test_evaluate_all_scores_in_range():
    scores, _, _ = evaluate()
    for s in scores:
        assert 0.0 <= s <= 1.0, f"Score hors bornes : {s}"

def test_evaluate_no_false_positives():
    """FAR=0 : aucun fichier genuine ne dépasse le seuil opérationnel."""
    scores, labels, _ = evaluate()
    fps = [s for s, l in zip(scores, labels) if l == 0 and s >= THRESHOLD]
    assert len(fps) == 0, f"Faux positifs détectés : {fps}"

def test_evaluate_detects_tool_signatures():
    """Au moins 30 deepfakes avec signature d'outil sont détectés."""
    scores, labels, descs = evaluate()
    tps = [descs[i] for i, (s, l) in enumerate(zip(scores, labels))
           if l == 1 and s >= THRESHOLD]
    assert len(tps) >= 30

def test_evaluate_auc_above_threshold():
    scores, labels, _ = evaluate()
    auc = _compute_auc(scores, labels)
    assert auc >= 0.90, f"AUC inattendu : {auc}"

def test_evaluate_eer_below_threshold():
    scores, labels, _ = evaluate()
    eer, _ = _compute_eer(scores, labels)
    assert eer <= 0.15, f"EER inattendu : {eer}"

def test_evaluate_precision_perfect():
    """Précision = 1.0 (FAR=0) au seuil opérationnel."""
    scores, labels, _ = evaluate()
    m = _compute_metrics_at_threshold(scores, labels, THRESHOLD)
    assert m["precision"] == pytest.approx(1.0)


# ── Production de metrics.json ────────────────────────────────────────────────

def test_build_metrics_report_valid_schema(tmp_path):
    scores, labels, _ = evaluate()
    # Patch protocol path pour pointer sur un fichier temp valide
    proto = tmp_path / "baseline_protocol.yaml"
    proto.write_text("status: applied_metadata_v1\nnotes: test\n")

    import eval_metadata_engine as em
    original_path = em._PROTOCOL_PATH
    em._PROTOCOL_PATH = proto
    try:
        report = build_metrics_report(scores, labels)
    finally:
        em._PROTOCOL_PATH = original_path

    assert report.engine_name == "metadata"
    assert report.model_version == "v1.0.0"
    assert 0.0 <= report.auc <= 1.0
    assert 0.0 <= report.eer <= 1.0
    assert 0.0 <= report.far <= 1.0
    assert 0.0 <= report.frr <= 1.0
    assert report.train_dataset.name != report.test_dataset.name

def test_build_metrics_report_to_file(tmp_path):
    scores, labels, _ = evaluate()
    proto = tmp_path / "baseline_protocol.yaml"
    proto.write_text("status: applied_metadata_v1\nnotes: test\n")

    import eval_metadata_engine as em
    original_path = em._PROTOCOL_PATH
    em._PROTOCOL_PATH = proto
    try:
        report = build_metrics_report(scores, labels)
    finally:
        em._PROTOCOL_PATH = original_path

    out_file = tmp_path / "metrics.json"
    sha = report.to_file(out_file)

    assert out_file.exists()
    assert len(sha) == 64  # SHA-256 hex = 64 chars

    data = json.loads(out_file.read_text())
    assert data["engine_name"] == "metadata"
    assert data["schema_version"] == "1.0"
    assert "auc" in data and "eer" in data

def test_metrics_json_cross_dataset_constraint(tmp_path):
    """MetricsReport rejette train_dataset == test_dataset."""
    from metrics_schema import DatasetRef, MetricsReport
    import pytest

    with pytest.raises(Exception):
        MetricsReport(
            schema_version="1.0",
            engine_name="metadata",
            model_version="v1.0.0",
            protocol_hash="abc",
            train_dataset=DatasetRef(name="same", version="1.0", split="train", n_samples=10, n_identities=1),
            test_dataset=DatasetRef(name="same", version="1.0", split="test", n_samples=10, n_identities=1),
            far=0.0, frr=0.0, eer=0.0, auc=0.0, threshold_used=0.35,
            evaluated_at=__import__("datetime").datetime.now(),
            evaluator="test",
        )
