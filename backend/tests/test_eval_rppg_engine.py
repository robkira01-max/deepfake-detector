"""Tests pour training/eval_rppg_engine.py — Règle 10, condition 1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "training"))
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

from eval_rppg_engine import (  # noqa: E402
    FPS,
    N_FRAMES,
    TEST_CASES,
    THRESHOLD,
    _build_test_cases,
    _compute_auc,
    _compute_eer,
    _compute_metrics_at_threshold,
    _generate_authentic,
    _generate_deepfake,
    build_metrics_report,
    compute_rppg_score,
    evaluate,
)


# ── Intégrité du jeu de test ──────────────────────────────────────────────────

def test_test_cases_count():
    assert len(TEST_CASES) == 100

def test_test_cases_balanced():
    assert sum(c.label for c in TEST_CASES) == 50
    assert sum(1 - c.label for c in TEST_CASES) == 50

def test_test_cases_labels_binary():
    for c in TEST_CASES:
        assert c.label in (0, 1)

def test_test_cases_descriptions_unique():
    descs = [c.description for c in TEST_CASES]
    assert len(descs) == len(set(descs))

def test_authentic_cases_have_bpm():
    for c in TEST_CASES[:50]:
        assert c.label == 0
        assert c.bpm is not None
        assert 60.0 <= c.bpm <= 100.0

def test_deepfake_cases_have_no_bpm():
    for c in TEST_CASES[50:]:
        assert c.label == 1
        assert c.bpm is None

def test_test_cases_shape():
    for c in TEST_CASES:
        assert c.rgb_signals.shape == (N_FRAMES, 3)

def test_rgb_signals_positive():
    for c in TEST_CASES:
        assert np.all(c.rgb_signals > 0)


# ── Génération synthétique ────────────────────────────────────────────────────

def test_generate_authentic_shape():
    sig = _generate_authentic(bpm=72.0, seed=0)
    assert sig.shape == (N_FRAMES, 3)

def test_generate_authentic_positive():
    sig = _generate_authentic(bpm=72.0, seed=0)
    assert np.all(sig > 0)

def test_generate_authentic_r_dominates():
    """Le canal R a la plus grande amplitude de variation (signal cardiaque)."""
    sig = _generate_authentic(bpm=72.0, seed=42)
    assert np.std(sig[:, 0]) > np.std(sig[:, 2])  # std(R) > std(B)

def test_generate_deepfake_shape():
    sig = _generate_deepfake(seed=100)
    assert sig.shape == (N_FRAMES, 3)

def test_generate_deepfake_proportional():
    """Tous les canaux deepfake ont le même profil de variation (proportionnel)."""
    sig = _generate_deepfake(seed=101)
    # R_norm ≈ G_norm ≈ B_norm si variation proportionnelle
    R_norm = sig[:, 0] / np.mean(sig[:, 0])
    G_norm = sig[:, 1] / np.mean(sig[:, 1])
    B_norm = sig[:, 2] / np.mean(sig[:, 2])
    # Différences normalisées doivent être proches de zéro
    assert np.max(np.abs(R_norm - G_norm)) < 0.01
    assert np.max(np.abs(R_norm - B_norm)) < 0.01

def test_generate_deepfake_reproducible():
    sig1 = _generate_deepfake(seed=999)
    sig2 = _generate_deepfake(seed=999)
    np.testing.assert_array_equal(sig1, sig2)

def test_generate_authentic_reproducible():
    sig1 = _generate_authentic(bpm=80.0, seed=7)
    sig2 = _generate_authentic(bpm=80.0, seed=7)
    np.testing.assert_array_equal(sig1, sig2)


# ── Algorithme rPPG ───────────────────────────────────────────────────────────

def test_authentic_score_low():
    """Signal authentique : score < 0.5 (SNR élevé → signal cardiaque présent)."""
    sig = _generate_authentic(bpm=72.0, seed=0)
    assert compute_rppg_score(sig) < 0.5

def test_deepfake_score_high():
    """Signal deepfake (illumination uniforme) : score ≥ 0.5 (SNR faible → CHROM annule)."""
    sig = _generate_deepfake(seed=100)
    assert compute_rppg_score(sig) >= 0.5

def test_score_range():
    """Le score doit être dans [0, 1]."""
    for case in TEST_CASES[:20]:
        s = compute_rppg_score(case.rgb_signals)
        assert 0.0 <= s <= 1.0, f"Score hors bornes : {s}"

def test_score_constant_signal():
    """Signal constant (aucune variation) → rppg ≈ 0 → score = 1.0 (deepfake)."""
    sig = np.ones((N_FRAMES, 3)) * np.array([150.0, 100.0, 80.0])
    score = compute_rppg_score(sig)
    assert score == pytest.approx(1.0, abs=0.01)

def test_score_too_short():
    """Signal trop court → retour 0.3 (neutre)."""
    sig = np.random.rand(20, 3)
    assert compute_rppg_score(sig) == pytest.approx(0.3)

def test_score_cardiac_bpm_range():
    """Tous les BPM 60-100 sont correctement détectés (score < 0.5)."""
    for bpm in [60, 72, 80, 90, 100]:
        sig = _generate_authentic(bpm=float(bpm), seed=bpm)
        score = compute_rppg_score(sig)
        assert score < 0.5, f"BPM {bpm}: score trop élevé ({score:.3f})"

def test_score_proportional_deepfake():
    """Signal proportionnel (illumination uniforme) → score ≥ 0.5."""
    t = np.arange(N_FRAMES) / FPS
    common = 0.05 * np.sin(2 * np.pi * 0.1 * t)
    sig = np.column_stack([
        150 * (1 + common),
        100 * (1 + common),
        80  * (1 + common),
    ])
    assert compute_rppg_score(sig) >= 0.5


# ── Métriques scalaires ───────────────────────────────────────────────────────

def test_auc_perfect_classifier():
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = [1,   1,   0,   0]
    assert _compute_auc(scores, labels) == pytest.approx(1.0)

def test_auc_worst_classifier():
    scores = [0.1, 0.2, 0.9, 0.8]
    labels = [1,   1,   0,   0]
    auc = _compute_auc(scores, labels)
    assert auc == pytest.approx(1.0)  # _compute_auc retourne max(auc, 1-auc)

def test_auc_no_positive():
    assert _compute_auc([0.1, 0.2], [0, 0]) == 0.0

def test_auc_no_negative():
    assert _compute_auc([0.1, 0.2], [1, 1]) == 0.0

def test_eer_perfect_separation():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1,   1,   0,   0]
    eer, _ = _compute_eer(scores, labels)
    assert eer < 0.05

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
    assert m["tp"] == 2 and m["fp"] == 0
    assert m["tn"] == 2 and m["fn"] == 0
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["far"] == pytest.approx(0.0)
    assert m["frr"] == pytest.approx(0.0)

def test_metrics_at_threshold_all_miss():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [1,   1,   0,   0]
    m = _compute_metrics_at_threshold(scores, labels, 0.5)
    assert m["fn"] == 2
    assert m["frr"] == pytest.approx(1.0)


# ── Évaluation sur le jeu complet ─────────────────────────────────────────────

def test_evaluate_returns_correct_length():
    scores, labels, descs = evaluate()
    assert len(scores) == len(labels) == len(descs) == 100

def test_evaluate_all_scores_in_range():
    scores, _, _ = evaluate()
    for s in scores:
        assert 0.0 <= s <= 1.0

def test_evaluate_authentic_mostly_low():
    """Au moins 90% des authentiques scorent < 0.5."""
    scores, labels, _ = evaluate()
    auth = [s for s, l in zip(scores, labels) if l == 0]
    pct_low = sum(1 for s in auth if s < 0.5) / len(auth)
    assert pct_low >= 0.90, f"Seulement {pct_low*100:.0f}% des authentiques < 0.5"

def test_evaluate_deepfake_all_high():
    """Tous les deepfakes (illumination uniforme) scorent >= 0.5."""
    scores, labels, _ = evaluate()
    fakes = [s for s, l in zip(scores, labels) if l == 1]
    pct_high = sum(1 for s in fakes if s >= 0.5) / len(fakes)
    assert pct_high == 1.0, f"Seulement {pct_high*100:.0f}% des deepfakes >= 0.5"

def test_evaluate_frr_zero():
    """FRR = 0 : aucun deepfake classé comme authentique."""
    scores, labels, _ = evaluate()
    m = _compute_metrics_at_threshold(scores, labels, THRESHOLD)
    assert m["frr"] == pytest.approx(0.0)

def test_evaluate_auc_near_perfect():
    scores, labels, _ = evaluate()
    auc = _compute_auc(scores, labels)
    assert auc >= 0.95

def test_evaluate_recall_perfect():
    scores, labels, _ = evaluate()
    m = _compute_metrics_at_threshold(scores, labels, THRESHOLD)
    assert m["recall"] == pytest.approx(1.0)


# ── Production de metrics.json ────────────────────────────────────────────────

def test_build_metrics_report_valid_schema(tmp_path):
    scores, labels, _ = evaluate()
    proto = tmp_path / "baseline_protocol.yaml"
    proto.write_text("status: applied_rppg_v1\nnotes: test\n")

    import eval_rppg_engine as er
    original = er._PROTOCOL_PATH
    er._PROTOCOL_PATH = proto
    try:
        report = build_metrics_report(scores, labels)
    finally:
        er._PROTOCOL_PATH = original

    assert report.engine_name == "rppg"
    assert report.model_version == "v1.0.0"
    assert 0.0 <= report.auc <= 1.0
    assert 0.0 <= report.far <= 1.0
    assert 0.0 <= report.frr <= 1.0
    assert report.train_dataset.name != report.test_dataset.name

def test_build_metrics_report_to_file(tmp_path):
    scores, labels, _ = evaluate()
    proto = tmp_path / "baseline_protocol.yaml"
    proto.write_text("status: applied_rppg_v1\n")

    import eval_rppg_engine as er
    original = er._PROTOCOL_PATH
    er._PROTOCOL_PATH = proto
    try:
        report = build_metrics_report(scores, labels)
    finally:
        er._PROTOCOL_PATH = original

    out = tmp_path / "rppg.json"
    sha = report.to_file(out)
    assert out.exists()
    assert len(sha) == 64  # SHA-256

    data = json.loads(out.read_text())
    assert data["engine_name"] == "rppg"
    assert data["schema_version"] == "1.0"
    assert "auc" in data and "eer" in data

def test_metrics_json_cross_dataset_constraint(tmp_path):
    from metrics_schema import DatasetRef, MetricsReport
    import datetime
    with pytest.raises(Exception):
        MetricsReport(
            schema_version="1.0",
            engine_name="rppg",
            model_version="v1.0.0",
            protocol_hash="abc",
            train_dataset=DatasetRef(name="same", version="1.0", split="train", n_samples=50, n_identities=1),
            test_dataset=DatasetRef(name="same", version="1.0", split="test", n_samples=50, n_identities=25),
            far=0.0, frr=0.0, eer=0.0, auc=1.0, threshold_used=0.5,
            evaluated_at=datetime.datetime.now(),
            evaluator="test",
        )

def test_rppg_json_file_exists():
    """Le fichier evaluation/metrics/rppg.json a été généré."""
    rppg_metrics = _PROJECT_ROOT / "evaluation" / "metrics" / "rppg.json"
    assert rppg_metrics.exists(), "Lancer eval_rppg_engine.py pour générer rppg.json"
    data = json.loads(rppg_metrics.read_text())
    assert data["engine_name"] == "rppg"
    assert data["far"] <= 0.05
    assert data["frr"] == pytest.approx(0.0)
    assert data["auc"] >= 0.95


# ── Porte de validation — per-engine metrics path ─────────────────────────────

def test_validation_gate_finds_rppg_metrics(tmp_path):
    """_metrics_path_for('rppg') trouve evaluation/metrics/rppg.json."""
    from engines.validation_gate import _metrics_path_for
    path = _metrics_path_for("rppg")
    assert path.exists()
    assert path.name == "rppg.json"

def test_validation_gate_fallback_legacy(tmp_path):
    """_metrics_path_for retourne le fichier legacy si le per-engine n'existe pas."""
    from engines.validation_gate import _metrics_path_for
    # Engine fictif sans fichier propre
    path = _metrics_path_for("nonexistent_engine_xyz")
    assert path.name == "metrics.json"

def test_validation_gate_rppg_condition1(tmp_path):
    """Condition 1 passée pour rppg (rppg.json valide)."""
    from engines.validation_gate import _check_metrics
    reasons: list[str] = []
    ok, path = _check_metrics("rppg", reasons)
    assert ok, f"Condition 1 échouée : {reasons}"
    assert path is not None and path.name == "rppg.json"
