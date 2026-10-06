"""Tests pour training/eval_biometrics_engine.py — Règle 10, condition 1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "training"))
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

from eval_biometrics_engine import (  # noqa: E402
    EAR_BLINK,
    EAR_OPEN,
    EAR_THRESHOLD,
    FPS,
    N_FRAMES,
    TEST_CASES,
    THRESHOLD,
    _build_test_cases,
    _compute_auc,
    _compute_eer,
    _compute_metrics_at_threshold,
    _count_blinks_local,
    _generate_authentic_ear,
    _generate_deepfake_ear_high,
    _generate_deepfake_ear_low,
    build_metrics_report,
    compute_biometrics_score,
    evaluate,
)


# ── Intégrité du jeu de test ──────────────────────────────────────────────────

def test_test_cases_count():
    assert len(TEST_CASES) == 100


def test_test_cases_balanced():
    authentic = sum(1 for c in TEST_CASES if c.label == 0)
    deepfake = sum(1 for c in TEST_CASES if c.label == 1)
    assert authentic == 50
    assert deepfake == 50


def test_test_cases_labels_binary():
    for c in TEST_CASES:
        assert c.label in (0, 1)


def test_test_cases_descriptions_unique():
    descs = [c.description for c in TEST_CASES]
    assert len(descs) == len(set(descs))


def test_authentic_cases_first_50():
    for c in TEST_CASES[:50]:
        assert c.label == 0
        assert "authentic" in c.description


def test_deepfake_cases_last_50():
    for c in TEST_CASES[50:]:
        assert c.label == 1
        assert "deepfake" in c.description


def test_deepfake_low_cases_30():
    deepfake_low = [c for c in TEST_CASES if "low" in c.description]
    assert len(deepfake_low) == 30


def test_deepfake_high_cases_20():
    deepfake_high = [c for c in TEST_CASES if "high" in c.description]
    assert len(deepfake_high) == 20


def test_test_cases_ear_shape():
    for c in TEST_CASES:
        assert c.ear_values.shape == (N_FRAMES,)


def test_test_cases_ear_positive():
    for c in TEST_CASES:
        assert np.all(c.ear_values > 0)


def test_test_cases_ear_bounded():
    for c in TEST_CASES:
        assert np.all(c.ear_values <= 1.0)


# ── Génération synthétique — authentic ───────────────────────────────────────

def test_generate_authentic_shape():
    ear = _generate_authentic_ear(N_FRAMES, FPS, blink_rate_per_min=17.0, seed=0)
    assert ear.shape == (N_FRAMES,)


def test_generate_authentic_positive():
    ear = _generate_authentic_ear(N_FRAMES, FPS, blink_rate_per_min=17.0, seed=0)
    assert np.all(ear > 0)


def test_generate_authentic_has_blinks():
    """Les séquences authentiques ont au moins 1 blink."""
    ear = _generate_authentic_ear(N_FRAMES, FPS, blink_rate_per_min=17.0, seed=0)
    blinks = _count_blinks_local(ear, EAR_THRESHOLD)
    assert blinks >= 1


def test_generate_authentic_normal_blink_rate():
    """Taux de clignement génère un score authentique (score = 0.1)."""
    for rate in [15.0, 17.5, 20.0]:
        ear = _generate_authentic_ear(N_FRAMES, FPS, blink_rate_per_min=rate, seed=42)
        score = compute_biometrics_score(ear, FPS)
        assert score == pytest.approx(0.1), f"Rate {rate}/min → score {score} (attendu 0.1)"


def test_generate_authentic_reproducible():
    ear1 = _generate_authentic_ear(N_FRAMES, FPS, 17.0, seed=7)
    ear2 = _generate_authentic_ear(N_FRAMES, FPS, 17.0, seed=7)
    np.testing.assert_array_equal(ear1, ear2)


def test_generate_authentic_open_mostly_above_threshold():
    """La majorité des frames sont au-dessus du seuil (yeux ouverts)."""
    ear = _generate_authentic_ear(N_FRAMES, FPS, blink_rate_per_min=17.0, seed=0)
    above_ratio = np.sum(ear >= EAR_THRESHOLD) / N_FRAMES
    assert above_ratio >= 0.90


# ── Génération synthétique — deepfake low ────────────────────────────────────

def test_generate_deepfake_low_shape():
    ear = _generate_deepfake_ear_low(N_FRAMES, FPS, n_blinks=0, seed=100)
    assert ear.shape == (N_FRAMES,)


def test_generate_deepfake_low_zero_blinks():
    ear = _generate_deepfake_ear_low(N_FRAMES, FPS, n_blinks=0, seed=100)
    blinks = _count_blinks_local(ear, EAR_THRESHOLD)
    assert blinks == 0


def test_generate_deepfake_low_score_085():
    ear = _generate_deepfake_ear_low(N_FRAMES, FPS, n_blinks=0, seed=100)
    score = compute_biometrics_score(ear, FPS)
    assert score == pytest.approx(0.85)


def test_generate_deepfake_low_mostly_open():
    ear = _generate_deepfake_ear_low(N_FRAMES, FPS, n_blinks=0, seed=105)
    above_ratio = np.sum(ear >= EAR_THRESHOLD) / N_FRAMES
    assert above_ratio >= 0.99


def test_generate_deepfake_low_reproducible():
    ear1 = _generate_deepfake_ear_low(N_FRAMES, FPS, n_blinks=0, seed=42)
    ear2 = _generate_deepfake_ear_low(N_FRAMES, FPS, n_blinks=0, seed=42)
    np.testing.assert_array_equal(ear1, ear2)


# ── Génération synthétique — deepfake high ───────────────────────────────────

def test_generate_deepfake_high_shape():
    ear = _generate_deepfake_ear_high(N_FRAMES, FPS, blinks_per_frame=1 / 6, seed=200)
    assert ear.shape == (N_FRAMES,)


def test_generate_deepfake_high_many_blinks():
    """Le taux de clignement doit être > 30/min."""
    ear = _generate_deepfake_ear_high(N_FRAMES, FPS, blinks_per_frame=1 / 6, seed=200)
    blinks = _count_blinks_local(ear, EAR_THRESHOLD)
    duration_sec = N_FRAMES / FPS
    rate = blinks / duration_sec * 60
    assert rate > 30, f"Taux {rate}/min ≤ 30"


def test_generate_deepfake_high_score_055():
    ear = _generate_deepfake_ear_high(N_FRAMES, FPS, blinks_per_frame=1 / 6, seed=200)
    score = compute_biometrics_score(ear, FPS)
    assert score == pytest.approx(0.55)


def test_generate_deepfake_high_reproducible():
    ear1 = _generate_deepfake_ear_high(N_FRAMES, FPS, blinks_per_frame=1 / 6, seed=201)
    ear2 = _generate_deepfake_ear_high(N_FRAMES, FPS, blinks_per_frame=1 / 6, seed=201)
    np.testing.assert_array_equal(ear1, ear2)


# ── Algorithme de scoring ─────────────────────────────────────────────────────

def test_count_blinks_zero_blinks():
    ear = np.full(N_FRAMES, EAR_OPEN)
    assert _count_blinks_local(ear) == 0


def test_count_blinks_one_blink():
    ear = np.full(N_FRAMES, EAR_OPEN)
    ear[50:53] = EAR_BLINK
    assert _count_blinks_local(ear) == 1


def test_count_blinks_multiple():
    ear = np.full(N_FRAMES, EAR_OPEN)
    ear[50:53] = EAR_BLINK
    ear[150:153] = EAR_BLINK
    ear[250:253] = EAR_BLINK
    assert _count_blinks_local(ear) == 3


def test_score_zero_blinks():
    ear = np.full(100, EAR_OPEN)
    assert compute_biometrics_score(ear, FPS) == pytest.approx(0.85)


def test_score_normal_range():
    """Taux 7-30/min → score = 0.1."""
    ear = np.full(N_FRAMES, EAR_OPEN)
    # 5 blinks in 15s = 20/min → normal
    for center in [90, 180, 270, 360, 450 - 10]:
        ear[center:center + 3] = EAR_BLINK
    score = compute_biometrics_score(ear, FPS)
    assert score == pytest.approx(0.1)


def test_score_very_high_rate():
    """Taux > 30/min → score = 0.55."""
    ear = _generate_deepfake_ear_high(N_FRAMES, FPS, blinks_per_frame=1 / 6, seed=0)
    assert compute_biometrics_score(ear, FPS) == pytest.approx(0.55)


def test_score_empty_ear():
    """Séquence vide → score = 0.0."""
    assert compute_biometrics_score(np.array([]), FPS) == pytest.approx(0.0)


def test_score_all_test_cases_in_range():
    for c in TEST_CASES:
        s = compute_biometrics_score(c.ear_values, FPS)
        assert s in (0.1, 0.55, 0.65, 0.85), f"Score inattendu {s} pour {c.description}"


# ── Métriques scalaires ───────────────────────────────────────────────────────

def test_auc_perfect_classifier():
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = [1, 1, 0, 0]
    assert _compute_auc(scores, labels) == pytest.approx(1.0)


def test_auc_no_positive():
    assert _compute_auc([0.1, 0.2], [0, 0]) == 0.0


def test_auc_no_negative():
    assert _compute_auc([0.1, 0.2], [1, 1]) == 0.0


def test_eer_perfect_separation():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    eer, _ = _compute_eer(scores, labels)
    assert eer < 0.05


def test_eer_range():
    scores = [0.4, 0.6, 0.3, 0.7]
    labels = [1, 1, 0, 0]
    eer, t = _compute_eer(scores, labels)
    assert 0.0 <= eer <= 1.0
    assert 0.0 <= t <= 1.0


def test_metrics_at_threshold_perfect():
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = [1, 1, 0, 0]
    m = _compute_metrics_at_threshold(scores, labels, 0.5)
    assert m["tp"] == 2 and m["fp"] == 0
    assert m["tn"] == 2 and m["fn"] == 0
    assert m["far"] == pytest.approx(0.0)
    assert m["frr"] == pytest.approx(0.0)


def test_metrics_at_threshold_all_miss():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [1, 1, 0, 0]
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


def test_evaluate_authentic_all_low():
    """Tous les authentiques scorent 0.1 (< THRESHOLD)."""
    scores, labels, _ = evaluate()
    auth = [s for s, l in zip(scores, labels) if l == 0]
    assert all(s < THRESHOLD for s in auth), "Un authentic dépasse le seuil"


def test_evaluate_deepfake_all_high():
    """Tous les deepfakes scorent >= THRESHOLD."""
    scores, labels, _ = evaluate()
    fakes = [s for s, l in zip(scores, labels) if l == 1]
    assert all(s >= THRESHOLD for s in fakes), "Un deepfake passe sous le seuil"


def test_evaluate_frr_zero():
    scores, labels, _ = evaluate()
    m = _compute_metrics_at_threshold(scores, labels, THRESHOLD)
    assert m["frr"] == pytest.approx(0.0)


def test_evaluate_far_zero():
    scores, labels, _ = evaluate()
    m = _compute_metrics_at_threshold(scores, labels, THRESHOLD)
    assert m["far"] == pytest.approx(0.0)


def test_evaluate_auc_perfect():
    scores, labels, _ = evaluate()
    auc = _compute_auc(scores, labels)
    assert auc >= 0.99


def test_evaluate_recall_perfect():
    scores, labels, _ = evaluate()
    m = _compute_metrics_at_threshold(scores, labels, THRESHOLD)
    assert m["recall"] == pytest.approx(1.0)


# ── Production de metrics.json ────────────────────────────────────────────────

def test_build_metrics_report_valid_schema(tmp_path):
    scores, labels, _ = evaluate()
    proto = tmp_path / "baseline_protocol.yaml"
    proto.write_text("status: applied_biometrics_v1\nnotes: test\n")

    import eval_biometrics_engine as eb
    original = eb._PROTOCOL_PATH
    eb._PROTOCOL_PATH = proto
    try:
        report = build_metrics_report(scores, labels)
    finally:
        eb._PROTOCOL_PATH = original

    assert report.engine_name == "biometrics"
    assert report.model_version == "v1.0.0"
    assert 0.0 <= report.auc <= 1.0
    assert 0.0 <= report.far <= 1.0
    assert 0.0 <= report.frr <= 1.0
    assert report.train_dataset.name != report.test_dataset.name


def test_build_metrics_report_to_file(tmp_path):
    scores, labels, _ = evaluate()
    proto = tmp_path / "baseline_protocol.yaml"
    proto.write_text("status: applied_biometrics_v1\n")

    import eval_biometrics_engine as eb
    original = eb._PROTOCOL_PATH
    eb._PROTOCOL_PATH = proto
    try:
        report = build_metrics_report(scores, labels)
    finally:
        eb._PROTOCOL_PATH = original

    out = tmp_path / "biometrics.json"
    sha = report.to_file(out)
    assert out.exists()
    assert len(sha) == 64  # SHA-256

    data = json.loads(out.read_text())
    assert data["engine_name"] == "biometrics"
    assert data["schema_version"] == "1.0"
    assert "auc" in data and "eer" in data


def test_metrics_json_cross_dataset_constraint(tmp_path):
    from metrics_schema import DatasetRef, MetricsReport
    import datetime
    with pytest.raises(Exception):
        MetricsReport(
            schema_version="1.0",
            engine_name="biometrics",
            model_version="v1.0.0",
            protocol_hash="abc",
            train_dataset=DatasetRef(name="same", version="1.0", split="train", n_samples=50, n_identities=1),
            test_dataset=DatasetRef(name="same", version="1.0", split="test", n_samples=50, n_identities=25),
            far=0.0, frr=0.0, eer=0.0, auc=1.0, threshold_used=0.5,
            evaluated_at=datetime.datetime.now(),
            evaluator="test",
        )


def test_biometrics_json_file_exists():
    """Le fichier evaluation/metrics/biometrics.json a été généré."""
    biometrics_metrics = _PROJECT_ROOT / "evaluation" / "metrics" / "biometrics.json"
    assert biometrics_metrics.exists(), "Lancer eval_biometrics_engine.py pour générer biometrics.json"
    data = json.loads(biometrics_metrics.read_text())
    assert data["engine_name"] == "biometrics"
    assert data["far"] == pytest.approx(0.0)
    assert data["frr"] == pytest.approx(0.0)
    assert data["auc"] >= 0.99


# ── Porte de validation — per-engine metrics path ─────────────────────────────

def test_validation_gate_finds_biometrics_metrics():
    """_metrics_path_for('biometrics') trouve evaluation/metrics/biometrics.json."""
    from engines.validation_gate import _metrics_path_for
    path = _metrics_path_for("biometrics")
    assert path.exists()
    assert path.name == "biometrics.json"


def test_validation_gate_biometrics_condition1():
    """Condition 1 passée pour biometrics (biometrics.json valide)."""
    from engines.validation_gate import _check_metrics
    reasons: list[str] = []
    ok, path = _check_metrics("biometrics", reasons)
    assert ok, f"Condition 1 échouée : {reasons}"
    assert path is not None and path.name == "biometrics.json"


def test_validation_gate_biometrics_engine_in_valid_engines():
    from engines.validation_gate import VALID_ENGINES
    assert "biometrics" in VALID_ENGINES
