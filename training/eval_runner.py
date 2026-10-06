"""Runner d'évaluation cross-dataset — Brief v3 §0.2 + §0.4."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration d'une évaluation (doit référencer un TrainingProtocol figé)."""

    engine_name: str
    model_version: str
    protocol_hash: str       # SHA-256 du TrainingProtocol figé (§0.1)
    train_dataset_name: str  # dataset d'entraînement
    test_dataset_name: str   # dataset de test — DOIT être différent (§0.4)
    threshold: float = 0.50
    evaluator: str = "automated"
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.train_dataset_name == self.test_dataset_name:
            raise ValueError(
                "Brief v3 §0.4 — Évaluation inter-jeux obligatoire : "
                f"train_dataset_name ('{self.train_dataset_name}') et "
                f"test_dataset_name ('{self.test_dataset_name}') doivent être différents."
            )


@dataclass
class EvalResult:
    far: float
    frr: float
    eer: float
    auc: float
    threshold_used: float
    n_genuine: int
    n_impostor: int
    scores_genuine: list[float] = field(default_factory=list)
    scores_impostor: list[float] = field(default_factory=list)


def compute_metrics(
    scores_genuine: list[float],
    scores_impostor: list[float],
    threshold: float,
) -> EvalResult:
    """Calcule FAR, FRR, EER, AUC à partir de scores bruts.

    Args:
        scores_genuine:  scores sur exemples authentiques (ground truth = 0)
        scores_impostor: scores sur exemples deepfake (ground truth = 1)
        threshold:       seuil de décision (score > threshold → DEEPFAKE)
    """
    if not scores_genuine or not scores_impostor:
        raise ValueError("scores_genuine et scores_impostor ne peuvent pas être vides")

    far = sum(1 for s in scores_genuine if s > threshold) / len(scores_genuine)
    frr = sum(1 for s in scores_impostor if s <= threshold) / len(scores_impostor)
    eer = _compute_eer(scores_genuine, scores_impostor)
    auc = _compute_auc(scores_genuine, scores_impostor)

    return EvalResult(
        far=round(far, 6),
        frr=round(frr, 6),
        eer=round(eer, 6),
        auc=round(auc, 6),
        threshold_used=threshold,
        n_genuine=len(scores_genuine),
        n_impostor=len(scores_impostor),
        scores_genuine=scores_genuine,
        scores_impostor=scores_impostor,
    )


def _compute_eer(genuine: list[float], impostor: list[float]) -> float:
    all_scores = sorted(set(genuine + impostor))
    best_eer, best_diff = 1.0, 1.0
    for t in all_scores:
        far_t = sum(1 for s in genuine if s > t) / len(genuine)
        frr_t = sum(1 for s in impostor if s <= t) / len(impostor)
        diff = abs(far_t - frr_t)
        if diff < best_diff:
            best_diff = diff
            best_eer = (far_t + frr_t) / 2
    return best_eer


def _compute_auc(genuine: list[float], impostor: list[float]) -> float:
    """AUC par méthode de Wilcoxon-Mann-Whitney (O(n²), suffisant pour évaluation offline)."""
    n_g, n_i = len(genuine), len(impostor)
    if n_g == 0 or n_i == 0:
        return 0.5
    return sum(
        1.0 if s_i > s_g else (0.5 if s_i == s_g else 0.0)
        for s_i in impostor
        for s_g in genuine
    ) / (n_g * n_i)


def run_evaluation(
    config: EvalConfig,
    predict_fn: Callable[[str], float],
    test_files_genuine: list[str],
    test_files_impostor: list[str],
    output_dir: Path,
) -> tuple[EvalResult, Path]:
    """Lance une évaluation cross-dataset et sauvegarde les résultats bruts.

    Args:
        config:               Configuration (protocol_hash vérifié)
        predict_fn:           f(path: str) → float score ∈ [0, 1]
        test_files_genuine:   Fichiers authentiques du jeu de TEST
        test_files_impostor:  Fichiers deepfake du jeu de TEST
        output_dir:           Répertoire de sortie pour raw_scores.json

    Returns:
        (EvalResult, chemin vers raw_scores.json)
    """
    log.info(
        "eval_start",
        engine=config.engine_name,
        version=config.model_version,
        test_dataset=config.test_dataset_name,
        n_genuine=len(test_files_genuine),
        n_impostor=len(test_files_impostor),
    )

    scores_genuine, scores_impostor = [], []
    for path in test_files_genuine:
        try:
            scores_genuine.append(float(predict_fn(path)))
        except Exception as exc:
            log.warning("eval_predict_failed", path=path, error=str(exc))

    for path in test_files_impostor:
        try:
            scores_impostor.append(float(predict_fn(path)))
        except Exception as exc:
            log.warning("eval_predict_failed", path=path, error=str(exc))

    result = compute_metrics(scores_genuine, scores_impostor, config.threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"raw_scores_{config.engine_name}_{config.model_version}.json"
    raw_path.write_text(
        json.dumps(
            {
                "config": {
                    "engine_name":    config.engine_name,
                    "model_version":  config.model_version,
                    "protocol_hash":  config.protocol_hash,
                    "train_dataset":  config.train_dataset_name,
                    "test_dataset":   config.test_dataset_name,
                    "threshold":      config.threshold,
                    "evaluated_at":   datetime.now(timezone.utc).isoformat(),
                    "evaluator":      config.evaluator,
                },
                "metrics": {
                    "far":        result.far,
                    "frr":        result.frr,
                    "eer":        result.eer,
                    "auc":        result.auc,
                    "n_genuine":  result.n_genuine,
                    "n_impostor": result.n_impostor,
                },
                "scores_genuine":  result.scores_genuine,
                "scores_impostor": result.scores_impostor,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info(
        "eval_complete",
        far=result.far,
        frr=result.frr,
        eer=result.eer,
        auc=result.auc,
        raw_scores=str(raw_path),
    )

    return result, raw_path
