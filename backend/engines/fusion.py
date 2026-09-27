"""Fusion d'ensemble des scores vidéo + audio + métadonnées.

Formule (composantes actives uniquement) :
  score_final = Σ (w_i / Σw_actif) · S_i   pour i dans composantes actives

Composante active si :
  - statut "validated"  → toujours incluse
  - statut "experimental" → incluse si settings.allow_experimental_engines = True
  - statut "disabled"   → toujours exclue (ex: Wav2Vec2 tête aléatoire)

Seuils de décision :
  0.00 – 0.35 → AUTHENTIQUE
  0.35 – 0.55 → INDÉTERMINÉ
  0.55 – 1.00 → DEEPFAKE DÉTECTÉ
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from config import settings
from models.analysis import Verdict

EngineStatus = Literal["validated", "experimental", "disabled"]

# Poids de l'ensemble (objectif : calibrage sur FaceForensics++ + ASVspoof 2021 après fine-tuning)
WEIGHTS: dict[str, float] = {
    "texture":    0.20,
    "temporal":   0.15,
    "rppg":       0.18,
    "biometrics": 0.10,
    "audio":      0.20,
    "phase":      0.10,
    "metadata":   0.07,
}

# Statut actuel de chaque composante (Brief v3 §0.6 / §6)
# Remplace l'ancien VALIDATED_COMPONENTS frozenset.
# Surchargeable par fuse_scores(engine_statuses=...) pour les tests ou le registre.
ENGINE_DEFAULT_STATUS: dict[str, EngineStatus] = {
    "texture":    "experimental",  # EfficientNet-B4 poids ImageNet — non fine-tuné deepfake
    "temporal":   "experimental",  # ResNet-50+LSTM poids ImageNet
    "rppg":       "experimental",  # algorithme CHROM — fonctionnel, non benchmarké deepfake
    "biometrics": "experimental",  # EAR+AU — fonctionnel, non benchmarké deepfake
    "audio":      "disabled",      # Wav2Vec2 tête de classification aléatoire (non entraînée)
    "phase":      "experimental",  # STFT phase — fonctionnel, non benchmarké deepfake
    "metadata":   "experimental",  # heuristiques métadonnées — fonctionnel, non benchmarké
}

THRESHOLD_AUTHENTIC = 0.35
THRESHOLD_DEEPFAKE  = 0.55

# Intervalle de confiance à 95% (bande d'incertitude du modèle)
IC_HALF_WIDTH = 0.08


@dataclass
class FusionResult:
    final_score: float
    verdict: Verdict
    confidence_low: float
    confidence_high: float
    component_scores: dict[str, float]
    shap_ranking: list[dict]
    plain_explanation: str

    # Composantes exclues du score (disabled ou experimental quand allow=False)
    unvalidated_components: list[str] = field(default_factory=list)
    # Composantes incluses mais pas encore validées sur un jeu de test indépendant
    experimental_components: list[str] = field(default_factory=list)

    # Métriques issues d'un jeu de test indépendant — None si non encore mesurées (Brief v3 §0.5)
    model_far: float | None = None
    model_frr: float | None = None
    model_eer: float | None = None
    model_auc: float | None = None


def fuse_scores(
    score_texture: float = 0.0,
    score_temporal: float = 0.0,
    score_rppg: float = 0.0,
    score_biometrics: float = 0.0,
    score_audio: float = 0.0,
    score_phase: float = 0.0,
    score_metadata: float = 0.0,
    engine_statuses: dict[str, EngineStatus] | None = None,
) -> FusionResult:
    """Calcule le score final pondéré et détermine le verdict.

    Args:
        engine_statuses: Surcharge du statut par composante.
                         Si None, utilise ENGINE_DEFAULT_STATUS.
                         Utilisé par le registre de modèles pour appliquer les statuts réels.
    """
    components = {
        "texture":    score_texture,
        "temporal":   score_temporal,
        "rppg":       score_rppg,
        "biometrics": score_biometrics,
        "audio":      score_audio,
        "phase":      score_phase,
        "metadata":   score_metadata,
    }

    statuses = engine_statuses if engine_statuses is not None else ENGINE_DEFAULT_STATUS
    allow_experimental = settings.allow_experimental_engines

    # Partition des composantes selon le statut
    active_set = frozenset(
        k for k, s in statuses.items()
        if s == "validated" or (s == "experimental" and allow_experimental)
    )
    excluded = sorted(k for k in components if k not in active_set)
    experimental = sorted(
        k for k, s in statuses.items()
        if k in active_set and s == "experimental"
    )

    # Score pondéré — composantes actives uniquement, poids renormalisés
    valid_weight_sum = sum(WEIGHTS[k] for k in active_set if k in WEIGHTS) or 1.0
    final = sum(
        WEIGHTS[k] / valid_weight_sum * v
        for k, v in components.items()
        if k in active_set
    )
    final = max(0.0, min(1.0, final))

    # Intervalle de confiance
    ic_low  = max(0.0, final - IC_HALF_WIDTH)
    ic_high = min(1.0, final + IC_HALF_WIDTH)

    # Verdict
    if final < THRESHOLD_AUTHENTIC:
        verdict = Verdict.authentic
    elif final < THRESHOLD_DEEPFAKE:
        verdict = Verdict.undetermined
    else:
        verdict = Verdict.deepfake

    # Classement SHAP — composantes actives uniquement, poids renormalisés
    weighted = {
        k: WEIGHTS[k] / valid_weight_sum * v
        for k, v in components.items()
        if k in active_set
    }
    total_w = sum(weighted.values()) or 1.0
    shap_ranking = sorted(
        [
            {
                "feature":      k,
                "contribution": round(v / total_w * 100, 1),
                "raw_score":    round(components[k], 4),
                "weight":       round(WEIGHTS[k] / valid_weight_sum, 4),
                "status":       statuses.get(k, "experimental"),
            }
            for k, v in weighted.items()
        ],
        key=lambda x: x["contribution"],
        reverse=True,
    )

    explanation = _generate_plain_explanation(
        verdict, final, shap_ranking, excluded, experimental
    )

    return FusionResult(
        final_score=round(final, 4),
        verdict=verdict,
        confidence_low=round(ic_low, 4),
        confidence_high=round(ic_high, 4),
        component_scores={k: round(v, 4) for k, v in components.items()},
        shap_ranking=shap_ranking,
        plain_explanation=explanation,
        unvalidated_components=excluded,
        experimental_components=experimental,
    )


def _generate_plain_explanation(
    verdict: Verdict,
    score: float,
    ranking: list[dict],
    excluded: list[str] | None = None,
    experimental: list[str] | None = None,
) -> str:
    """Génère une explication en langage clair pour les non-techniciens
    (juges, avocats, banquiers) — exigence XAI du cahier des charges.
    """
    pct = int(score * 100)
    top_features = [r["feature"] for r in ranking[:3]]

    feature_labels = {
        "texture":    "anomalies de texture du visage",
        "temporal":   "incohérences entre images successives",
        "rppg":       "absence de signal cardiaque visible",
        "biometrics": "comportement facial anormal (clignements, lèvres)",
        "audio":      "caractéristiques vocales synthétiques",
        "phase":      "discontinuités dans le signal sonore",
        "metadata":   "incohérences dans les métadonnées du fichier",
    }

    top_labels = [feature_labels.get(f, f) for f in top_features]

    excl_note = (
        f" Module(s) exclu(s) du score (non validés) : {', '.join(excluded)}."
        if excluded else ""
    )
    exp_note = (
        f" Module(s) expérimental(aux) inclus (poids ImageNet, non fine-tunés deepfake) : "
        f"{', '.join(experimental)}."
        if experimental else ""
    )

    if verdict == Verdict.authentic:
        return (
            f"Le fichier analysé présente un score de risque de {pct}%, "
            f"ce qui est en dessous du seuil de détection de 35%. "
            f"Le système n'a pas détecté de signes caractéristiques de manipulation numérique. "
            f"Les indicateurs analysés — notamment {', '.join(top_labels[:2])} — "
            f"sont cohérents avec un enregistrement authentique.{excl_note}{exp_note}"
        )
    elif verdict == Verdict.undetermined:
        return (
            f"Le fichier présente un score de risque de {pct}% (zone d'incertitude : 35-55%). "
            f"Le système a relevé des anomalies modérées concernant : "
            f"{', '.join(top_labels)}. "
            f"Une expertise humaine complémentaire est recommandée avant de tirer des conclusions."
            f"{excl_note}{exp_note}"
        )
    else:
        return (
            f"Le fichier présente un score de risque élevé de {pct}%, "
            f"dépassant le seuil de détection de 55%. "
            f"Les principales anomalies détectées sont : {', '.join(top_labels)}. "
            f"Ces caractéristiques sont typiques d'une manipulation par intelligence artificielle. "
            f"Note : ce résultat doit être corroboré par d'autres éléments de preuve."
            f"{excl_note}{exp_note}"
        )
