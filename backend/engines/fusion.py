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
from pathlib import Path
from typing import Literal

from config import settings
from models.analysis import Verdict

# Chemin du fichier de métriques validé (CLAUDE.md Règle 7)
_METRICS_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "metrics" / "metrics.json"
# Statuts persistés (écrits par POST /models/engines/{name}/promote — Règle 10)
_ENGINE_STATUSES_PATH = Path(__file__).resolve().parents[2] / "evaluation" / "engine_statuses.json"


def _require_metrics() -> None:
    """Lève ValueError si aucun metrics.json n'est disponible.

    Appelée avant toute production de chiffre de fiabilité (FAR/FRR/AUC/EER).
    CLAUDE.md Règle 7 : ValueError, pas None ni valeur par défaut.
    """
    if not _METRICS_PATH.exists():
        raise ValueError(
            f"Aucun artefact de métriques disponible ({_METRICS_PATH}). "
            "Produire des chiffres de fiabilité sans metrics.json est interdit "
            "(CLAUDE.md Règle 7). Voir evaluation/protocols/ pour la procédure de validation."
        )

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

def load_engine_statuses() -> dict[str, EngineStatus]:
    """Charge les statuts depuis engine_statuses.json (overlay sur ENGINE_DEFAULT_STATUS).

    Si le fichier n'existe pas ou est illisible, retourne ENGINE_DEFAULT_STATUS.
    Appelée par fuse_scores() quand engine_statuses n'est pas fourni explicitement.
    """
    if not _ENGINE_STATUSES_PATH.exists():
        return ENGINE_DEFAULT_STATUS.copy()
    try:
        import json as _json  # noqa: PLC0415
        data = _json.loads(_ENGINE_STATUSES_PATH.read_text(encoding="utf-8"))
        overrides = data.get("statuses", {})
        result = ENGINE_DEFAULT_STATUS.copy()
        for name, s in overrides.items():
            if name in result and s in ("validated", "experimental", "disabled"):
                result[name] = s  # type: ignore[assignment]
        return result
    except Exception:
        return ENGINE_DEFAULT_STATUS.copy()


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

    # Indicateur de disponibilité des artefacts de métriques (CLAUDE.md Règle 7)
    # False tant que evaluation/metrics/metrics.json n'existe pas.
    # Ne jamais retourner FAR/FRR/AUC/EER sans ce fichier — utiliser _require_metrics().
    metrics_available: bool = False
    metrics_artifact: str | None = None


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
                         Si None, lit evaluation/engine_statuses.json via load_engine_statuses()
                         (ENGINE_DEFAULT_STATUS si le fichier est absent).
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

    statuses = engine_statuses if engine_statuses is not None else load_engine_statuses()
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

    # Abstention : 0 moteur actif → score indisponible (CLAUDE.md Règle 6)
    # Se produit quand allow_experimental=False et aucun moteur validated.
    # Ne pas retourner Verdict.authentic avec score=0.0 : ce serait trompeur.
    if not active_set:
        return FusionResult(
            final_score=0.0,
            verdict=Verdict.abstain,
            confidence_low=0.0,
            confidence_high=0.0,
            component_scores={k: round(v, 4) for k, v in components.items()},
            shap_ranking=[],
            plain_explanation=(
                "Aucun moteur de détection actif. "
                "Activer allow_experimental_engines ou valider au moins un moteur "
                "(CLAUDE.md Règle 6 + Règle 10) pour obtenir un score."
            ),
            unvalidated_components=excluded,
            experimental_components=[],
            metrics_available=_METRICS_PATH.exists(),
            metrics_artifact=str(_METRICS_PATH) if _METRICS_PATH.exists() else None,
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

    metrics_available = _METRICS_PATH.exists()
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
        metrics_available=metrics_available,
        metrics_artifact=str(_METRICS_PATH) if metrics_available else None,
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


# ── Fusion depuis le registre de greffons ────────────────────────────────────

def fuse_from_registry(
    results: dict,  # dict[str, EngineResult] — import local pour éviter circularité
    registry,       # PluginRegistry
    *,
    allow_experimental: bool | None = None,
) -> FusionResult:
    """Calcule la fusion à partir des résultats d'un PluginRegistry.

    Contrairement à fuse_scores(), cette fonction n'impose pas de noms de moteurs
    prédéfinis — elle utilise les poids et statuts déclarés dans le registre.

    Args:
        results: dict {engine_name: EngineResult} retourné par registry.run_all().
        registry: PluginRegistry avec les entrées enregistrées.
        allow_experimental: Si None, utilise settings.allow_experimental_engines.

    Returns:
        FusionResult — avec Verdict.abstain si aucun moteur actif.
    """
    if allow_experimental is None:
        allow_experimental = settings.allow_experimental_engines

    entries = registry.entries()
    active_names = registry.active_names(allow_experimental=allow_experimental)

    # Partition exclu / expérimental
    excluded_names = sorted(
        name for name in entries if name not in active_names
    )
    experimental_names = sorted(
        name for name in active_names
        if entries[name].status == "experimental"
    )

    # Abstention si aucun moteur actif
    if not active_names:
        component_scores = {
            name: round(r.score, 4) for name, r in results.items()
        }
        return FusionResult(
            final_score=0.0,
            verdict=Verdict.abstain,
            confidence_low=0.0,
            confidence_high=0.0,
            component_scores=component_scores,
            shap_ranking=[],
            plain_explanation=(
                "Aucun moteur de détection actif dans le registre. "
                "Enregistrer un moteur validated ou activer allow_experimental_engines "
                "(CLAUDE.md Règle 6 + Règle 10)."
            ),
            unvalidated_components=excluded_names,
            experimental_components=[],
            metrics_available=_METRICS_PATH.exists(),
            metrics_artifact=str(_METRICS_PATH) if _METRICS_PATH.exists() else None,
        )

    # Score pondéré — poids renormalisés sur les moteurs actifs
    valid_weight_sum = sum(
        entries[name].weight for name in active_names if name in entries
    ) or 1.0

    final = 0.0
    for name in active_names:
        r = results.get(name)
        score = r.score if (r and r.error is None) else 0.0
        w = entries[name].weight if name in entries else 0.0
        final += (w / valid_weight_sum) * score
    final = max(0.0, min(1.0, final))

    ic_low  = max(0.0, final - IC_HALF_WIDTH)
    ic_high = min(1.0, final + IC_HALF_WIDTH)

    if final < THRESHOLD_AUTHENTIC:
        verdict = Verdict.authentic
    elif final < THRESHOLD_DEEPFAKE:
        verdict = Verdict.undetermined
    else:
        verdict = Verdict.deepfake

    # Classement SHAP
    weighted_scores = {
        name: (entries[name].weight / valid_weight_sum) * (results[name].score if results.get(name) and not results[name].error else 0.0)
        for name in active_names
        if name in entries
    }
    total_w = sum(weighted_scores.values()) or 1.0
    shap_ranking = sorted(
        [
            {
                "feature":      name,
                "contribution": round(v / total_w * 100, 1),
                "raw_score":    round(results[name].score if results.get(name) else 0.0, 4),
                "weight":       round(entries[name].weight / valid_weight_sum, 4),
                "status":       entries[name].status,
            }
            for name, v in weighted_scores.items()
        ],
        key=lambda x: x["contribution"],
        reverse=True,
    )

    explanation = _generate_plain_explanation(
        verdict, final, shap_ranking, excluded_names, experimental_names
    )

    metrics_available = _METRICS_PATH.exists()
    return FusionResult(
        final_score=round(final, 4),
        verdict=verdict,
        confidence_low=round(ic_low, 4),
        confidence_high=round(ic_high, 4),
        component_scores={
            name: round(results[name].score if results.get(name) else 0.0, 4)
            for name in (*active_names, *excluded_names)
        },
        shap_ranking=shap_ranking,
        plain_explanation=explanation,
        unvalidated_components=excluded_names,
        experimental_components=experimental_names,
        metrics_available=metrics_available,
        metrics_artifact=str(_METRICS_PATH) if metrics_available else None,
    )
