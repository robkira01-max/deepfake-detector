"""Porte de validation des moteurs — CLAUDE.md Règle 10.

Un engine ne peut passer à 'validated' que si les 4 conditions sont réunies :
  1. evaluation/metrics/{engine_name}.json (ou metrics.json en fallback) valide
  2. evaluation/model_cards/{engine_name}.md existe et est substantielle
  3. evaluation/protocols/baseline_protocol.yaml a status != 'draft'
  4. AuditLog contient une entrée ENGINE_STATUS_CHANGED action=approved pour cet engine
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_EVAL_ROOT = Path(__file__).resolve().parents[2] / "evaluation"
_METRICS_DIR = _EVAL_ROOT / "metrics"
_METRICS_PATH = _METRICS_DIR / "metrics.json"          # legacy — metadata engine
_PROTOCOL_PATH = _EVAL_ROOT / "protocols" / "baseline_protocol.yaml"
_MODEL_CARDS_DIR = _EVAL_ROOT / "model_cards"


def _metrics_path_for(engine_name: str) -> Path:
    """Retourne le chemin du fichier de métriques pour un engine donné.

    Lookup order:
      1. evaluation/metrics/{engine_name}.json  (fichier per-engine)
      2. evaluation/metrics/metrics.json        (legacy — metadata engine)
    """
    per_engine = _METRICS_DIR / f"{engine_name}.json"
    if per_engine.exists():
        return per_engine
    return _METRICS_PATH

VALID_ENGINES = frozenset({
    "texture", "temporal", "rppg", "biometrics",
    "audio", "phase", "metadata",
})

_METRICS_REQUIRED_FIELDS = ("far", "frr", "eer", "auc", "engine_name")


@dataclass
class ValidationGateResult:
    engine_name: str
    metrics_ok: bool
    model_card_ok: bool
    protocol_ok: bool
    human_approval_ok: bool
    blocking_reasons: list[str] = field(default_factory=list)
    metrics_path: str | None = None
    model_card_path: str | None = None
    can_promote: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.can_promote = (
            self.metrics_ok
            and self.model_card_ok
            and self.protocol_ok
            and self.human_approval_ok
        )

    def to_dict(self) -> dict:
        return {
            "engine_name": self.engine_name,
            "can_promote": self.can_promote,
            "conditions": {
                "metrics_ok": self.metrics_ok,
                "model_card_ok": self.model_card_ok,
                "protocol_ok": self.protocol_ok,
                "human_approval_ok": self.human_approval_ok,
            },
            "blocking_reasons": self.blocking_reasons,
            "metrics_path": self.metrics_path,
            "model_card_path": self.model_card_path,
        }


class EngineValidationGate:
    """Vérifie les 4 conditions de la Règle 10 pour un moteur donné."""

    @staticmethod
    def check(engine_name: str, db=None) -> ValidationGateResult:
        """
        Args:
            engine_name: nom du moteur (texture, temporal, rppg, biometrics, audio, phase, metadata)
            db: session SQLAlchemy (requis pour la condition 4 — approbation humaine)
        """
        reasons: list[str] = []

        metrics_ok, metrics_path = _check_metrics(engine_name, reasons)
        model_card_ok, model_card_path = _check_model_card(engine_name, reasons)
        protocol_ok = _check_protocol(reasons)
        human_approval_ok = _check_human_approval(engine_name, db, reasons)

        return ValidationGateResult(
            engine_name=engine_name,
            metrics_ok=metrics_ok,
            model_card_ok=model_card_ok,
            protocol_ok=protocol_ok,
            human_approval_ok=human_approval_ok,
            blocking_reasons=reasons,
            metrics_path=str(metrics_path) if metrics_path else None,
            model_card_path=str(model_card_path) if model_card_ok and model_card_path else None,
        )


def _check_metrics(engine_name: str, reasons: list[str]) -> tuple[bool, Path | None]:
    """Condition 1 : fichier de métriques valide pour cet engine.

    Cherche evaluation/metrics/{engine_name}.json en premier,
    puis evaluation/metrics/metrics.json (compat. legacy).
    """
    path = _metrics_path_for(engine_name)
    if not path.exists():
        reasons.append(
            f"Fichier de métriques absent pour '{engine_name}'. "
            f"Attendu : evaluation/metrics/{engine_name}.json "
            "(ou evaluation/metrics/metrics.json en fallback). "
            "Voir evaluation/protocols/baseline_protocol.yaml pour la procédure."
        )
        return False, None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        reasons.append(f"{path.name} illisible : {exc}")
        return False, None

    if data.get("engine_name") != engine_name:
        reasons.append(
            f"{path.name} référence l'engine '{data.get('engine_name')}', "
            f"pas '{engine_name}'."
        )
        return False, path

    for required in _METRICS_REQUIRED_FIELDS:
        if data.get(required) is None:
            reasons.append(f"{path.name} manque le champ requis '{required}'.")
            return False, path

    return True, path


def _check_model_card(engine_name: str, reasons: list[str]) -> tuple[bool, Path | None]:
    """Condition 2 : model card substantielle présente."""
    card = _MODEL_CARDS_DIR / f"{engine_name}.md"
    if not card.exists():
        reasons.append(
            f"Model card absente : evaluation/model_cards/{engine_name}.md. "
            "Voir evaluation/model_cards/template.md."
        )
        return False, None

    content = card.read_text(encoding="utf-8").strip()
    if len(content) < 200:
        reasons.append(
            f"Model card trop courte ({len(content)} chars). "
            "Elle doit documenter l'architecture, les données et les limites (min 200 chars)."
        )
        return False, card

    return True, card


def _check_protocol(reasons: list[str]) -> bool:
    """Condition 3 : baseline_protocol.yaml marqué 'completed' (pas 'draft')."""
    if not _PROTOCOL_PATH.exists():
        reasons.append("evaluation/protocols/baseline_protocol.yaml absent.")
        return False

    try:
        import yaml  # noqa: PLC0415
        data = yaml.safe_load(_PROTOCOL_PATH.read_text(encoding="utf-8"))
        proto_status = data.get("status", "draft")
        if proto_status == "draft":
            reasons.append(
                "baseline_protocol.yaml a status: 'draft'. "
                "Un humain doit l'appliquer et mettre status: 'completed'."
            )
            return False
        return True
    except Exception as exc:
        reasons.append(f"Erreur lecture baseline_protocol.yaml : {exc}")
        return False


def _check_human_approval(engine_name: str, db, reasons: list[str]) -> bool:
    """Condition 4 : approbation humaine journalisée dans AuditLog."""
    if db is None:
        reasons.append(
            "Session DB requise pour vérifier l'approbation humaine (condition 4)."
        )
        return False

    try:
        from models.audit_log import AuditAction, AuditLog  # noqa: PLC0415
        entries = (
            db.query(AuditLog)
            .filter(AuditLog.action == AuditAction.ENGINE_STATUS_CHANGED)
            .all()
        )
        for entry in entries:
            details = entry.details or {}
            if (
                details.get("engine_name") == engine_name
                and details.get("action") == "approved"
            ):
                return True
        reasons.append(
            f"Aucune approbation humaine pour '{engine_name}' dans l'AuditLog. "
            "Appeler POST /models/engines/{name}/approve d'abord."
        )
        return False
    except Exception as exc:
        reasons.append(f"Erreur vérification AuditLog : {exc}")
        return False
