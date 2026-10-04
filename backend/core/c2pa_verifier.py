"""Vérification structurée des manifestes C2PA (Content Credentials).

C2PA est une preuve de provenance cryptographique — pas une détection ML.
Un manifeste valide d'une caméra est une forte indication d'authenticité.
Un manifeste avec `c2pa.ai.generatedContent` confirme une origine IA.
L'absence de manifeste ne prouve rien dans un sens ou dans l'autre.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog

from core.c2pa_handler import C2PAHandler

log = structlog.get_logger(__name__)

# Labels C2PA normalisés (C2PA-1.3 spec §9)
_LABEL_AI_GENERATED      = "c2pa.ai.generatedContent"
_LABEL_TRAINING_MINING   = "c2pa.training-mining"
_LABEL_ACTIONS           = "c2pa.actions"
_PREFIX_HASH_HARD        = "c2pa.hash."
_PREFIX_HASH_SOFT        = "c2pa.soft-binding."


@dataclass
class C2PAVerificationResult:
    """Résultat structuré d'une vérification C2PA.

    Séparer clairement les champs cryptographiques (vérifiables de façon déterministe)
    des champs déclaratifs (renseignés par l'auteur du manifeste — non vérifiables seuls).
    """

    # ── Présence et validité ──────────────────────────────────────────────────
    has_manifest: bool
    is_cryptographically_valid: bool | None     # None = bibliothèque indisponible ou erreur

    # ── Provenance déclarative ────────────────────────────────────────────────
    producer: str | None                        # outil ou organisation déclarant le contenu
    tool: str | None                            # logiciel ayant créé le manifeste
    created_at: str | None                      # date de création déclarée (ISO-8601)

    # ── Signaux IA ────────────────────────────────────────────────────────────
    # True = assertion c2pa.ai.generatedContent présente dans le manifeste actif.
    # Ce champ est déclaratif : un outil non conforme peut omettre cette assertion.
    is_ai_generated: bool | None

    # True = assertion c2pa.training-mining avec opt-out présente.
    ai_training_opted_out: bool | None

    # ── Intégrité ─────────────────────────────────────────────────────────────
    hard_bindings_count: int                    # nb d'assertions c2pa.hash.* (liaisons fortes)
    soft_bindings_count: int                    # nb d'assertions c2pa.soft-binding.*
    actions: list[str] = field(default_factory=list)   # liste des types d'action déclarés
    validation_errors: list[str] = field(default_factory=list)
    raw_assertions_count: int = 0

    def to_dict(self) -> dict:
        return {
            "has_manifest": self.has_manifest,
            "is_cryptographically_valid": self.is_cryptographically_valid,
            "producer": self.producer,
            "tool": self.tool,
            "created_at": self.created_at,
            "is_ai_generated": self.is_ai_generated,
            "ai_training_opted_out": self.ai_training_opted_out,
            "hard_bindings_count": self.hard_bindings_count,
            "soft_bindings_count": self.soft_bindings_count,
            "actions": self.actions,
            "validation_errors": self.validation_errors,
            "raw_assertions_count": self.raw_assertions_count,
        }


_NO_MANIFEST = C2PAVerificationResult(
    has_manifest=False,
    is_cryptographically_valid=None,
    producer=None,
    tool=None,
    created_at=None,
    is_ai_generated=None,
    ai_training_opted_out=None,
    hard_bindings_count=0,
    soft_bindings_count=0,
)


class C2PAVerifier:
    """Vérification déterministe des manifestes C2PA.

    Utilise C2PAHandler pour la lecture bas niveau.
    Produit C2PAVerificationResult avec les champs normalisés.
    """

    @staticmethod
    def verify(file_path: str | Path) -> C2PAVerificationResult:
        """Lit et interprète le manifeste C2PA d'un fichier.

        Retourne _NO_MANIFEST si aucun manifeste n'est trouvé ou si la
        bibliothèque c2pa n'est pas installée.
        """
        manifest = C2PAHandler.read_manifest(file_path)
        if not manifest:
            return _NO_MANIFEST

        assertions = C2PAHandler.extract_assertions(manifest)
        summary = C2PAHandler.get_provenance_summary(manifest)

        # Validité cryptographique
        is_valid = summary.get("is_valid")

        # Liaisons fortes et souples
        hard = sum(1 for a in assertions if a.get("label", "").startswith(_PREFIX_HASH_HARD))
        soft = sum(1 for a in assertions if a.get("label", "").startswith(_PREFIX_HASH_SOFT))

        # Signal IA — assertion déclarative
        is_ai_generated = any(
            a.get("label") == _LABEL_AI_GENERATED for a in assertions
        ) or None
        if not any(a.get("label") == _LABEL_AI_GENERATED for a in assertions):
            is_ai_generated = False

        # Opt-out entraînement IA
        ai_training_opted_out: bool | None = None
        for a in assertions:
            if a.get("label") == _LABEL_TRAINING_MINING:
                entries = a.get("data", {}).get("entries", [])
                if isinstance(entries, list):
                    ai_training_opted_out = any(
                        e.get("use") in ("notAllowed", "constrained")
                        for e in entries
                        if isinstance(e, dict)
                    )
                break

        # Actions déclarées
        actions: list[str] = []
        for a in assertions:
            if a.get("label") == _LABEL_ACTIONS:
                acts = a.get("data", {}).get("actions", [])
                if isinstance(acts, list):
                    actions.extend(
                        str(act.get("action", "")) for act in acts if isinstance(act, dict)
                    )

        return C2PAVerificationResult(
            has_manifest=True,
            is_cryptographically_valid=is_valid,
            producer=summary.get("producer"),
            tool=summary.get("tool"),
            created_at=summary.get("created_at"),
            is_ai_generated=is_ai_generated,
            ai_training_opted_out=ai_training_opted_out,
            hard_bindings_count=hard,
            soft_bindings_count=soft,
            actions=actions,
            validation_errors=summary.get("validation_errors", []),
            raw_assertions_count=len(assertions),
        )
