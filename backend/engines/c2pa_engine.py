"""C2PAEngine — EnginePlugin adapter pour la vérification C2PA.

Signal de provenance non-ML : retourne un score binaire basé sur la présence
d'une assertion `c2pa.ai.generatedContent` dans le manifeste C2PA.

Différence fondamentale avec les moteurs ML :
  - Déterministe, pas de modèle, pas de poids
  - Score 1.0 = IA confirmée par le manifeste (assertion présente + manifeste valide)
  - Score 0.7 = manifeste présent mais invalide (possible falsification)
  - Score 0.0 = pas de manifeste, ou manifeste valide sans assertion IA
  - confidence = 0.0 si pas de manifeste (aucun signal — ne contribue pas à la fusion)

Enregistrer dans PluginRegistry avec status="experimental" tant que c2pa-python
n'est pas disponible en prod et que le comportement sur les formats edge-cases n'est
pas documenté dans evaluation/model_cards/.
"""
from __future__ import annotations

from pathlib import Path

from core.c2pa_verifier import C2PAVerifier
from engines.plugin import EnginePlugin, EngineResult


class C2PAEngine(EnginePlugin):
    """Moteur de provenance C2PA.

    Score :
      1.0  — c2pa.ai.generatedContent présent ET manifeste cryptographiquement valide
      0.7  — manifeste présent mais invalide/corrompu (signal de falsification)
      0.0  — manifeste valide sans assertion IA, ou pas de manifeste (confidence=0.0)
    """

    name = "c2pa"
    version = "1.0.0"

    def supports(self, media_type: str) -> bool:
        return media_type in ("image", "video", "audio", "document")

    def analyze(self, file_path: Path) -> EngineResult:
        try:
            result = C2PAVerifier.verify(file_path)
        except Exception as exc:  # noqa: BLE001
            return EngineResult(
                engine_name=self.name,
                score=0.0,
                confidence=0.0,
                error=str(exc),
                metadata={"has_manifest": False},
            )

        if not result.has_manifest:
            return EngineResult(
                engine_name=self.name,
                score=0.0,
                confidence=0.0,    # aucun signal — ne contribue pas au score de fusion
                metadata={"has_manifest": False},
            )

        if not result.is_cryptographically_valid:
            # Manifeste présent mais invalide — possible falsification
            return EngineResult(
                engine_name=self.name,
                score=0.7,
                confidence=0.8,
                metadata=result.to_dict(),
            )

        if result.is_ai_generated:
            return EngineResult(
                engine_name=self.name,
                score=1.0,
                confidence=1.0,    # preuve déterministe
                metadata=result.to_dict(),
            )

        # Manifeste valide, pas d'assertion IA → signal d'authenticité
        return EngineResult(
            engine_name=self.name,
            score=0.0,
            confidence=0.9,
            metadata=result.to_dict(),
        )
