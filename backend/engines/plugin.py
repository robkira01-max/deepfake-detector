"""Plugin API — interface standard pour les moteurs de détection tiers.

Pour enregistrer un moteur tiers dans PluginRegistry :
    from engines.plugin import EnginePlugin, EngineResult
    from engines.registry import default_registry

    class MyPlugin(EnginePlugin):
        name = "my_plugin"
        version = "1.0.0"

        def analyze(self, file_path: Path) -> EngineResult:
            score = ...  # logique de détection
            return EngineResult(engine_name=self.name, score=score)

    default_registry.register(MyPlugin, status="experimental", weight=0.15)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EngineResult:
    """Résultat brut d'un moteur de détection."""

    engine_name: str
    score: float                            # 0.0–1.0, probabilité de manipulation
    confidence: float = 1.0                 # 0.0–1.0, confiance du moteur dans son score
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None                # non-None si le moteur a échoué


class EnginePlugin(ABC):
    """Classe de base abstraite pour les moteurs de détection.

    Surcharger `name` (str), `version` (str) et `analyze()`.
    `supports()` peut être surchargée pour restreindre le moteur à certains types de médias.
    """

    name: str = ""
    version: str = "0.0.0"

    @abstractmethod
    def analyze(self, file_path: Path) -> EngineResult:
        """Analyse un fichier et retourne un score de manipulation.

        Contrat :
        - score doit être dans [0.0, 1.0]
        - en cas d'erreur : retourner EngineResult(engine_name=..., score=0.0, error="message")
          plutôt que de lever une exception (sauf pour les erreurs de programmation)
        """
        ...

    def supports(self, media_type: str) -> bool:
        """Retourne True si ce moteur supporte le type de média donné.

        Surcharger pour restreindre le moteur à video, audio, image, document, etc.
        """
        return True
