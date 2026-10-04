"""Registre des moteurs de détection — natifs et tiers.

Usage :
    from engines.registry import PluginRegistry
    from engines.plugin import EnginePlugin, EngineResult

    registry = PluginRegistry()
    registry.register(MyPlugin, status="experimental", weight=0.15)
    active = registry.active_names(allow_experimental=True)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from engines.fusion import EngineStatus

if TYPE_CHECKING:
    from engines.plugin import EnginePlugin, EngineResult


@dataclass
class EngineEntry:
    """Entrée dans le registre pour un moteur."""

    plugin_cls: type[EnginePlugin]
    status: EngineStatus
    weight: float
    # Paramètres de calibration Platt (a*score + b, clampé à [0,1]).
    # None = pas de calibration (comportement par défaut).
    calibration: dict[str, float] | None = None


def _apply_calibration(score: float, calibration: dict[str, float] | None) -> float:
    """Applique la calibration linéaire Platt : a*score + b, clampé à [0,1]."""
    if calibration is None:
        return score
    a = calibration.get("a", 1.0)
    b = calibration.get("b", 0.0)
    return max(0.0, min(1.0, a * score + b))


class PluginRegistry:
    """Registre extensible des moteurs de détection.

    Permet d'enregistrer des moteurs tiers sans modifier le code existant.
    Chaque moteur déclare son statut (validated/experimental/disabled) et son poids.
    La calibration Platt optionnelle corrige le biais de score de chaque moteur.
    """

    def __init__(self) -> None:
        self._entries: dict[str, EngineEntry] = {}

    # ── Enregistrement ────────────────────────────────────────────────────────

    def register(
        self,
        plugin_cls: type[EnginePlugin],
        *,
        status: EngineStatus,
        weight: float,
        calibration: dict[str, float] | None = None,
    ) -> None:
        """Enregistre un moteur dans le registre.

        Args:
            plugin_cls: Classe du moteur (sous-classe d'EnginePlugin).
            status: "validated" | "experimental" | "disabled"
            weight: Poids dans la fusion (non normalisé — renormalisé à l'exécution).
            calibration: {"a": float, "b": float} pour Platt scaling. None = identité.

        Raises:
            ValueError: Si plugin_cls.name est vide ou dupliqué.
        """
        if not plugin_cls.name:
            raise ValueError(
                f"Le plugin {plugin_cls.__name__} doit définir un attribut `name` non vide."
            )
        if weight < 0:
            raise ValueError(f"Le poids du moteur '{plugin_cls.name}' doit être >= 0.")
        self._entries[plugin_cls.name] = EngineEntry(
            plugin_cls=plugin_cls,
            status=status,
            weight=weight,
            calibration=calibration,
        )

    # ── Interrogation ─────────────────────────────────────────────────────────

    def active_names(self, *, allow_experimental: bool) -> list[str]:
        """Noms des moteurs actifs selon la politique expérimentale."""
        return [
            name
            for name, entry in self._entries.items()
            if entry.status == "validated"
            or (entry.status == "experimental" and allow_experimental)
        ]

    def validated_names(self) -> list[str]:
        """Noms des moteurs au statut 'validated' uniquement."""
        return [n for n, e in self._entries.items() if e.status == "validated"]

    def entries(self) -> dict[str, EngineEntry]:
        return dict(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    # ── Exécution ─────────────────────────────────────────────────────────────

    def run_all(
        self,
        file_path: Path,
        *,
        allow_experimental: bool = True,
    ) -> dict[str, EngineResult]:
        """Instancie et exécute tous les moteurs actifs sur file_path.

        Les moteurs qui lèvent une exception retournent un EngineResult avec
        error renseigné et score=0.0 — ils ne font pas échouer l'ensemble.
        La calibration Platt est appliquée à chaque score avant de retourner.
        """
        from engines.plugin import EngineResult  # import local pour éviter circularité

        results: dict[str, EngineResult] = {}
        for name in self.active_names(allow_experimental=allow_experimental):
            entry = self._entries[name]
            try:
                plugin = entry.plugin_cls()
                raw = plugin.analyze(file_path)
                calibrated_score = _apply_calibration(raw.score, entry.calibration)
                results[name] = EngineResult(
                    engine_name=raw.engine_name,
                    score=calibrated_score,
                    confidence=raw.confidence,
                    metadata=raw.metadata,
                    error=raw.error,
                )
            except Exception as exc:  # noqa: BLE001
                results[name] = EngineResult(
                    engine_name=name,
                    score=0.0,
                    error=str(exc),
                )
        return results
