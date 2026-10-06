"""Schéma Pydantic pour metrics.json — Brief v3 §0.5 + §7 (livrables par version)."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DatasetRef(BaseModel):
    name: str = Field(..., description="Identifiant unique du dataset (ex: FaceForensics++_v3)")
    version: str
    split: Literal["train", "val", "test"]
    n_samples: int = Field(..., gt=0)
    n_identities: int = Field(..., gt=0, description="Nombre de personnes/locuteurs uniques dans ce split")
    source_hash: str | None = Field(None, description="SHA-256 de la liste des fichiers (optionnel)")


class MetricsReport(BaseModel):
    """Rapport de métriques obligatoire avant toute publication de chiffres dans un rapport forensique.

    Une instance correspond exactement à un fichier metrics.json versionné et haché.
    Le SHA-256 du fichier est stocké dans ModelVersion.metrics_source_hash.
    """

    schema_version: str = "1.0"
    engine_name: str = Field(
        ...,
        description="Nom de la composante (texture, temporal, rppg, biometrics, audio, phase, metadata)",
    )
    model_version: str = Field(..., description="Version du modèle évalué (ex: v0.1.0)")
    protocol_hash: str = Field(
        ..., description="SHA-256 du TrainingProtocol figé utilisé pour cette évaluation"
    )

    # Datasets — la règle §0.4 est vérifiée ici
    train_dataset: DatasetRef
    test_dataset: DatasetRef

    # Métriques mesurées sur le jeu de TEST uniquement
    far: float = Field(..., ge=0.0, le=1.0, description="False Acceptance Rate")
    frr: float = Field(..., ge=0.0, le=1.0, description="False Rejection Rate")
    eer: float = Field(..., ge=0.0, le=1.0, description="Equal Error Rate")
    auc: float = Field(..., ge=0.0, le=1.0, description="Aire sous la courbe ROC")
    threshold_used: float = Field(
        ..., ge=0.0, le=1.0, description="Seuil de décision utilisé pour FAR/FRR"
    )

    # Contexte
    evaluated_at: datetime
    evaluator: str = Field(
        ..., description="Nom/identifiant de la personne qui a conduit l'évaluation"
    )
    evaluation_environment: str | None = Field(
        None, description="GPU, Python, PyTorch, OS — pour reproductibilité"
    )
    notes: str | None = None

    @model_validator(mode="after")
    def check_cross_dataset(self) -> "MetricsReport":
        """Brief v3 §0.4 : évaluation inter-jeux obligatoire."""
        if self.train_dataset.name == self.test_dataset.name:
            raise ValueError(
                "Évaluation inter-jeux requise (Brief v3 §0.4) : "
                "train_dataset.name et test_dataset.name doivent être différents. "
                f"Reçu : '{self.train_dataset.name}' pour les deux."
            )
        if self.train_dataset.split == "test":
            raise ValueError("Le jeu d'entraînement ne doit pas avoir split='test'.")
        return self

    def to_file(self, path: Path) -> str:
        """Sérialise le rapport vers path et retourne son SHA-256.

        Le SHA-256 retourné doit être stocké dans ModelVersion.metrics_source_hash.
        """
        content = self.model_dump_json(indent=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def from_file(cls, path: Path) -> tuple["MetricsReport", str]:
        """Désérialise et retourne (rapport, sha256_actuel).

        Comparer sha256_actuel avec ModelVersion.metrics_source_hash pour vérifier l'intégrité.
        """
        content = path.read_text(encoding="utf-8")
        sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return cls.model_validate_json(content), sha256
