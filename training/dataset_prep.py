"""Utilitaire de split de dataset avec séparation stricte par identité — Brief v3 §0.3."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SplitResult:
    train: list[str]            # chemins/IDs de fichiers
    val:   list[str]
    test:  list[str]
    identity_map: dict[str, str]  # fichier → identité
    split_hash: str               # SHA-256 déterministe de l'affectation


def split_by_identity(
    samples: list[dict[str, Any]],
    identity_key: str = "identity_id",
    path_key: str = "path",
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> SplitResult:
    """Divise un dataset en train/val/test avec séparation stricte par identité.

    Une même identité n'apparaît que dans UN SEUL split — évite la fuite de données
    (data leakage) qui produirait des métriques gonflées (Brief v3 §0.3).

    Args:
        samples:      Liste de dicts avec au minimum {identity_key, path_key}
        identity_key: Clé du champ identifiant la personne/locuteur
        path_key:     Clé du champ chemin/ID du fichier
        ratios:       (train, val, test) — doivent sommer à 1.0
        seed:         Graine aléatoire pour la reproductibilité

    Raises:
        ValueError: Identité manquante, ratios invalides, dataset trop petit.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Les ratios doivent sommer à 1.0, reçu {sum(ratios):.4f}")
    if any(r < 0 for r in ratios):
        raise ValueError("Tous les ratios doivent être positifs")

    identity_to_files: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        iid = sample.get(identity_key)
        path = sample.get(path_key)
        if iid is None:
            raise ValueError(f"Champ identité '{identity_key}' manquant dans : {sample}")
        if path is None:
            raise ValueError(f"Champ chemin '{path_key}' manquant dans : {sample}")
        identity_to_files[str(iid)].append(str(path))

    identities = sorted(identity_to_files.keys())
    if len(identities) < 3:
        raise ValueError(
            f"Au moins 3 identités requises pour train/val/test, reçu {len(identities)}"
        )

    rng = random.Random(seed)
    shuffled = list(identities)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, round(n * ratios[0]))
    n_val   = max(1, round(n * ratios[1]))
    n_test  = n - n_train - n_val
    if n_test < 1:
        raise ValueError(
            f"Trop peu d'identités ({n}) pour créer 3 splits non vides "
            f"avec ratios {ratios}."
        )

    train_ids = set(shuffled[:n_train])
    val_ids   = set(shuffled[n_train:n_train + n_val])

    train_files, val_files, test_files = [], [], []
    identity_map: dict[str, str] = {}

    for iid, files in identity_to_files.items():
        for f in files:
            identity_map[f] = iid
        if iid in train_ids:
            train_files.extend(files)
        elif iid in val_ids:
            val_files.extend(files)
        else:
            test_files.extend(files)

    assignment = {
        "train":  sorted(train_files),
        "val":    sorted(val_files),
        "test":   sorted(test_files),
        "seed":   seed,
        "ratios": list(ratios),
    }
    split_hash = hashlib.sha256(
        json.dumps(assignment, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return SplitResult(
        train=train_files,
        val=val_files,
        test=test_files,
        identity_map=identity_map,
        split_hash=split_hash,
    )


def save_split(result: SplitResult, output_dir: Path) -> Path:
    """Sauvegarde le split dans output_dir/split.json et retourne le chemin."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "split.json"
    payload = {
        "split_hash": result.split_hash,
        "train":      result.train,
        "val":        result.val,
        "test":       result.test,
        "n_train":    len(result.train),
        "n_val":      len(result.val),
        "n_test":     len(result.test),
    }
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return split_path
