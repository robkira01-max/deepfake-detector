"""
Pré-téléchargement des modèles ML — à exécuter une fois avant le premier démarrage.

Usage :
    source .venv/bin/activate
    python backend/scripts/preload_models.py

Ce script télécharge dans le cache local tous les poids nécessaires pour que
le premier démarrage en production n'exige pas d'accès Internet.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Pointer le cache HF sur le dossier projet
os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent.parent / "data" / "models"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])

MODELS = [
    # Texte IA (RoBERTa)
    ("transformers", "roberta-base",           "RoBERTa base (détection texte IA)"),
    # Audio (Wav2Vec2)
    ("transformers", "facebook/wav2vec2-base", "Wav2Vec2 base (analyse audio)"),
    # Vision (EfficientNet-B4, ResNet-50 via torchvision — téléchargés à la demande)
    ("torchvision",  "efficientnet_b4",        "EfficientNet-B4 (vidéo texture)"),
    ("torchvision",  "resnet50",               "ResNet-50 (vidéo temporel)"),
]


def _banner(msg: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print('─' * 60)


def download_transformers(model_id: str, label: str) -> bool:
    try:
        from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
        _banner(f"Téléchargement : {label}")
        print(f"  Repo : {model_id}")
        t0 = time.time()
        try:
            AutoTokenizer.from_pretrained(model_id)
        except Exception:
            pass
        try:
            AutoFeatureExtractor.from_pretrained(model_id)
        except Exception:
            pass
        AutoModel.from_pretrained(model_id)
        print(f"  ✅ Terminé en {time.time() - t0:.1f}s")
        return True
    except Exception as exc:
        print(f"  ⚠️  Échec ({exc}) — le modèle sera téléchargé au premier appel")
        return False


def download_torchvision(model_name: str, label: str) -> bool:
    try:
        import torchvision.models as tvm
        _banner(f"Téléchargement : {label}")
        t0 = time.time()
        if model_name == "efficientnet_b4":
            tvm.efficientnet_b4(weights=tvm.EfficientNet_B4_Weights.DEFAULT)
        elif model_name == "resnet50":
            tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        print(f"  ✅ Terminé en {time.time() - t0:.1f}s")
        return True
    except Exception as exc:
        print(f"  ⚠️  Échec ({exc}) — le modèle sera téléchargé au premier appel")
        return False


def download_insightface() -> bool:
    try:
        import insightface
        from insightface.app import FaceAnalysis
        _banner("Téléchargement : insightface buffalo_s (KYC face matching)")
        t0 = time.time()
        app = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"],
            root=os.environ["HF_HOME"],
        )
        app.prepare(ctx_id=0, det_size=(320, 320))
        print(f"  ✅ Terminé en {time.time() - t0:.1f}s")
        return True
    except Exception as exc:
        print(f"  ⚠️  Échec ({exc}) — modèle téléchargé au premier appel KYC")
        return False


def main() -> None:
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  DeepfakeDetector — Pré-chargement des modèles ML   ║")
    print("║  v3.1.0 — Cache : data/models/                      ║")
    print("╚══════════════════════════════════════════════════════╝")

    cache_dir = Path(os.environ["HF_HOME"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Cache : {cache_dir}")

    results: list[tuple[str, bool]] = []

    for backend, model_id, label in MODELS:
        if backend == "transformers":
            ok = download_transformers(model_id, label)
        else:
            ok = download_torchvision(model_id, label)
        results.append((label, ok))

    results.append(("insightface buffalo_s (KYC)", download_insightface()))

    print("\n\n╔══════════════════════════════════════════════════════╗")
    print("║  Résumé                                              ║")
    print("╠══════════════════════════════════════════════════════╣")
    for label, ok in results:
        status = "✅" if ok else "⚠️ "
        print(f"║  {status}  {label[:48]:<48}  ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    failed = [l for l, ok in results if not ok]
    if failed:
        print("Les modèles manquants seront téléchargés automatiquement")
        print("au premier appel — assurez-vous d'avoir Internet à ce moment.\n")
        sys.exit(0)
    else:
        print("Tous les modèles sont en cache. Démarrage hors-ligne possible.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
