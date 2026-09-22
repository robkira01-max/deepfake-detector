# ADR-003 — Moteurs ML : ONNX Runtime + PyTorch CPU

**Status** : Accepted  
**Date** : 2026-09-21  
**Auteur** : Ruflo Agent (ml-engineer)

## Contexte

DeepfakeDetector tourne sur Kali Linux sans GPU (VirtualBox). Les modèles ML doivent être optimisés pour CPU.

## Décision

- **ONNX Runtime** pour l'inférence (plus rapide que PyTorch natif sur CPU)
- **PyTorch CPU** (`+cpu`) pour le fine-tuning et preprocessing
- **MediaPipe** pour la détection de visage (optimisé mobile/CPU)
- **librosa + torchaudio** pour l'analyse audio
- **Fusion pondérée** vidéo 60% + audio 40% pour le score final

## Architecture des moteurs

```python
# video_engine.py — Pipeline vidéo
MediaPipe FaceMesh → Extraction frames → ONNX Inference → Score par frame → Agrégation

# audio_engine.py — Pipeline audio  
librosa MFCC → rPPG extraction → ONNX/torch → Score de probabilité

# fusion.py — Fusion finale
score_final = 0.6 * score_video + 0.4 * score_audio
```

## Modèles prioritaires

| Modèle | Format | Usage |
|--------|--------|-------|
| EfficientNet-B4 FaceForensics++ | ONNX | Deepfake vidéo principal |
| XceptionNet | ONNX | Artifacts compression |
| Wav2Vec2 | HuggingFace → ONNX | Voix synthétique |
| RawNet2 | ONNX | Anti-spoofing audio |

## Contraintes de performance (CPU)

- Latence max vidéo : 30s pour 1 min de vidéo
- Latence max audio : 10s pour 1 min d'audio
- RAM max par analyse : 4GB
- Traitement par chunks (pas de chargement complet en RAM)

## Conséquences

- Les analyses ML sont exécutées en tâches Celery (async, pas de timeout HTTP)
- Le résultat inclut TOUJOURS : score, confiance, modèle utilisé, version
- Les bounding boxes des zones suspectes sont incluses dans le rapport
- Les modèles ONNX sont chargés une fois au démarrage (pas à chaque requête)

## Fichiers liés

- `backend/engines/video_engine.py`
- `backend/engines/audio_engine.py`
- `backend/engines/fusion.py`
- `backend/tasks/analysis_tasks.py`
