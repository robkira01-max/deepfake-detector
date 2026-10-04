# Model Card — Moteur de métadonnées (metadata)

**Version :** 1.0.0  
**Type :** Heuristique — aucun modèle neuronal  
**Statut :** experimental (pas encore validated — voir procédure ci-dessous)  
**Dernière mise à jour :** 2026-10-04  

---

## 1. Description

Le moteur `metadata` analyse les métadonnées EXIF/XMP/ID3 d'un fichier média pour détecter des signatures caractéristiques des outils de génération deepfake (DeepFaceLab, FaceSwap, FaceApp, Avatarify, etc.).

Ce moteur est **déterministe et heuristique** : il ne repose pas sur un réseau de neurones, mais sur une liste de règles.  
Il constitue un signal complémentaire, pas un détecteur principal.

---

## 2. Architecture

Aucun modèle ML. Règles codées dans `backend/engines/metadata_engine.py` :

| Catégorie | Signal | Score |
|-----------|--------|-------|
| Logiciel deepfake détecté | Champ `Software` ou `Creator_Tool` contient "DeepFaceLab", "FaceSwap", "Avatarify", etc. | 0.95 |
| Compression agressive post-traitement | Ratio bitrate/résolution anormalement bas | 0.40–0.60 |
| Timestamps incohérents | `DateTimeOriginal` > `DateTime` (modification post-capture) | 0.30 |
| Métadonnées absentes sur format qui les requiert | JPEG sans EXIF | 0.10 |
| Création et modification simultanées | Δ(create, modify) < 1s pour une vidéo > 1 Mo | 0.20 |

Score final = max des signaux individuels détectés, pondéré par la confiance.

---

## 3. Données d'entraînement

**Aucune.** Règles écrites manuellement sur la base de la documentation des outils deepfake et des observations forensiques.

Sources de référence :
- Documentation DeepFaceLab (GitHub deepfakes/DeepFaceLab)
- Signatures ExifTool connues pour les outils de synthèse
- Littérature forensique (IEEE TIFS, MediaForensics)

---

## 4. Données d'évaluation requises (avant promotion à `validated`)

Pour satisfaire la Règle 10, une évaluation sur jeu de test indépendant est requise :

| Paramètre | Valeur cible |
|-----------|-------------|
| Taille minimale jeu de test | 500 vrais + 500 deepfakes |
| Jeu d'entraînement | N/A (pas de ML) — utiliser FaceForensics++ comme référence |
| Jeu de test | Différent du jeu de référence (§0.4) |
| Conditions de test | Compression sociale (YouTube, WhatsApp), basse résolution |

---

## 5. Limites connues

- **Facilement contournable** : un attaquant peut effacer ou forger les métadonnées.
- **Faible rappel** : la plupart des deepfakes bien construits ne portent pas de signatures de logiciels.
- **Faux positifs** : logiciels de retouche légitimes (Premiere, DaVinci) peuvent déclencher le signal de modification.
- **Non applicable aux flux en direct** : pas de métadonnées EXIF en temps réel.
- Ce moteur ne peut pas, seul, conclure qu'un fichier est authentique ou un deepfake.

---

## 6. Utilisation dans la fusion

Poids dans l'ensemble : **7%** (`WEIGHTS["metadata"] = 0.07` dans `engines/fusion.py`).

Ce faible poids reflète la contournabilité du signal. Il est utilisé comme signal d'alerte secondaire.

---

## 7. Procédure de validation (Règle 10)

Pour passer de `experimental` à `validated` :

1. [ ] Produire `evaluation/metrics/metrics.json` avec `engine_name: "metadata"`, sur jeu de test indépendant
2. [ ] Ce fichier de model card doit être présent (déjà fait)
3. [ ] Mettre `status: "completed"` dans `evaluation/protocols/baseline_protocol.yaml` après application du protocole
4. [ ] Un admin appelle `POST /models/engines/metadata/approve` avec ses notes de revue
5. [ ] Un admin appelle `POST /models/engines/metadata/promote`

---

## 8. Responsabilité

Ce moteur assiste l'expert humain. Il ne produit pas de conclusion. Toute utilisation dans un contexte judiciaire requiert une supervision et une validation par un expert en forensique numérique.

> *Conçu pour soutenir l'évaluation de l'intégrité, de la provenance et de l'authenticité technique de contenus numériques, sans constituer à lui seul une conclusion d'expert ou une preuve définitive de manipulation.*
