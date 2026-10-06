# Model Card — Moteur Biométrie comportementale (biometrics)

**Version :** 1.0.0  
**Type :** Algorithmique — Eye Aspect Ratio (EAR) + détection de clignements  
**Statut :** experimental (pas encore validated — voir procédure ci-dessous)  
**Dernière mise à jour :** 2026-10-06  

---

## 1. Description

Le moteur `biometrics` détecte les anomalies comportementales du clignement des yeux dans une vidéo.

**Principe** : un visage réel cligne des yeux à un rythme physiologique normal de 15-20 fois par minute. Les deepfakes générés par les premiers GANs et réseaux de diffusion présentent un clignement anormalement rare (< 7/min), car les modèles entraînés sur des images fixes ou peu longues ne capturent pas correctement cette dynamique. Le moteur calcule le ratio Eye Aspect Ratio (EAR) par frame et compte les transitions sous le seuil EAR=0.2 pour estimer le taux de clignement.

---

## 2. Architecture

### Formule EAR (Soukupová & Čech, 2016)

```
        ||p2 - p6|| + ||p3 - p5||
EAR = ─────────────────────────────
              2 × ||p1 - p4||
```

Où p1…p6 sont les 6 landmarks de l'œil (MediaPipe FaceMesh), p1/p4 = coins horizontal, p2/p3/p5/p6 = points verticaux.

**Pipeline dans VideoEngine._analyze_biometrics :**

1. **Extraction des landmarks** : MediaPipe FaceMesh 468 points par frame
2. **Calcul EAR** : pour l'œil gauche (indices [362, 385, 387, 263, 373, 380]) et droit (indices [33, 160, 158, 133, 153, 144]), moyenner les deux
3. **Comptage** : `_count_blinks(ear_array, threshold=0.2)` — machine à états, compte les passages sous 0.2
4. **Taux** : `blink_rate_per_min = (blinks / duration_sec) × 60`
5. **Score** (déterministe) :

| Taux de clignement | Score | Interprétation |
|--------------------|-------|----------------|
| < 3/min            | 0.85  | Deepfake probable (très rare) |
| 3–7/min            | 0.65  | Deepfake probable (anormal) |
| 7–30/min           | 0.10  | Authentique (plage normale) |
| > 30/min           | 0.55  | Suspicious (trop fréquent) |

---

## 3. Données d'entraînement

**Aucun apprentissage.** Algorithme entièrement déterministe, pas de paramètres appris.

Références :
- Soukupová & Čech, « Real-Time Eye Blink Detection using Facial Landmarks », *CVWW 2016*
- Pan et al., « Exposing DeepFake Videos By Detecting Face Warping Artifacts », *CVPR Workshop 2019*
- Li et al., « Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics », *CVPR 2020*

---

## 4. Évaluation (Règle 10, condition 1)

Fichier de métriques : `evaluation/metrics/biometrics.json`  
Script : `training/eval_biometrics_engine.py`  
Date : 2026-10-06

### Jeu de test synthétique (N=100)

| Classe | Modèle | N |
|--------|--------|---|
| Authentique | EAR séquences avec 15-20 blinks/min (4-5 blinks en 15 s) | 50 |
| Deepfake type A | 0 blink total, EAR plat ≈ 0.30 | 30 |
| Deepfake type B | Clignements rapides ~300/min (cycle 6 frames) | 20 |

### Résultats (seuil 0.5)

| Métrique | Valeur |
|----------|--------|
| FAR (faux positifs / authentiques) | 0.00 (0/50) |
| FRR (faux négatifs / deepfakes) | 0.00 (0/50) |
| AUC | 1.00 |
| EER | 0.00 |
| Précision | 1.00 |
| Rappel | 1.00 |

---

## 5. Limites connues

1. **Deepfakes récents non détectés** : les modèles entraînés post-2021 (NeRF, Stable Diffusion, méthodes one-shot) ont appris à générer un clignement réaliste. Ce moteur n'est efficace que sur les deepfakes des premières générations (GAN early). Évaluation sur Celeb-DF v2 et FaceForensics++ recommandée.

2. **Faux positifs visuels** : certaines personnes clignent naturellement moins de 7 fois par minute (e.g., concentration, certaines conditions médicales) — le moteur les classerait comme deepfakes.

3. **Occultation partielle** : si les yeux sont partiellement couverts (lunettes épaisses, angle de vue > 45°), MediaPipe produit des EAR incorrects. Le moteur retourne 0.0 sans landmarks.

4. **Basse résolution** : EAR non fiable si les yeux occupent moins de 20×10 pixels dans le frame.

5. **Angle de vue** : EAR calculé sur projection 2D des landmarks 3D — dégradation pour profils > 30°.

6. **Fréquence d'images** : vidéos < 20 fps → détection des blinks dégradée (un blink dure ~100-200 ms soit 2-4 frames à 25 fps — peut être raté à faible fps).

7. **Aucun fine-tuning** : le moteur est basé sur des seuils biologiques, pas sur apprentissage supervisé. Aucune métrique sur benchmark public disponible.

---

## 6. Utilisation dans la fusion

Poids dans l'ensemble : **10 %** (`WEIGHTS["biometrics"] = 0.10` dans `engines/fusion.py`).

Ce poids faible reflète les limites du moteur face aux deepfakes récents.

---

## 7. Procédure de validation (Règle 10)

Pour passer de `experimental` à `validated` :

1. [x] Produire `evaluation/metrics/biometrics.json` avec `engine_name: "biometrics"` ✅ 2026-10-06
2. [x] Ce fichier de model card est présent ✅
3. [ ] Mettre `status: "completed"` dans `evaluation/protocols/baseline_protocol.yaml` après application du protocole sur données réelles
4. [ ] Un admin appelle `POST /models/engines/biometrics/approve` avec notes de revue
5. [ ] Un admin appelle `POST /models/engines/biometrics/promote`

---

## 8. Responsabilité

Ce moteur assiste l'expert humain. Il ne produit pas de conclusion. Un taux de clignement anormal n'est pas une preuve de manipulation — une vidéo d'une personne fatiguée, concentrée, ou présentant certaines conditions médicales peut aussi produire un score élevé.

> *Conçu pour soutenir l'évaluation de l'intégrité, de la provenance et de l'authenticité technique de contenus numériques, sans constituer à lui seul une conclusion d'expert ou une preuve définitive de manipulation.*
