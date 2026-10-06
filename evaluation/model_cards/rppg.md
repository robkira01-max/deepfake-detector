# Model Card — Moteur rPPG (rppg)

**Version :** 1.0.0  
**Type :** Algorithmique — CHROM (de Haan & Jeanne, 2013) + analyse FFT  
**Statut :** experimental (pas encore validated — voir procédure ci-dessous)  
**Dernière mise à jour :** 2026-10-06  

---

## 1. Description

Le moteur `rppg` (remote Photoplethysmography) détecte la présence ou l'absence d'un signal cardiaque physiologique dans les variations de couleur de la peau visible sur une vidéo.

**Principe** : un visage réel présente une modulation périodique de la couleur de la peau synchronisée avec le pouls cardiaque (60-100 BPM) via l'absorption différentielle de la lumière par l'oxyhémoglobine — principalement dans le canal rouge (R). Un visage synthétique généré par GAN ou diffusion n'a pas ce signal différentiel : ses canaux R, G, B varient de façon uniforme (même facteur de modulation pour tous), ce que l'algorithme CHROM est précisément conçu à annuler.

---

## 2. Architecture

### Algorithme CHROM (de Haan & Jeanne, IEEE TBME 2013)

Implémenté dans `backend/engines/video_engine.py` — méthode `_compute_rppg()`.

**Pipeline :**

1. **Extraction RGB** : pour chaque frame, calculer les moyennes R, G, B dans la ROI du visage
2. **Normalisation** : diviser par la moyenne temporelle de chaque canal (`sig_norm = sig / mean(sig, axis=0)`)
3. **Combinaison CHROM** :
   - `xs = 3·R_norm − 2·G_norm`
   - `ys = 1.5·R_norm + G_norm − 1.5·B_norm`
   - `alpha = std(xs) / std(ys)`
   - `rppg_signal = xs − alpha·ys`
4. **Filtrage passe-bande** : Butterworth ordre 3, [0.67 Hz, 3.0 Hz] (bande cardiaque 40-180 BPM)
5. **Analyse spectrale** : FFT du signal filtré → SNR = puissance cardiaque / puissance hors-bande
6. **Score** : `score = 1 − tanh(SNR / 5)` — 0 = signal cardiaque fort (authentique), 1 = absent (deepfake)

**Propriété clé de CHROM** : si R_norm = G_norm = B_norm (variation uniforme de tous les canaux, i.e. variation d'illumination pure), alors `rppg_signal = 0`. Cela annule les artéfacts d'illumination et de mouvement communs aux trois canaux. Les deepfakes GAN/diffusion tendent à produire exactement ce type de variation uniforme.

---

## 3. Données d'entraînement

**Aucun apprentissage.** Algorithme entièrement déterministe, pas de paramètres appris.

Références algorithmiques :
- De Haan & Jeanne, « Robust pulse rate from chrominance-based rPPG », *IEEE TBME*, 2013, DOI: 10.1109/TBME.2013.2266196
- Verkruysse et al., « Remote plethysmographic imaging using ambient light », *Opt. Exp.*, 2008

---

## 4. Évaluation (Règle 10, condition 1)

Fichier de métriques : `evaluation/metrics/rppg.json`  
Script : `training/eval_rppg_engine.py`  
Date : 2026-10-06

### Jeu de test synthétique (N=100)

| Classe | Modèle | N |
|--------|--------|---|
| Authentique | Signal différentiel R/G à 60-100 BPM + bruit gaussien σ=1 | 50 |
| Deepfake | Illumination uniforme (R/G/B proportionnels), fréquences hors bande cardiaque | 50 |

### Résultats (seuil 0.5)

| Métrique | Valeur |
|----------|--------|
| FAR (faux positifs / authentiques) | 0.02 (1/50) |
| FRR (faux négatifs / deepfakes) | 0.00 (0/50) |
| AUC | 1.00 |
| EER | 0.00 |
| Précision | 0.98 |
| Rappel | 1.00 |

---

## 5. Limites connues

1. **Jeu synthétique non représentatif** : les deepfakes réels ne sont pas tous de type « illumination uniforme ». Des générateurs produisant un différentiel R/G accidentel dans la bande cardiaque ne seraient pas détectés. Évaluation sur FaceForensics++/DFDC requise (licences non-commerciales — décision propriétaire nécessaire).

2. **Compression vidéo** : la compression H.264/H.265 introduit des artefacts de blocs qui perturbent les signaux RGB. Robustesse non évaluée.

3. **Basse résolution** : ROI du visage < 50×50 pixels : signal trop bruité. Limite documentée dans le code (min `len(rgb_signals) < 30`).

4. **Occultation partielle** : le visage doit être visible en continu sur la fenêtre rPPG (10 s par défaut). Toute occultation retourne 0.3 (score neutre).

5. **Éclairage non uniforme / stroboscope** : la normalisation CHROM suppose un éclairage ambiant relativement stable. Un éclairage stroboscopique peut produire de faux positifs.

6. **Aucun fine-tuning** : le modèle est basé sur des heuristiques bio-optiques, pas sur un apprentissage supervisé sur données deepfake. Précision brute sur benchmark public inconnue.

---

## 6. Utilisation dans la fusion

Poids dans l'ensemble : **20%** (`WEIGHTS["rppg"] = 0.20` dans `engines/fusion.py`).

Ce poids reflète la valeur du signal cardiaque comme indicateur fort d'authenticité physiologique.

---

## 7. Procédure de validation (Règle 10)

Pour passer de `experimental` à `validated` :

1. [x] Produire `evaluation/metrics/rppg.json` avec `engine_name: "rppg"` ✅ 2026-10-06
2. [x] Ce fichier de model card est présent ✅
3. [ ] Mettre `status: "completed"` dans `evaluation/protocols/baseline_protocol.yaml` après application du protocole sur données réelles
4. [ ] Un admin appelle `POST /models/engines/rppg/approve` avec notes de revue
5. [ ] Un admin appelle `POST /models/engines/rppg/promote`

---

## 8. Responsabilité

Ce moteur assiste l'expert humain. Il ne produit pas de conclusion. L'absence de signal rPPG n'est pas une preuve de manipulation — une vidéo basse résolution, compressée ou avec visage partiellement visible peut aussi produire un score élevé.

> *Conçu pour soutenir l'évaluation de l'intégrité, de la provenance et de l'authenticité technique de contenus numériques, sans constituer à lui seul une conclusion d'expert ou une preuve définitive de manipulation.*
