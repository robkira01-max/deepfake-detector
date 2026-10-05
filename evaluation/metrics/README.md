# evaluation/metrics/

Ce répertoire doit contenir `metrics.json` avant toute production de chiffres de fiabilité.

## Requis pour déverrouiller les métriques (CLAUDE.md Règle 7)

- `metrics.json` — résultats sur jeu de test indépendant
- `confidence_intervals.json` — intervalles de confiance à 95%
- `calibration.json` — courbe de calibration des probabilités
- `subgroup_metrics.json` — performance par sous-groupe et condition technique

## Format minimal de metrics.json

```json
{
  "schema_version": "1.0",
  "model_id": "video-detector-x.y.z",
  "model_hash": "sha256:...",
  "evaluation_date": "YYYY-MM-DD",
  "protocol": "cross_generator_protocol:v1",
  "dataset": "external_test_manifest.json",
  "n_samples": 0,
  "auc": null,
  "eer": null,
  "far_at_threshold": {},
  "frr_at_threshold": {},
  "precision": null,
  "recall": null,
  "f1": null,
  "mcc": null,
  "notes": "PLACEHOLDER — aucune valeur réelle. Fichier créé pour documenter la structure requise."
}
```

## Procédure de validation

Voir `evaluation/protocols/baseline_protocol.yaml`.
Aucun engine ne peut passer de `experimental` à `validated` sans :
1. Ce fichier complété avec des valeurs réelles
2. Une model card dans `evaluation/model_cards/`
3. Une approbation humaine journalisée dans l'AuditLog
