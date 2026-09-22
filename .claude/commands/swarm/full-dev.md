---
name: full-dev
description: Essaim complet pour développement majeur DeepfakeDetector — 5 agents en pipeline
---

# Full Dev Swarm — DeepfakeDetector

Lance un essaim de 5 agents spécialisés en pipeline pour un développement majeur.

## Usage

```
/full-dev [objectif de développement]
```

Exemples :
- `/full-dev Implémenter le moteur de détection vidéo complet avec ONNX`
- `/full-dev Refactoriser la chain of custody avec support multi-hash`
- `/full-dev Ajouter l'API de gestion des cas (CRUD complet)`

## Pipeline d'agents

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐
│  researcher  │ → │    planner   │ → │   coder    │
│ (contexte)   │    │ (plan 5 étapes)│    │ (implémente)│
└─────────────┘    └──────────────┘    └────────────┘
                                              ↓
                   ┌──────────────┐    ┌────────────┐
                   │   reviewer   │ ← │   tester   │
                   │ (code review)│    │  (tests)   │
                   └──────────────┘    └────────────┘
```

## Règles du swarm

1. **researcher** lit TOUS les fichiers concernés avant de passer la main
2. **planner** décompose en tâches atomiques de <50 lignes de code chacune
3. **coder** implémente une tâche à la fois, commit à chaque étape
4. **tester** crée les tests avant que reviewer intervienne
5. **reviewer** bloque si checklist forensique non-respectée

## Agents disponibles

Pour ce projet, les agents suivants sont configurés :
- `core/coder` — FastAPI Python
- `core/tester` — pytest
- `core/reviewer` — code review forensique
- `security/security-auditor` — OWASP
- `project/ml-engineer` — PyTorch/ONNX
- `project/forensic-analyst` — chain of custody
- `goal/goal-planner` — planification GOAP
