---
name: new-feature
description: Développer une nouvelle fonctionnalité DeepfakeDetector avec SPARC — de la spec au test
---

# SPARC New Feature — DeepfakeDetector

Utilise la méthodologie SPARC pour développer une nouvelle fonctionnalité de manière structurée et sécurisée.

## Usage

```
/new-feature [description de la fonctionnalité]
```

Exemples :
- `/new-feature Ajouter la détection deepfake par image fixe (JPEG/PNG)`
- `/new-feature Implémenter l'export de cas en format ZIP avec chain of custody`
- `/new-feature Ajouter un dashboard d'administration pour gérer les utilisateurs`

## Workflow SPARC appliqué

### S — Spécification
- Quel endpoint API ? (méthode, path, auth requise)
- Quelles données entrent et sortent ? (schémas Pydantic)
- Quelles contraintes de sécurité ? (RBAC, rate limit)
- Faut-il une tâche Celery async ou sync ?
- Impact sur la chain of custody ?

### P — Pseudocode
```python
# Exemple structure
@router.post("/nouveau/endpoint")
async def nouveau_endpoint(payload, db, user):
    # 1. Valider autorisation RBAC
    # 2. Valider payload Pydantic
    # 3. Exécuter logique métier
    # 4. Créer AuditLog
    # 5. Retourner réponse typée
```

### A — Architecture
- Nouveau fichier dans `routers/` ou extension existante ?
- Nouveau modèle SQLAlchemy dans `models/` ?
- Nouvelle tâche Celery dans `tasks/` ?
- Nouveau service dans `core/` ?

### R — Raffinement (TDD + sécurité)
- Écrire les tests AVANT l'implémentation
- Audit sécurité de la route
- Vérifier integration chain of custody

### C — Complétion
- Intégrer dans `main.py` (include_router)
- Documentation endpoint (docstring FastAPI → Swagger auto)
- Tests de régression sur les routes existantes
