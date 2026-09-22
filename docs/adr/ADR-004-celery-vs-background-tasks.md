# ADR-004 — Architecture de queue : Celery + Redis vs FastAPI BackgroundTasks

**Date** : 2026-09-22
**Statut** : Accepté
**Décideurs** : Équipe DeepfakeDetector

---

## Contexte

L'analyse deepfake d'un fichier vidéo ou audio peut prendre de 30 secondes à plusieurs
minutes selon la taille du fichier et les moteurs activés. La réponse HTTP doit être
immédiate (< 2s). Deux approches sont disponibles dans l'écosystème FastAPI :

1. **FastAPI `BackgroundTasks`** — tâche lancée après la réponse HTTP, dans le même
   processus worker uvicorn
2. **Celery + Redis** — tâche déléguée à un worker externe, communique via broker Redis

---

## Décision

**Celery + Redis** est retenu comme architecture de queue.

---

## Raisons

### Pourquoi Celery plutôt que BackgroundTasks

| Critère | BackgroundTasks | Celery + Redis |
|---------|-----------------|----------------|
| Isolation processus | Non (même worker) | Oui (worker dédié) |
| Survie au crash | Non (perte si worker redémarre) | Oui (tâche persistée dans Redis) |
| Scalabilité horizontale | Non | Oui (N workers) |
| Monitoring | Aucun natif | Flower UI + métriques |
| Retry automatique | Manuel | Natif (max_retries, countdown) |
| Progression temps réel | Non | Oui (task.update_state) |
| Annulation tâche | Non | Oui (task.revoke()) |
| Analyses parallèles | Bloquant (GIL) | Oui (workers séparés) |

### Contraintes légales (Canada — LPC art. 31.1-31.6)

- La chaîne de possession exige que chaque étape soit **traçable et non répudiable**
- Celery permet de lier `analysis.celery_task_id` à chaque enregistrement `Analysis`
- En cas de contestation judiciaire, l'état de la tâche (pending/running/completed/failed)
  est consultable dans les logs Redis avec timestamp

### Considérations opérationnelles

- **Fail-open vs fail-secure** : si Redis est indisponible, l'analyse est refusée
  (fail-secure) plutôt que lancée dans le processus web (risque OOM)
- **Timeout** : les analyses ont un timeout de 10 min (600s) configuré dans Celery
  pour éviter les workers bloqués
- **Priorités** : Celery permet de prioriser les analyses urgentes (cas judiciaires actifs)

---

## Alternatives considérées

### FastAPI BackgroundTasks
- Rejeté : pas de persistance entre redémarrages, pas de monitoring, impossible
  de scaler indépendamment les workers ML (GPU/CPU intensifs)

### RQ (Redis Queue)
- Envisagé : plus simple que Celery, mais moins de fonctionnalités (pas de Canvas,
  pas de rate limiting natif, moins bon support Python 3.14)

### ARQ (Async RQ)
- Envisagé : async natif, mais intégration SQLAlchemy sync plus complexe, écosystème
  plus petit

### Dramatiq
- Rejeté : middleware supplémentaire, moins répandu, moins de ressources communautaires

---

## Conséquences

### Positives
- Workers ML peuvent être déployés sur des machines séparées (CPU/GPU)
- Retry automatique sur erreur transiente (ex: modèle ML pas encore chargé)
- Flower UI disponible pour monitoring des queues (`localhost:5555`)
- `analysis.celery_task_id` = preuve d'exécution dans la chaîne de custody

### Négatives
- Dépendance Redis (point unique de défaillance si non clustérisé)
- Complexité opérationnelle accrue (2 services supplémentaires : Redis + Celery worker)
- Tests plus complexes (mock `run_deepfake_analysis.delay`)

### Atténuations
- Redis Sentinel ou Redis Cluster en production pour HA
- `demo.py` fourni pour analyses locales sans Celery (usage développement)
- Tests mockent `run_deepfake_analysis.delay` — aucune dépendance réelle à Celery en CI

---

## Impact sur le code

- `backend/tasks/analysis_tasks.py` — tâche Celery principale
- `backend/routers/analyze.py` — `POST /analyze/start/{media_file_id}` → `.delay()`
- `backend/tasks/celery_app.py` — configuration du broker
- `demo.py` — contourne Celery pour usage local/démo

---

## Références

- [Celery docs — Canvas](https://docs.celeryq.dev/en/stable/userguide/canvas.html)
- [FastAPI BackgroundTasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- R. c. Mohan [1994] 2 RCS 9 — critères d'admissibilité preuve scientifique
- LPC art. 31.1-31.6 — intégrité des documents technologiques
