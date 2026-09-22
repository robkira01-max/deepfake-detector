# ADR-006 — Alembic pour les migrations de schéma PostgreSQL

**Date** : 2026-09-22  
**Statut** : Accepté  
**Décideurs** : Équipe DeepfakeDetector

---

## Contexte

Le projet utilise SQLAlchemy 2.0 avec PostgreSQL. Le schéma doit évoluer dans le temps
(nouveaux champs, index, tables) sans perdre les données de production. Une stratégie
de migration versionnée et réversible est nécessaire.

## Décision

Alembic est utilisé comme outil de migration. Configuration :

1. **`env.py` connecté aux models** : importe `Base` depuis `database.py` et tous les
   models pour que `--autogenerate` détecte les changements.

2. **`DATABASE_URL` depuis l'environnement** : `env.py` lit `os.environ["DATABASE_URL"]`
   en priorité — pas d'URL hardcodée dans `alembic.ini`.

3. **Migration initiale manuelle** : `0001_initial_schema.py` créée manuellement avec
   toutes les tables, FK, et index — parce que PostgreSQL n'est pas disponible en
   environnement de dev local (Kali VM sans Docker actif).

4. **Commandes Makefile** :
   - `make migrate` → `alembic upgrade head`
   - `make migration MSG="description"` → `alembic revision --autogenerate`

5. **Options `compare_type=True` et `compare_server_default=True`** activées pour
   détecter les changements de types et de valeurs par défaut.

## Alternatives rejetées

- **Pas de migrations / init_db direct** : `Base.metadata.create_all()` ne gère pas
  les schémas existants — dangereux en production.
- **Django Migrations** : non compatible avec FastAPI/SQLAlchemy.
- **Flyway / Liquibase** : outils Java, overhead pour un projet Python.

## Conséquences

- Chaque modification de model doit être accompagnée d'une migration Alembic.
- Les migrations doivent être testées en staging avant production.
- `downgrade()` doit être implémenté pour chaque migration (rollback possible).
- En CI/CD : `alembic upgrade head` doit passer avant le démarrage de l'API.

## Schéma courant (révision 0001)

6 tables : `users`, `cases`, `media_files`, `audit_logs`, `analyses`, `reports`

Toutes les FK utilisent `ON DELETE RESTRICT` sauf `audit_logs.user_id` (`SET NULL`)
pour préserver l'historique d'audit même si un utilisateur est supprimé.
