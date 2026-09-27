# DeepfakeDetector — Guide de développement Claude Code + Ruflo

## Description du projet

API forensique de détection de deepfake (vidéo + audio) pour le marché canadien.
Contraintes légales : LPRPDE, preuve numérique, chain of custody RFC3161.

**Statut** : v3.1.0 — Démo active sur http://127.0.0.1:8082

## Stack technique

| Composant | Technologie |
|-----------|------------|
| API | FastAPI 0.115 + Pydantic v2 |
| Auth | JWT RS256 + MFA TOTP + RBAC |
| Base de données | PostgreSQL + SQLAlchemy 2.0 |
| Queue | Celery + Redis |
| Stockage fichiers | MinIO |
| ML | PyTorch CPU + ONNX Runtime + MediaPipe |
| PDF | WeasyPrint + Jinja2 |
| Sécurité | bcrypt/12 + Fernet + RSA-4096 |

## Structure du projet

```
deepfake_detector/
├── backend/
│   ├── main.py              # Point d'entrée FastAPI
│   ├── config.py            # Settings depuis .env
│   ├── database.py          # SQLAlchemy init
│   ├── core/
│   │   ├── security.py      # JWT, MFA, RBAC
│   │   ├── chain_of_custody.py  # Blake3 + TSA
│   │   ├── ingestion.py     # Upload + validation
│   │   ├── encryption.py    # Fernet at-rest
│   │   └── token_blocklist.py   # Redis blocklist
│   ├── engines/
│   │   ├── video_engine.py  # Détection deepfake vidéo
│   │   ├── audio_engine.py  # Détection deepfake audio
│   │   └── fusion.py        # Fusion des scores
│   ├── routers/             # Routes API (auth, analyze, cases, reports)
│   ├── models/              # SQLAlchemy models
│   ├── tasks/               # Celery async tasks
│   └── reporting/           # Génération PDF
├── .claude/                 # Configuration Ruflo
│   ├── agents/              # Agents spécialisés
│   └── commands/            # Commandes slash
└── docs/
    └── adr/                 # Architecture Decision Records
```

## Agents Ruflo disponibles

| Agent | Rôle | Invoquer quand |
|-------|------|----------------|
| `core/coder` | FastAPI Python | Nouvelle route, service, modèle |
| `core/tester` | Tests pytest | Génération tests, couverture |
| `core/reviewer` | Code review | Avant tout merge |
| `security/security-auditor` | Audit OWASP/LPRPDE | Audit sécurité |
| `project/ml-engineer` | PyTorch/ONNX | Moteurs deepfake |
| `project/forensic-analyst` | Chain of custody | Intégrité légale |
| `goal/goal-planner` | Planification GOAP | Fonctionnalités complexes |

## Commandes slash disponibles

| Commande | Description |
|----------|-------------|
| `/security-audit` | Audit OWASP complet |
| `/new-feature [desc]` | SPARC nouvelle fonctionnalité |
| `/full-dev [objectif]` | Essaim 5 agents pipeline complet |

## Lancement du projet

```bash
# Venv
source /home/kali/deepfake_detector/.venv/bin/activate
cd /home/kali/deepfake_detector/backend

# API seule (dev)
uvicorn main:app --reload --port 8080

# Stack complète (Docker)
cd /home/kali/deepfake_detector
docker-compose up
```

## Règles de développement (NON-NÉGOCIABLES)

1. **Jamais de secret hardcodé** — utiliser `settings.*` depuis `config.py`
2. **Chain of custody obligatoire** sur tout fichier ingéré (Blake3 + TSA)
3. **AuditLog** pour toute action sur Case/Analysis/Report
4. **Type hints complets** — mypy doit passer
5. **Tests avant merge** — couverture >80% sur les nouveaux modules
6. **Pas de .env commité** — vérifié par .gitignore

## ADRs (Architecture Decision Records)

Voir `docs/adr/` pour les décisions architecturales bindantes.

## Variables d'environnement

Toutes dans `backend/.env` — ne jamais afficher en clair.
Voir `backend/config.py` pour la liste complète des variables.
