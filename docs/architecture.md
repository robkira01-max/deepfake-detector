# Architecture — DeepfakeDetector Canada v3.1.0

---

## Services (Docker Compose)

```
Client HTTP / HTTPS
        │
        ▼
   Nginx (port 80 / 443)
        │
        ▼
   FastAPI API (port 8001 en dev)
        │
   ┌────┴──────────────────────┐
   │                           │
   ▼                           ▼
PostgreSQL                  Redis
(port 5432, localhost)      (port 6379, localhost)
SQLAlchemy 2.x              Token blocklist
Alembic migrations          Celery broker
        │                           │
        └──────────┬────────────────┘
                   ▼
            Celery Worker
            (analyses ML async)
                   │
                   ▼
              MinIO
        (port 9000/9001, localhost)
        Médias uploadés + rapports PDF

Flower (monitoring Celery) : port 5555, localhost
```

---

## Endpoints API

Lu depuis `backend/main.py` :

| Préfixe | Module | Rôle |
|---------|--------|------|
| `/auth` | `routers/auth.py` | Authentification JWT RS256 + MFA TOTP |
| `/cases` | `routers/cases.py` | Dossiers forensiques (CRUD, RBAC) |
| `/analyze` | `routers/analyze.py` | Upload fichiers + lancement analyse Celery |
| `/reports` | `routers/reports.py` | Génération et téléchargement PDF/A-3 |
| `/kyc` | `routers/kyc.py` | Vérification d'identité biométrique |
| `/export` | `routers/export.py` | Export CSV / batch |
| `/webhooks` | `routers/webhooks.py` | Notifications sortantes |
| `/analytics` | `routers/analytics.py` | Tableau de bord statistiques |
| `/documents` | `routers/document.py` | Analyse forensique de documents |
| `/templates` | `routers/templates.py` | Modèles de rapports |
| `/models` | `routers/models.py` | Registre des modèles ML |
| `/feedback` | `routers/feedback.py` | Feedback utilisateur |
| `/dashboard` | `routers/dashboard.py` | Dashboard analyst |

---

## Pipeline d'analyse

```
1. Upload fichier
        │
        ▼
2. compute_hashes()           SHA-256 (probatoire) + Blake3 (dédup)
        │
        ▼
3. stamp_file()               TSA RFC 3161 → token horodatage certifié
        │
        ▼
4. AuditLog FILE_INGESTED     entry_hash + previous_entry_hash (chaîné SHA-256)
        │
        ▼
5. Celery task → moteurs en parallèle
   ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
   │ texture  │ temporal │  rppg    │biometrics│  phase   │ metadata │  audio   │
   │EfficNet-B│ResNet-50 │  CHROM   │ EAR + AU │  STFT    │heurist.  │Wav2Vec2  │
   │[experim.]│[experim.]│[experim.]│[experim.]│[experim.]│[experim.]│[disabled]│
   └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
        │
        ▼
6. fusion.py                  Score pondéré (abstention si aucun validated)
        │
        ▼
7. Génération PDF              WeasyPrint → rapport PDF/A-3
        │
        ▼
8. Signature RSA-4096          + SHA-256 du PDF + TSA RFC 3161
        │
        ▼
9. AuditLog REPORT_GENERATED  chaîné SHA-256
```

---

## Statut des moteurs ML

Lu depuis `backend/engines/fusion.py` — `ENGINE_DEFAULT_STATUS` :

| Moteur | Algorithme | Statut | Poids |
|--------|-----------|--------|-------|
| `texture` | EfficientNet-B4 | `experimental` | ImageNet (non fine-tuné deepfake) |
| `temporal` | ResNet-50 + LSTM | `experimental` | ImageNet |
| `rppg` | CHROM | `experimental` | — algorithme déterministe |
| `biometrics` | EAR + Action Units | `experimental` | MediaPipe |
| `phase` | STFT phase | `experimental` | — algorithme déterministe |
| `metadata` | Heuristiques | `experimental` | — règles statiques |
| `audio` | Wav2Vec2 | **`disabled`** | Tête aléatoire — non entraîné |

**Aucun moteur `validated` à ce jour.**
Avec `ALLOW_EXPERIMENTAL_ENGINES=false` (défaut prod recommandé) → score = 0.0.

Conditions pour passer à `validated` (CLAUDE.md Règle 10) :
1. `evaluation/metrics/metrics.json` produit sur jeu de test indépendant
2. Model card dans `evaluation/model_cards/`
3. `evaluation/protocols/baseline_protocol.yaml` complété
4. Approbation humaine journalisée dans `AuditLog` (`ENGINE_STATUS_CHANGED`)

---

## Couche sécurité

```
Auth
  └── JWT RS256 (asymétrique) + MFA TOTP obligatoire en prod
  └── RBAC : admin / analyst / readonly
  └── Token blocklist Redis (révocation immédiate)

Chiffrement at-rest
  └── AES-256-GCM — nonce 96 bits aléatoire par opération
  └── Clé : ENCRYPTION_KEY (base64url 32 octets)

Chain of custody
  └── SHA-256 (probatoire → TSA)
  └── Blake3 (performance / déduplication)
  └── TSA RFC 3161 : dev=FreeTSA.org / prod=TSA accréditée (à configurer)
  └── RSA-4096 PSS : signature de chaque entrée d'audit + chaque rapport PDF
  └── AuditLog chaîné SHA-256 (previous_entry_hash)

Headers HTTP
  └── HSTS, CSP, X-Frame-Options, X-Content-Type-Options (ADR-005)
```

---

## Stack technique

Lu depuis `backend/config.py` et `requirements.txt` :

| Composant | Technologie |
|-----------|-------------|
| API | FastAPI 0.115 + Pydantic v2 |
| Auth | JWT RS256 + MFA TOTP + RBAC |
| Base de données | PostgreSQL (prod) / SQLite (dev uniquement) |
| ORM | SQLAlchemy 2.x + Alembic |
| Queue | Celery + Redis |
| Stockage fichiers | MinIO |
| ML inférence | ONNX Runtime + PyTorch CPU |
| PDF | WeasyPrint |
| Chiffrement | AES-256-GCM (at-rest) · RSA-4096 PSS (signatures) |
| Hachage | SHA-256 (probatoire/TSA) + Blake3 (perf) |
| Horodatage | RFC 3161 TSA |

> **Divergence dev/prod** : SQLite en dev, PostgreSQL en prod.
> La CI doit tourner sur PostgreSQL pour détecter les divergences de comportement.

---

## Décisions d'architecture (ADRs)

Lu depuis `docs/adr/` :

| ADR | Décision |
|-----|----------|
| [ADR-001](adr/ADR-001-auth-jwt-rs256.md) | JWT RS256 (asymétrique) + MFA TOTP obligatoire — pas HS256 |
| [ADR-002](adr/ADR-002-chain-of-custody.md) | Blake3 + SHA-256 + TSA RFC 3161 pour chaîne de possession |
| [ADR-003](adr/ADR-003-ml-engines.md) | ONNX Runtime pour l'inférence CPU + PyTorch pour fine-tuning |
| [ADR-004](adr/ADR-004-celery-vs-background-tasks.md) | Celery + Redis pour la queue async (pas FastAPI BackgroundTasks) |
| [ADR-005](adr/ADR-005-security-headers.md) | SecurityHeadersMiddleware sur toutes les réponses HTTP |
| [ADR-006](adr/ADR-006-alembic-migrations.md) | Alembic pour les migrations de schéma PostgreSQL |

---

## Structure des répertoires

```
deepfake_detector/
├── backend/
│   ├── main.py                  # FastAPI app + _validate_production_config()
│   ├── config.py                # Settings (pydantic-settings, depuis .env)
│   ├── core/
│   │   ├── chain_of_custody.py  # SHA-256 + Blake3 + TSA + signature RSA
│   │   ├── encryption.py        # AES-256-GCM at-rest
│   │   ├── ingestion.py         # Upload + validation + chain of custody
│   │   └── security.py          # JWT RS256 + MFA + RBAC
│   ├── engines/
│   │   ├── fusion.py            # Fusion pondérée + _require_metrics()
│   │   ├── video_engine.py      # EfficientNet-B4 + ResNet-50+LSTM
│   │   ├── audio_engine.py      # Wav2Vec2 [disabled]
│   │   ├── biometric_engine.py  # EAR + Action Units
│   │   ├── metadata_engine.py   # Heuristiques métadonnées
│   │   └── text_engine.py       # Analyse texte
│   ├── models/                  # SQLAlchemy ORM
│   ├── routers/                 # FastAPI routes (15 modules)
│   ├── tasks/                   # Celery tasks (analysis + retrain)
│   ├── reporting/               # PDF WeasyPrint + templates Jinja2
│   └── migrations/              # Alembic (0001 → 0004)
├── evaluation/
│   ├── metrics/                 # metrics.json requis pour moteur validated
│   ├── model_cards/
│   ├── protocols/baseline_protocol.yaml
│   └── robustness/ · drift/ · validation_reports/
├── training/
│   ├── dataset_prep.py          # Split strict par identité
│   ├── eval_runner.py           # Cross-dataset evaluation
│   └── metrics_schema.py        # Schéma Pydantic pour metrics.json
├── legal_references.yaml        # Source unique des références légales
├── datasets/registry.yaml       # Inventaire + licences datasets
└── docs/adr/                    # ADR-001 à ADR-006
```
