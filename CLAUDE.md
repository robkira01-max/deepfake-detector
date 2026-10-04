# DeepfakeDetector Canada — Guide de développement Claude Code + Ruflo

## Description du projet

Prototype de plateforme d'analyse forensique de médias synthétiques, conçu pour soutenir
l'évaluation de l'intégrité, de la provenance et de l'authenticité technique de contenus
numériques, sans constituer à lui seul une conclusion d'expert ou une preuve définitive
de manipulation.

**Marchés cibles** (par ordre de réalisme) :
1. **Litige / dossier de preuve numérique** : ingestion hachée + horodatage + revue humaine
2. **Assurance / KYC** : chaîne de possession + hébergement canadien + rapport bilingue
3. **Élections (C-25)** : la loi crée une infraction, pas un marché de détection — faible priorité commerciale

**Statut** : v3.2.0 — 693 tests, 0 régression · ⏸ EN PAUSE — reprise 2026-10-04

---

## ⚠️ Risque central : aucun moteur validé

Tous les moteurs sont `experimental`. L'audio est `disabled`.
Avec `allow_experimental_engines=False`, le score est à 0,0.

**La plateforme est une excellente chaîne de possession sans capacité de détection commercialisable.**
Ce n'est pas un défaut de conception — mais il faut le dire clairement à tout évaluateur.

**Positionnement recommandé** : « dossier de preuve numérique avec expert dans la boucle »,
pas « détecteur de deepfakes ». La concurrence (Reality Defender, Hive, Pindrop) est trop
avancée sur la précision brute.

---

## Règles NON-NÉGOCIABLES (10 règles — refus automatique si violation)

### Règle 1 — Jamais de secret hardcodé
Utiliser `settings.*` depuis `config.py`. Aucune clé, mot de passe, token en dur dans le code.

### Règle 2 — MD5 INTERDIT
MD5 n'a aucune valeur probatoire. **Retiré partout.**
- SHA-256 = empreinte probatoire → envoyée au TSA RFC 3161
- Blake3 = performance et déduplication interne
- `hash_md5` dans le modèle `MediaFile` est nullable/déprécié — ne pas alimenter

### Règle 3 — Chain of custody obligatoire
Sur tout fichier ingéré : SHA-256 + Blake3 + horodatage TSA RFC 3161.
`AuditLog` obligatoire pour toute action sur `Case`, `Analysis`, `Report`.

### Règle 4 — Config de production sécurisée
`_validate_production_config()` dans `main.py` refuse le démarrage si :
- `app_env=production` + `debug=True`
- `app_env=production` + `mfa_required=False`
- `app_env=production` + `encryption_key` absent

### Règle 5 — Chiffrement AES-256-GCM uniquement
`core/encryption.py` utilise AES-256-GCM (nonce 96 bits aléatoire par opération).
**Fernet est déprécié** — conservé uniquement pour migration de données existantes.
Générer une clé : `python3 -c "import os,base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"`

### Règle 6 — Statuts des moteurs ML
Trois états dans `engines/fusion.py` :
- `validated` — inclus toujours dans le score
- `experimental` — inclus si `settings.allow_experimental_engines=True`
- `disabled` — **toujours exclu** (ex : Wav2Vec2 tête aléatoire)

**Actuellement** : audio=`disabled`, tous les autres=`experimental`. **Aucun moteur `validated`.**

### Règle 7 — Aucun chiffre de fiabilité sans metrics.json
`_require_metrics()` dans `fusion.py` lève `ValueError` (pas `None`, pas 0.0) si
`evaluation/metrics/metrics.json` est absent.
**Interdits sans metrics.json** : FAR, FRR, AUC, EER, tout pourcentage de fiabilité du modèle.

### Règle 8 — Jamais de formulation définitive
Toujours utiliser :
> *"conçu pour soutenir l'évaluation de l'intégrité, de la provenance et de l'authenticité
> technique de contenus numériques, sans constituer à lui seul une conclusion d'expert ou
> une preuve définitive de manipulation"*

**Interdit** : "conforme", "recevable en cour", "validé par un expert", "preuve de manipulation".

### Règle 9 — Références légales via legal_references.yaml uniquement
Source unique de vérité : `legal_references.yaml` à la racine.
Un élément marqué `status: a_valider` **ne doit jamais apparaître dans un rapport émis**.
Ne jamais écrire une loi ou norme en dur dans le code ou les templates.

### Règle 10 — Validation d'un moteur = 4 conditions
Un engine ne passe à `validated` que si **toutes** les conditions sont réunies :
1. `evaluation/metrics/metrics.json` produit sur jeu de test indépendant
2. Model card dans `evaluation/model_cards/`
3. Protocole `evaluation/protocols/baseline_protocol.yaml` complété
4. Approbation humaine journalisée dans `AuditLog`

---

## Formule de refus de l'agent `forensic-analyst`

```
Refusé — affaiblirait la chaîne de possession (voir CLAUDE.md Règle [N])
```

---

## Références légales — État de vérification

Fichier source : `legal_references.yaml`

| Référence | Usage | État | Note |
|-----------|-------|------|------|
| LPC art. 31.1 | Authentification documents électroniques — **volet judiciaire** | `a_valider` | À ajouter dans legal_references.yaml |
| LPRPDE / Loi 25 | Protection données personnelles et biométriques | `a_valider` | Inclut obligations banques de données biométriques |
| LRPCFAT | KYC/AML FINTRAC | `a_valider` | — |
| C-25 — Strong and Free Elections Act | Deepfakes électoraux (art. 480.1 CEA) | `confirme` | Sanction royale **18 juin 2026**, 45e Parl. 1re session. Infraction pénale, pas un marché de détection. |
| RFC-3161 | TSA horodatage | `confirme` | — |
| CAN/DGSI 120 | KYC biométrique uniquement | `a_valider` | ⚠️ Concerne l'**authentification biométrique**, pas la forensique judiciaire. À retirer du volet judiciaire. |
| CPCivQ art. 282-293 | Procédure civile Québec | `a_valider` | ⚠️ Articles non confirmés — validation par juriste québécois requise |
| C2PA-1.3 | Provenance des contenus | `confirme` | — |

> CAN/DGSI 120 = programme de travail « Use of biometrics for authentication » (comité Biométrie de l'Institut).
> Utile pour le module KYC, hors sujet pour la forensique judiciaire.

---

## Licences des jeux de données — À valider avant mise en marché

| Dataset / Modèle | Usage prévu | Licence commerciale ? |
|------------------|-------------|----------------------|
| FaceForensics++ | Entraînement/évaluation | ❌ CC BY-NC 4.0 — non commercial confirmé (contact TUM) |
| DFDC (Meta/Facebook) | Entraînement/évaluation | ❌ Non-commercial research license Meta — confirmé |
| ASVspoof 2019 | Audio | ❌ CC BY-NC-ND 4.0 — non commercial confirmé (vérifier 2021/2024) |
| insightface/buffalo_s | KYC / biométrie | ❌ Modèles non-commercial (code MIT OK) — contact recognition-oss-pack@insightface.ai |

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| API | FastAPI 0.115 + Pydantic v2 |
| Auth | JWT RS256 + MFA TOTP + RBAC |
| Base de données | PostgreSQL + SQLAlchemy 2.0 (dev : SQLite ⚠️ voir note) |
| Queue | Celery + Redis |
| Stockage fichiers | MinIO |
| ML | PyTorch CPU + ONNX Runtime + MediaPipe |
| PDF | WeasyPrint + Jinja2 |
| Chiffrement | AES-256-GCM (at-rest) + RSA-4096 (JWT + audit) + bcrypt/12 (mots de passe) |
| Hachage | SHA-256 (probatoire/TSA) + Blake3 (perf) |
| Horodatage | RFC 3161 — dev : FreeTSA.org · **prod : TSA accréditée à configurer (Entrust Canada)** |

> ⚠️ **Divergence dev/prod** : SQLite en dev, PostgreSQL en prod.
> Risque de comportements divergents. **La CI doit tourner sur PostgreSQL**, pas SQLite.

---

## Structure du projet

```
deepfake_detector/
├── backend/
│   ├── main.py                      # FastAPI + _validate_production_config()
│   ├── config.py                    # Settings (encryption_key, pas fernet_key)
│   ├── core/
│   │   ├── chain_of_custody.py      # SHA-256 + Blake3 + TSA RFC 3161
│   │   ├── encryption.py            # AES-256-GCM
│   │   ├── ingestion.py             # Upload + validation + chain of custody
│   │   ├── security.py              # JWT RS256 + MFA + RBAC
│   │   └── token_blocklist.py       # Redis JWT blocklist
│   ├── engines/
│   │   ├── fusion.py                # Fusion pondérée + _require_metrics() + _METRICS_PATH
│   │   ├── video_engine.py          # EfficientNet-B4 + ResNet-50+LSTM [experimental]
│   │   ├── audio_engine.py          # Wav2Vec2 [disabled — tête aléatoire]
│   │   ├── biometric_engine.py      # EAR + AU [experimental]
│   │   ├── metadata_engine.py       # Heuristiques métadonnées [experimental]
│   │   ├── document_engine.py       # Analyse documents [experimental]
│   │   └── text_engine.py           # Analyse texte [experimental]
│   ├── models/
│   │   ├── audit_log.py             # entry_hash + signature_b64
│   │   ├── media_file.py            # hash_md5 nullable/déprécié
│   │   └── ...
│   ├── routers/                     # auth, cases, analyze, reports, kyc, webhooks, analytics
│   ├── tasks/                       # Celery async
│   └── reporting/                   # PDF WeasyPrint + templates Jinja2
├── evaluation/
│   ├── metrics/                     # metrics.json (requis Règle 7 — absent = experimental)
│   ├── model_cards/
│   ├── protocols/baseline_protocol.yaml
│   ├── robustness/ · drift/ · validation_reports/ · datasets/
├── training/
│   ├── dataset_prep.py              # Split strict par identité
│   ├── eval_runner.py               # Runner cross-dataset
│   ├── metrics_schema.py            # Schéma Pydantic pour metrics.json
│   └── video/ audio/ document/ text/ common/ configs/
├── legal_references.yaml            # Source unique des références légales (Règle 9)
├── datasets/registry.yaml           # Inventaire + licences datasets
└── docs/adr/                        # ADR-001 à ADR-006
```

---

## Agents Ruflo disponibles

| Agent | Rôle | Invoquer quand |
|-------|------|----------------|
| `core/coder` | FastAPI Python | Nouvelle route, service, modèle |
| `core/tester` | Tests pytest | Génération tests, couverture |
| `core/reviewer` | Code review | Avant tout merge |
| `security/security-auditor` | Audit OWASP/LPRPDE | Audit sécurité |
| `project/ml-engineer` | PyTorch/ONNX | Moteurs deepfake |
| `project/forensic-analyst` | Chain of custody | Intégrité légale — a un mandat de refus |
| `goal/goal-planner` | Planification GOAP | Fonctionnalités complexes |

---

## Lancement du projet

```bash
source /home/kali/deepfake_detector/.venv/bin/activate
cd /home/kali/deepfake_detector/backend

# API seule (dev — SQLite)
uvicorn main:app --reload --port 8080

# Stack complète (Docker — PostgreSQL + Redis + MinIO)
cd /home/kali/deepfake_detector
docker-compose up
```

---

## Variables d'environnement clés

| Variable | Description |
|----------|-------------|
| `ENCRYPTION_KEY` | Clé AES-256-GCM base64url 32 octets — **obligatoire en prod** |
| `FERNET_KEY` | Déprécié — migration uniquement |
| `APP_ENV` | `development` / `production` |
| `DEBUG` | `false` en prod (forcé par garde démarrage) |
| `MFA_REQUIRED` | `true` en prod (forcé par garde démarrage) |
| `TSA_URL` | dev : freetsa.org · **prod : TSA accréditée (à configurer)** |
| `ALLOW_EXPERIMENTAL_ENGINES` | `true` en dev · décision à prendre pour la prod |

---

## ADRs bindants

| ADR | Décision |
|-----|----------|
| ADR-001 | Auth JWT RS256 (pas HS256) |
| ADR-002 | Chain of custody SHA-256 + Blake3 + TSA RFC 3161 |
| ADR-003 | Moteurs ML — statuts validated/experimental/disabled |
| ADR-004 | Celery pour tâches async ML (pas BackgroundTasks) |
| ADR-005 | Security headers obligatoires |
| ADR-006 | Migrations Alembic |

---

## Directives priorisées (évaluation 2026-10-04)

| Priorité | Action |
|----------|--------|
| 1 | ✅ 2026-10-04 Architecture de greffons — `EnginePlugin` ABC, `PluginRegistry`, `fuse_from_registry()`, `Verdict.abstain`. 38 tests. |
| 2 | Vérification C2PA / Content Credentials (preuve non basée sur la détection) |
| 3 | Valider un premier moteur selon la Règle 10, jeu en conditions réelles + robustesse compression |
| 4 | Revoir legal_references.yaml : C-25 (état Parlement), retirer DGSI 120 du judiciaire, ajouter LPC art. 31.1, juriste québécois pour CPCivQ |
| 5 | Audit licences datasets et modèles pour usage commercial |
| 6 | Recruter expert forensique + 2–3 pilotes (litige, assurance) |
| 7 | Prod : TSA accréditée, clés KMS/HSM, CI PostgreSQL, analyse vie privée (Loi 25 + biométrie) |

---

## Ce qui nécessite une décision humaine (non codable)

- Valider l'état exact de C-25 sur le site du Parlement avant tout usage
- Reclasser ou retirer CAN/DGSI 120 du volet judiciaire (confirmer avec juriste)
- Vérifier les licences commerciales des jeux de données et modèles
- Recruter expert forensique et juriste (indispensable pour les pilotes)
- Remplacer FreeTSA.org par une TSA accréditée avant la production
- Migrer les clés RSA des fichiers `.pem` locaux vers HSM/KMS
- Décision sur `allow_experimental_engines` en production (score → 0,0 si False)
- Obligations Loi 25 sur les banques de données biométriques (Québec)
