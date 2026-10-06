# DeepfakeDetector Canada — v3.1.0

**Python 3.14 · FastAPI · PostgreSQL · 655+ tests**

---

## Avertissement fondamental

> Cet outil est **conçu pour soutenir l'évaluation de l'intégrité, de la provenance
> et de l'authenticité technique de contenus numériques, sans constituer à lui seul
> une conclusion d'expert ou une preuve définitive de manipulation.**

L'outil assiste l'expert humain — il ne le remplace pas.
Toute conclusion probatoire exige la revue et la signature d'un expert qualifié.

---

## Ce que c'est

Plateforme d'analyse forensique de médias synthétiques (deepfakes) :

- Ingestion de fichiers avec chaîne de possession cryptographique (SHA-256 + Blake3 + TSA RFC 3161)
- Pipeline d'analyse multi-moteurs (vidéo, biométrique, métadonnées, audio)
- Génération de rapports PDF/A-3 signés RSA-4096 avec horodatage certifié
- Journal d'audit immuable chaîné SHA-256
- Auth JWT RS256 + MFA TOTP + RBAC

## Ce que ce n'est pas

- **Pas un outil autonome de décision** : aucun moteur n'est `validated` à ce jour
  (voir [Statut des moteurs ML](#statut-des-moteurs-ml))
- **Pas prêt pour un pilote sans les critères listés en fin de document**
- **Pas un substitut à l'expertise humaine** (juriste, expert forensique)

---

## Marchés cibles

| Priorité | Marché | Justification |
|----------|--------|---------------|
| 1 | Litige / dossier de preuve numérique | Chaîne de possession + revue humaine |
| 2 | Assurance / KYC | Hébergement canadien + rapport bilingue |
| 3 | Élections (Loi C-25) | La loi crée une infraction, pas un marché — faible priorité commerciale |

---

## Prérequis

- Python 3.11+
- Docker Compose v2
- Clés RSA-4096 : `make keys-gen` (générées dans `backend/keys/`)
- Fichier `.env` : copier `.env.example` et remplir les variables

---

## Installation

```bash
git clone <repo>
cd deepfake_detector

# Générer les clés RSA
make keys-gen

# Créer l'environnement virtuel et installer les dépendances
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copier et configurer l'environnement
cp .env.example .env
# Éditer .env : ENCRYPTION_KEY, DATABASE_URL, JWT_*, TSA_URL, etc.

# Appliquer les migrations
make migrate
```

---

## Lancement

```bash
# Développement local (3 processus séparés)
make api       # API FastAPI sur localhost:8001
make worker    # Celery worker (analyses ML)
make demo      # Serveur demo sur localhost:8080

# Ou avec Docker Compose (stack complète)
make up        # API :80, MinIO :9001, Flower :5555

# Vérification
curl http://localhost:8001/health
```

---

## Commandes

```
make api          — API FastAPI seule (port 8001)
make worker       — Celery worker ML
make test         — Suite unitaire (655+ tests, skip integration)
make test-fast    — Suite rapide (stop au 1er échec)
make coverage     — Tests + rapport couverture HTML dans docs/coverage/
make lint         — Vérification types (mypy)
make bandit       — Scan SAST sécurité (rapport dans docs/security/)
make up / down    — Docker Compose
make migrate      — Alembic upgrade head
make keys-gen     — Génère les clés RSA-4096
```

---

## Architecture

Voir [`docs/architecture.md`](docs/architecture.md) pour le schéma complet (services, pipeline d'analyse, ADRs).

---

## Statut des moteurs ML

Lu depuis `backend/engines/fusion.py` — `ENGINE_DEFAULT_STATUS` :

| Moteur | Algorithme | Statut | Note |
|--------|-----------|--------|------|
| `texture` | EfficientNet-B4 | `experimental` | Poids ImageNet, non fine-tuné deepfake |
| `temporal` | ResNet-50 + LSTM | `experimental` | Poids ImageNet |
| `rppg` | CHROM | `experimental` | Fonctionnel, non benchmarké deepfake |
| `biometrics` | EAR + Action Units | `experimental` | Fonctionnel, non benchmarké deepfake |
| `phase` | STFT phase | `experimental` | Fonctionnel, non benchmarké deepfake |
| `metadata` | Heuristiques | `experimental` | Fonctionnel, non benchmarké deepfake |
| `audio` | Wav2Vec2 | `disabled` | Tête de classification aléatoire — non entraîné |

**Aucun moteur `validated` à ce jour.** Un moteur passe à `validated` seulement après :
metrics.json sur jeu indépendant + model card + protocole + approbation humaine journalisée.

Avec `ALLOW_EXPERIMENTAL_ENGINES=false`, le score de fusion est **0.0**.

---

## Sécurité

- Chiffrement at-rest : **AES-256-GCM** (nonce 96 bits par opération)
- Auth : **JWT RS256** + MFA TOTP obligatoire en production
- Journal : **SHA-256 chaîné** — toute altération casse la chaîne
- Signatures rapports : **RSA-4096 PSS**
- Horodatage : **RFC 3161** — dev : FreeTSA.org · **prod : TSA accréditée à configurer**

Variables d'environnement : voir `.env.example`. Ne jamais commiter `.env`.

---

## Références légales

Fichier source : [`legal_references.yaml`](legal_references.yaml)

Les références marquées `status: a_valider` ne doivent **jamais** apparaître dans un rapport émis.
Toutes doivent être validées par un juriste avant usage en contexte probatoire.

---

## Critères d'entrée en pilote

Ces critères sont **nécessaires avant tout test avec de vraies données** :

| Critère | État |
|---------|------|
| Au moins 1 moteur `validated` (metrics.json + model card + protocole + audit) | ❌ |
| TSA accréditée (pas FreeTSA.org) configurée en production | ❌ |
| Clés RSA dans HSM/KMS (pas fichiers `.pem` locaux) | ❌ |
| CI sur PostgreSQL (pas SQLite) | ❌ |
| Références légales `a_valider` validées par un juriste | ❌ |
| Licences des datasets (FaceForensics++, DFDC, ASVspoof, insightface) auditées | ❌ |
| Expert forensique recruté | ❌ |

---

## Chain of custody

Voir [`docs/chain-of-custody-checklist.md`](docs/chain-of-custody-checklist.md) pour la checklist opérationnelle à utiliser avant tout dépôt en preuve.
