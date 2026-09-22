---
name: security-audit
description: Lance un audit de sécurité complet sur DeepfakeDetector — OWASP, JWT, chain of custody, LPRPDE
---

# Security Audit — DeepfakeDetector

Lance l'agent `security-auditor` pour un audit complet du projet.

## Ce qui est audité

1. **`backend/core/security.py`** — JWT RS256, MFA TOTP, RBAC
2. **`backend/routers/auth.py`** — Routes d'auth, rate limiting
3. **`backend/core/chain_of_custody.py`** — Intégrité forensique
4. **`backend/core/ingestion.py`** — Upload, validation MIME, quarantaine
5. **`backend/config.py`** — Secrets dans .env, pas hardcodés
6. **`backend/main.py`** — CORS, trusted hosts, middleware sécurité
7. **`backend/.env`** — Variables sensibles (ne pas afficher les valeurs)

## Commandes de lancement

```bash
# Activer le venv
source /home/kali/deepfake_detector/.venv/bin/activate
cd /home/kali/deepfake_detector/backend

# Scan de secrets dans le code
grep -rn "password\s*=\s*['\"][^$]" --include="*.py" .
grep -rn "secret\s*=\s*['\"][^$]" --include="*.py" .

# Vérifier .gitignore couvre .env
grep -E "^\.env" ../.gitignore 2>/dev/null || echo "ATTENTION: .env pas dans .gitignore!"

# Dépendances avec CVEs connus
pip list --outdated
```

## Rapport de sortie

L'agent génère un rapport structuré avec :
- CRITIQUE / ÉLEVÉ / MOYEN / FAIBLE
- Fichier + numéro de ligne pour chaque finding
- Recommandation de correction concrète
- Score de risque global (1-10)
