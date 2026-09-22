# ADR-001 — Authentification JWT RS256 + MFA TOTP

**Status** : Accepted  
**Date** : 2026-09-21  
**Auteur** : Ruflo Agent (security-auditor)

## Contexte

L'API DeepfakeDetector manipule des données biométriques sensibles (visages, voix) soumises à la LPRPDE. Une authentification robuste est obligatoire.

## Décision

- **JWT RS256** (asymétrique) plutôt que HS256 (symétrique)
- **MFA TOTP** obligatoire pour tous les comptes (`mfa_required=True`)
- **RBAC** à 3 niveaux : `admin`, `analyst`, `readonly`
- **Refresh token** avec rotation et blocklist Redis
- **Clés RSA-4096** séparées : une paire JWT, une paire audit

## Justification

- RS256 : la clé privée signe, la clé publique vérifie — pas besoin de partager le secret
- MFA TOTP : second facteur sans infrastructure SMS
- RBAC strict : principe de moindre privilège pour données forensiques
- Blocklist Redis : révocation immédiate possible (logout, compromission)

## Conséquences

- Les clés RSA sont générées au premier démarrage dans `/app/keys/`
- `core/security.py` est le seul module autorisé à manipuler les JWT
- Toute route accédant à des données de cas DOIT utiliser `Depends(get_current_user)`
- Le rôle `readonly` ne peut PAS créer, modifier ou supprimer — lecture seule

## Fichiers liés

- `backend/core/security.py`
- `backend/routers/auth.py`
- `backend/models/user.py`
