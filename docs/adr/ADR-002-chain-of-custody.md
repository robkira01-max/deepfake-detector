# ADR-002 — Chain of Custody : Blake3 + TSA RFC3161

**Status** : Accepted  
**Date** : 2026-09-21  
**Auteur** : Ruflo Agent (forensic-analyst)

## Contexte

Les analyses deepfake peuvent être utilisées comme preuves légales en cour canadienne. L'intégrité des fichiers analysés doit être prouvable et infalsifiable.

## Décision

1. **Hash Blake3** calculé sur le fichier original AVANT tout traitement
2. **Timestamp TSA RFC3161** (freetsa.org) lié au hash pour preuve d'antériorité
3. **Stockage immuable** dans MinIO avec versioning activé
4. **Audit Log** horodaté et signé pour chaque action
5. **Signature RSA-4096** sur les rapports PDF finaux

## Justification

- Blake3 : plus rapide que SHA-256/SHA-3, résistant aux collisions, adapté aux gros fichiers médias
- TSA RFC3161 : standard légal accepté comme preuve d'antériorité (Loi sur la preuve au Canada)
- MinIO immuable : empêche toute modification postérieure (Object Lock)
- Audit Log signé : traçabilité complète, non-répudiation

## Conséquences

- **Obligatoire** : toute ingestion doit passer par `core/chain_of_custody.py`
- L'analyse se fait sur une COPIE, jamais l'original
- Les AuditLogs ne peuvent pas être modifiés une fois créés (pas de UPDATE dans le code)
- La clé `audit_private.pem` doit être séparée de `private.pem` JWT
- En cas d'indisponibilité TSA : stocker localement et réessayer (ne pas bloquer l'analyse)

## Fichiers liés

- `backend/core/chain_of_custody.py`
- `backend/core/ingestion.py`
- `backend/models/audit_log.py`
- `backend/models/media_file.py`
