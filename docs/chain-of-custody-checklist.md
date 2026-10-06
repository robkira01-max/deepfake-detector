# Checklist — Chaîne de possession numérique

**Version 1.0 — DeepfakeDetector Canada v3.1.0**

---

> **Avertissement**
>
> Cet outil est conçu pour soutenir l'évaluation de l'intégrité, de la provenance
> et de l'authenticité technique de contenus numériques, **sans constituer à lui seul
> une conclusion d'expert ou une preuve définitive de manipulation.**
>
> Cette checklist est un aide-mémoire opérationnel. Elle ne constitue pas un avis
> juridique. Toute conclusion probatoire exige la revue et la signature d'un expert
> qualifié ainsi que la supervision d'un avocat.

---

## A. Ingestion du fichier

- [ ] SHA-256 calculé et enregistré (`compute_hashes()` → `HashBundle.sha256`)
- [ ] Blake3 calculé et enregistré (`HashBundle.blake3` — déduplication)
- [ ] Horodatage TSA RFC 3161 obtenu (`stamp_file()` → `TSAToken`)
  - En production : TSA accréditée (pas FreeTSA.org)
- [ ] `AuditLog FILE_INGESTED` créé avec `entry_hash` et `previous_entry_hash`
- [ ] Fichier stocké dans MinIO sans aucune modification
- [ ] Hash SHA-256 vérifié après stockage (`verify_file_integrity()`)

---

## B. Analyse

- [ ] `AuditLog ANALYSIS_STARTED` créé
- [ ] Liste des moteurs actifs notée (`ALLOW_EXPERIMENTAL_ENGINES` : True / False)
- [ ] Statut de chaque moteur noté (`experimental` / `disabled`)
  - Aucun moteur n'est `validated` en v3.1 — le consigner explicitement
- [ ] Résultats bruts de chaque moteur enregistrés (`Analysis.engine_scores`)
- [ ] Score de fusion noté avec le flag `allow_experimental_engines`
- [ ] `AuditLog ANALYSIS_COMPLETED` créé

> Si `ALLOW_EXPERIMENTAL_ENGINES=False` : score = 0.0 — l'analyse ne peut pas
> soutenir une conclusion technique sur la manipulation.

---

## C. Génération du rapport

- [ ] PDF généré avec WeasyPrint (`generate_pdf_report()`)
- [ ] SHA-256 du PDF calculé et enregistré (`Report.report_hash_sha256`)
- [ ] PDF signé RSA-4096 PSS (`Report.signature_b64`)
- [ ] TSA RFC 3161 appliqué au PDF (`Report.tsa_token_b64`)
- [ ] `AuditLog REPORT_GENERATED` créé
- [ ] Rapport marqué `is_signed=True`
- [ ] Numéro de rapport unique noté (`Report.report_number`)

---

## D. Vérification de la chaîne d'audit

- [ ] `verify_audit_chain(db)` exécuté
- [ ] Résultat `ok=True` et `errors=[]` confirmé
- [ ] En cas d'erreur : NE PAS déposer le rapport — investiguer avant tout dépôt

---

## E. Avant dépôt en preuve

### Références légales
- [ ] Toutes les références légales citées ont `status: confirme` dans `legal_references.yaml`
- [ ] Aucune référence `status: a_valider` n'apparaît dans le rapport émis
- [ ] Vérification de la version consolidée des lois sur `laws-lois.justice.gc.ca`

### Formulation du rapport
- [ ] Le rapport contient la formulation exacte de la Règle 8 :
  *« conçu pour soutenir l'évaluation de l'intégrité, de la provenance et de
  l'authenticité technique de contenus numériques, sans constituer à lui seul
  une conclusion d'expert ou une preuve définitive de manipulation »*
- [ ] Aucune formulation définitive : ni « conforme », ni « recevable en cour »,
  ni « validé par un expert », ni « preuve de manipulation »

### Validation humaine
- [ ] Un expert humain qualifié a relu l'intégralité du rapport
- [ ] L'expert a signé le rapport manuellement (en plus de la signature numérique)
- [ ] Un avocat a supervisé la préparation du dossier

---

## F. Conservation post-dépôt

- [ ] Fichier original archivé (hash SHA-256 identique à `MediaFile.hash_sha256`)
- [ ] Rapport PDF archivé dans MinIO
- [ ] Journal d'audit exporté (`GET /audit/export` → CSV) et archivé
- [ ] Durée de conservation conforme à `Case.retain_until`
- [ ] Accès au dossier restreint après clôture (`Case.status = archived`)

---

## G. En cas de contestation de la chaîne

Si la validité de la chaîne est contestée :

1. Exécuter `verify_audit_chain(db)` → fournir le résultat complet au tribunal
2. Vérifier les `entry_hash` et `previous_entry_hash` de chaque entrée concernée
3. Présenter les tokens TSA RFC 3161 (vérifiables publiquement)
4. Présenter les signatures RSA-4096 PSS du rapport PDF
5. Vérifier l'intégrité du fichier original : `verify_file_integrity(path, expected_sha256)`

---

## Références

- `backend/core/chain_of_custody.py` — `compute_hashes()`, `stamp_file()`, `verify_file_integrity()`, `sign_audit_entry()`
- `backend/models/audit_log.py` — `verify_audit_chain()`
- `legal_references.yaml` — liste des références légales avec statut de validation
- `docs/architecture.md` — pipeline d'analyse complet
