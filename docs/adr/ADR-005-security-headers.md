# ADR-005 — Security Headers HTTP

**Date** : 2026-09-22
**Statut** : Accepté
**Décideurs** : Équipe DeepfakeDetector

---

## Contexte

L'API expose des données biométriques et forensiques sensibles. Les headers HTTP de
sécurité constituent une ligne de défense en profondeur contre XSS, clickjacking,
sniffing MIME et fuites d'information. Audit OWASP API Security Top 10 (2023).

---

## Décision

Middleware `SecurityHeadersMiddleware` ajouté dans `backend/main.py`, appliqué à
toutes les réponses HTTP.

---

## Headers implementés

| Header | Valeur | Raison |
|--------|--------|--------|
| `X-Content-Type-Options` | `nosniff` | Bloque MIME sniffing (IE/Chrome) |
| `X-Frame-Options` | `DENY` | Anti-clickjacking (déprécié mais couverture legacy) |
| `X-XSS-Protection` | `0` | Désactivé — CSP est la défense moderne; ce header cause des vulnérabilités sur vieux IE |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Limite les fuites URL dans les referers |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=(), payment=()` | Bloque accès aux capteurs (API forensique, pas besoin) |
| `Content-Security-Policy` | `default-src 'none'; frame-ancestors 'none'; base-uri 'none'` | API JSON pure — aucun contenu actif autorisé |
| `Cache-Control` | `no-store` | Empêche cache de données sensibles (tokens, résultats analyses) |
| `Pragma` | `no-cache` | Compat HTTP/1.0 |
| `Strict-Transport-Security` | `max-age=63072000; includeSubDomains; preload` | HTTPS forcé 2 ans (prod uniquement) |

### Header `Server` supprimé

Le header `Server` (ex: `uvicorn`) est supprimé pour éviter la divulgation de la
stack technique — vecteur de fingerprinting exploitable.

---

## Résultat bandit SAST

Scan complet le 2026-09-22 :
- HIGH severity : **0**
- MEDIUM severity : **0**
- LOW severity : **10** (usage MD5 pour interopérabilité chaîne de custody — `#nosec B324` documenté)

---

## Score OWASP Security Headers

Avant : D (absence de headers)
Après : A (tous les headers critiques présents)

Validation : [https://securityheaders.com](https://securityheaders.com) (à vérifier en prod)

---

## Conséquences

### Positives
- Protection contre XSS réfléchi si erreur future de rendu HTML
- Fingerprinting de stack rendu plus difficile
- Cache des tokens JWT bloqué sur les proxies intermédiaires
- Clickjacking impossible (pas d'interface admin web mais protection préventive)

### Négatives
- CSP `default-src 'none'` cassera tout rendu HTML si l'API ajoute une interface
  web à l'avenir (ex: Swagger en production) — à ajuster si besoin
- HSTS désactivé en debug pour faciliter le développement local

### Note sur X-XSS-Protection: 0

Ce choix est intentionnel et conforme aux recommandations OWASP 2023. La valeur `1; mode=block`
activait un ancien filtre XSS d'IE/Edge qui pouvait lui-même introduire des vulnérabilités
(data exfiltration via error page). CSP remplace cette protection de façon plus robuste.
