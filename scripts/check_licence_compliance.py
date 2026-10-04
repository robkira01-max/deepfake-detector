"""
Reads datasets/registry.yaml and model_weights entries, reports any non-commercial
items, and exits with code 1 if the --strict flag is passed and blockers exist.

Usage:
  python scripts/check_licence_compliance.py            # informational
  python scripts/check_licence_compliance.py --strict   # CI gate (exit 1 on blockers)
  python scripts/check_licence_compliance.py --json     # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not installed — run: pip install pyyaml")

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "datasets" / "registry.yaml"

_STATUS_LABELS = {
    "libre": "✅ LIBRE",
    "non_commercial_uniquement": "❌ NON-COMMERCIAL",
    "audit_requis": "⚠️  AUDIT REQUIS",
    "inconnu": "❓ INCONNU",
}


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        sys.exit(f"Fichier introuvable : {REGISTRY_PATH}")
    with REGISTRY_PATH.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        sys.exit("Format YAML invalide.")
    return data


def _entries(data: dict) -> list[dict]:
    out: list[dict] = []
    for section_key in ("datasets", "model_weights"):
        for item in data.get(section_key, []):
            item["_section"] = section_key
            out.append(item)
    return out


def _check_entry(entry: dict) -> dict:
    status = entry.get("statut_licence", "inconnu")
    commercial = entry.get("usage_commercial")
    if commercial is True:
        status = "libre"
    elif commercial is False:
        status = "non_commercial_uniquement"
    return {
        "id": entry.get("id", "?"),
        "nom": entry.get("nom", "?"),
        "section": entry.get("_section", "?"),
        "statut": status,
        "label": _STATUS_LABELS.get(status, f"? {status}"),
        "bloquer": status == "non_commercial_uniquement",
        "notes": (entry.get("notes") or "").strip().replace("\n", " "),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérifie les licences du registre.")
    parser.add_argument("--strict", action="store_true",
                        help="Retourne exit code 1 si des éléments bloquants existent")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Sortie JSON machine-readable")
    args = parser.parse_args()

    data = _load_registry()
    entries = _entries(data)
    results = [_check_entry(e) for e in entries]
    blockers = [r for r in results if r["bloquer"]]

    if args.as_json:
        print(json.dumps({
            "total": len(results),
            "blockers_count": len(blockers),
            "entries": results,
            "commercial_readiness": data.get("commercial_readiness", {}),
        }, ensure_ascii=False, indent=2))
        sys.exit(1 if (args.strict and blockers) else 0)

    print(f"\n{'─'*60}")
    print("  AUDIT LICENCES — DeepfakeDetector Canada")
    print(f"  Source : {REGISTRY_PATH.relative_to(REGISTRY_PATH.parents[2])}")
    print(f"{'─'*60}\n")

    for r in results:
        print(f"  {r['label']:<26}  [{r['section']}]  {r['id']}")
        print(f"  {'':26}  {r['nom']}")
        print()

    print(f"{'─'*60}")
    print(f"  Total : {len(results)}  |  Bloquants : {len(blockers)}  |  Libres : {len(results)-len(blockers)}")
    print(f"{'─'*60}\n")

    if blockers:
        print("  ÉLÉMENTS BLOQUANTS POUR UN USAGE COMMERCIAL :\n")
        conclusion = (data.get("commercial_readiness") or {}).get("conclusion", "")
        if conclusion:
            for line in conclusion.strip().splitlines():
                print(f"  {line.strip()}")
        else:
            for b in blockers:
                print(f"  • {b['id']} — {b['nom']}")
        print()

    if args.strict and blockers:
        print("  ⛔  Mode --strict : exit 1 (licences non commerciales présentes)")
        sys.exit(1)
    elif not blockers:
        print("  ✅  Aucun bloquant — tous les assets sont commercialement libres.")
    else:
        print("  ℹ️   Mode informatif (pas de --strict) — aucune action bloquée.")


if __name__ == "__main__":
    main()
