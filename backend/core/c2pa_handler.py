"""Lecture des manifestes C2PA (Coalition for Content Provenance and Authenticity).

Phase 5 — Provenance cryptographique des médias.
Graceful degradation si le package c2pa n'est pas disponible.
"""
from __future__ import annotations

import json
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


class C2PAUnavailable(Exception):
    """Levée si le package c2pa n'est pas installé."""


def _get_c2pa():
    """Retourne le module c2pa ou lève C2PAUnavailable."""
    try:
        import c2pa
        return c2pa
    except ImportError as exc:
        raise C2PAUnavailable("c2pa-python non installé") from exc


class C2PAHandler:
    """Interface de lecture des manifestes C2PA sur les fichiers médias."""

    @staticmethod
    def read_manifest(file_path: str | Path) -> dict | None:
        """
        Lit le manifeste C2PA d'un fichier.

        Retourne un dict structuré, ou None si :
        - c2pa n'est pas disponible
        - le fichier n'a pas de manifeste C2PA
        - le fichier n'est pas dans un format supporté
        """
        try:
            c2pa = _get_c2pa()
        except C2PAUnavailable:
            log.debug("c2pa_unavailable")
            return None

        path_str = str(file_path)
        try:
            reader = c2pa.Reader.try_create(format_or_path=path_str)
            if reader is None:
                return None

            manifest_json = reader.json()
            if not manifest_json:
                return None

            raw = json.loads(manifest_json)

            active_label = raw.get("active_manifest")
            manifests = raw.get("manifests", {})
            active = manifests.get(active_label) if active_label else None

            producer = None
            created_at = None
            if active:
                claim_gen = active.get("claim_generator", "")
                producer = claim_gen.split("/")[0].strip() if claim_gen else None
                assertions = active.get("assertions", [])
                for a in assertions:
                    if a.get("label") == "stds.schema-org.CreativeWork":
                        created_at = a.get("data", {}).get("dateCreated")
                        break

            validation_state = None
            try:
                validation_state = reader.get_validation_state()
            except Exception:
                pass

            return {
                "active_manifest": active_label,
                "manifests": manifests,
                "validation_status": str(validation_state) if validation_state is not None else "unknown",
                "producer": producer,
                "created_at": created_at,
            }
        except Exception as exc:
            log.warning("c2pa_read_failed", file=path_str, error=str(exc))
            return None

    @staticmethod
    def extract_assertions(manifest: dict) -> list[dict]:
        """Extrait les assertions C2PA du manifest actif."""
        if not manifest:
            return []
        active_label = manifest.get("active_manifest")
        manifests = manifest.get("manifests", {})
        active = manifests.get(active_label) if active_label else None
        if not active:
            return []
        return active.get("assertions", [])

    @staticmethod
    def get_provenance_summary(manifest: dict | None) -> dict:
        """
        Retourne un résumé de provenance lisible par les analystes.

        {has_c2pa, producer, tool, created_at, actions_count, is_valid, validation_errors}
        """
        if not manifest:
            return {
                "has_c2pa": False,
                "producer": None,
                "tool": None,
                "created_at": None,
                "actions_count": 0,
                "is_valid": None,
                "validation_errors": [],
            }

        assertions = C2PAHandler.extract_assertions(manifest)
        actions_count = sum(1 for a in assertions if a.get("label") == "c2pa.actions")

        tool = None
        active_label = manifest.get("active_manifest")
        manifests = manifest.get("manifests", {})
        active = manifests.get(active_label) if active_label else None
        if active:
            software = active.get("claim_generator_info", [])
            if software:
                tool = software[0].get("name") if isinstance(software, list) else None

        validation = manifest.get("validation_status", "unknown")
        is_valid = validation in ("valid", "Valid")

        return {
            "has_c2pa": True,
            "producer": manifest.get("producer"),
            "tool": tool,
            "created_at": manifest.get("created_at"),
            "actions_count": actions_count,
            "is_valid": is_valid,
            "validation_errors": [] if is_valid else [validation],
        }
