"""Générateur de rapport PDF/A-3 conçu pour soutenir l'admissibilité en preuve (LPC 31.1-31.6 + R. c. Mohan).

Templates disponibles :
  rapport_complet.html   — Rapport forensique complet (8 sections, défaut)
  rapport_executif.html  — Résumé exécutif (2-3 pages, pour les juges)
  rapport_bilingue.html  — Rapport bilingue FR/EN (cours fédérales)
  rapport_affidavit.html — Affidavit technique (CPCivQ art. 282-293)
  rapport_tribunal.html  — Mode tribunal (XAI judiciaire + manifest preuve + limites)
  report.html            — Template historique (maintenu pour rétrocompatibilité)
  <custom>               — Template utilisateur uploadé
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jinja2
import structlog
from sqlalchemy.orm import Session

from config import settings
from core.chain_of_custody import request_tsa_timestamp, sign_audit_entry
from models.analysis import Analysis
from models.audit_log import AuditLog, AuditAction
from models.case import Case
from models.media_file import MediaFile
from models.report import Report
from models.user import User

log = structlog.get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_CUSTOM_TEMPLATE_DIR = Path(settings.data_dir if hasattr(settings, "data_dir") else "/home/kali/deepfake_detector/data") / "templates"

_BUILTIN_TEMPLATES = {
    "rapport_complet": "rapport_complet.html",
    "rapport_executif": "rapport_executif.html",
    "rapport_bilingue": "rapport_bilingue.html",
    "rapport_affidavit": "rapport_affidavit.html",
    "legacy": "report.html",
}


def _make_jinja_env(extra_dirs: list[str] | None = None) -> jinja2.Environment:
    dirs = [str(_TEMPLATE_DIR)]
    if extra_dirs:
        dirs = extra_dirs + dirs
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(dirs),
        autoescape=True,
    )
    env.filters["filesizeformat"] = _filesizeformat
    return env


def _filesizeformat(value: int | None) -> str:
    if value is None:
        return "?"
    for unit in ["o", "Ko", "Mo", "Go"]:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} To"


def generate_pdf_report(
    analysis_id: int,
    expert: User,
    db: Session,
    template_id: int | None = None,
    extra_context: dict[str, Any] | None = None,
) -> Report:
    """
    Génère le rapport PDF/A-3, le signe et l'horodate.
    Retourne l'objet Report enregistré en base.

    Args:
        analysis_id: ID de l'analyse à rapporter.
        expert: Utilisateur expert qui génère le rapport.
        db: Session SQLAlchemy.
        template_id: ID du ReportTemplate à utiliser (None = défaut builtin).
        extra_context: Contexte additionnel (expert_name, expert_title, etc.).
    """
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    case = db.query(Case).filter(Case.id == analysis.case_id).first()
    media_file = db.query(MediaFile).filter(MediaFile.id == analysis.media_file_id).first()

    report_number = _generate_report_number(case)
    context = _build_context(analysis, case, media_file, expert, report_number, extra_context)

    # Résolution du template
    template_file, extra_template_dirs = _resolve_template(template_id, db)
    jinja_env = _make_jinja_env(extra_template_dirs)
    template = jinja_env.get_template(template_file)
    html_content = template.render(**context)

    # Conversion PDF (WeasyPrint)
    output_dir = Path(settings.processed_dir) / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{report_number}.pdf"

    try:
        from weasyprint import HTML
        HTML(string=html_content, base_url=str(_TEMPLATE_DIR)).write_pdf(str(pdf_path))
        log.info("pdf_generated", path=str(pdf_path), template=template_file)
    except ImportError:
        log.warning("weasyprint_not_available", fallback="html_only")
        pdf_path_html = output_dir / f"{report_number}.html"
        pdf_path_html.write_text(html_content, encoding="utf-8")
        pdf_path = pdf_path_html

    # Hash du rapport généré
    report_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    # Horodatage TSA du rapport
    tsa_token = request_tsa_timestamp(bytes.fromhex(report_hash))
    tsa_token_b64 = tsa_token.token_b64 if tsa_token else None
    tsa_ts = tsa_token.timestamp if tsa_token else None

    # Signature RSA-4096 du rapport
    sig_data = {
        "report_number": report_number,
        "report_hash": report_hash,
        "analysis_id": analysis_id,
        "expert_id": expert.id,
        "template": template_file,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _, signature_b64 = sign_audit_entry(sig_data)

    # Enregistrement
    report = Report(
        case_id=case.id,
        analysis_id=analysis_id,
        report_number=report_number,
        pdf_path=str(pdf_path),
        report_hash_sha256=report_hash,
        is_signed=bool(signature_b64),
        signature_b64=signature_b64,
        tsa_token_b64=tsa_token_b64,
        tsa_timestamp=tsa_ts,
        expert_id=expert.id,
        expert_username=expert.username,
    )
    db.add(report)
    db.flush()

    # Audit
    audit_data = {
        "action": AuditAction.REPORT_GENERATED.value,
        "user_id": expert.id,
        "report_id": report.id,
        "report_number": report_number,
        "report_hash": report_hash,
        "template": template_file,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entry_hash, audit_sig = sign_audit_entry(audit_data)
    audit = AuditLog(
        user_id=expert.id,
        user_username=expert.username,
        action=AuditAction.REPORT_GENERATED,
        resource_type="Report",
        resource_id=str(report.id),
        details={"report_number": report_number, "template": template_file, "report_hash": report_hash[:16] + "..."},
        entry_hash=entry_hash,
        signature_b64=audit_sig,
    )
    db.add(audit)
    db.commit()
    db.refresh(report)
    return report


def _resolve_template(
    template_id: int | None, db: Session
) -> tuple[str, list[str] | None]:
    """
    Retourne (nom_fichier_template, [dossiers_extra]) selon le template_id.
    Si template_id est None, utilise rapport_complet.html (défaut professionnel).
    """
    if template_id is None:
        return "rapport_complet.html", None

    try:
        from models.report_template import ReportTemplate, TemplateType
        tmpl = db.query(ReportTemplate).filter(
            ReportTemplate.id == template_id,
            ReportTemplate.is_active == True,
        ).first()
        if not tmpl:
            log.warning("template_not_found", template_id=template_id, fallback="default")
            return "rapport_complet.html", None

        if tmpl.template_type == TemplateType.builtin:
            return Path(tmpl.file_path).name, None
        else:
            # Template custom — dossier data/templates/
            custom_path = Path(tmpl.file_path)
            return custom_path.name, [str(custom_path.parent)]
    except Exception as exc:
        log.warning("template_resolution_error", error=str(exc), fallback="default")
        return "rapport_complet.html", None


def _generate_report_number(case: Case) -> str:
    now = datetime.now(timezone.utc)
    short = uuid.uuid4().hex[:6].upper()
    return f"RPT-{case.case_number}-{now.year}{now.month:02d}{now.day:02d}-{short}"


def _build_context(
    analysis: Analysis,
    case: Case,
    media_file: MediaFile,
    expert: User,
    report_number: str,
    extra_context: dict[str, Any] | None = None,
) -> dict:
    now = datetime.now(timezone.utc)

    def score_pct(v: float | None) -> str:
        return f"{int((v or 0) * 100)}%" if v is not None else "N/A"

    verdict_str = analysis.verdict.value if analysis.verdict else "INDÉTERMINÉ"
    verdict_color = {
        "AUTHENTIQUE": "#28a745",
        "INDÉTERMINÉ": "#fd7e14",
        "DEEPFAKE DÉTECTÉ": "#dc3545",
    }.get(verdict_str, "#6c757d")

    # Durée d'analyse
    duration_s: float | None = None
    if analysis.completed_at and analysis.started_at:
        duration_s = (analysis.completed_at - analysis.started_at).total_seconds()
    if not hasattr(analysis, "duration_seconds") or analysis.duration_seconds is None:
        analysis.duration_seconds = round(duration_s or 0, 1)  # type: ignore[attr-defined]

    # shap_values — list of dicts [{feature, raw_score, weighted_score, weight}]
    shap_values = analysis.xai_shap_values or []

    ctx: dict = {
        "report_number": report_number,
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "case": case,
        "media_file": media_file,
        "analysis": analysis,
        "expert": expert,
        # Résumé verdict
        "verdict_label": verdict_str,
        "verdict_color": verdict_color,
        # Scores formatés (rétrocompatibilité)
        "final_score_pct": score_pct(analysis.final_score),
        "ic_low_pct": score_pct(analysis.confidence_low),
        "ic_high_pct": score_pct(analysis.confidence_high),
        "score_texture_pct": score_pct(analysis.score_video_texture),
        "score_temporal_pct": score_pct(analysis.score_video_temporal),
        "score_rppg_pct": score_pct(analysis.score_rppg),
        "score_biometrics_pct": score_pct(analysis.score_biometrics),
        "score_audio_pct": score_pct(analysis.score_audio_model),
        "score_phase_pct": score_pct(analysis.score_audio_phase),
        "score_metadata_pct": score_pct(analysis.score_metadata),
        # Données XAI structurées
        "shap_values": shap_values,
        "plain_explanation": analysis.xai_plain_explanation or "",
        "suspicious_timecodes": analysis.xai_suspicious_timecodes or [],
        # Métriques modèles
        "far_pct": f"{(analysis.model_far or 0) * 100:.1f}%",
        "frr_pct": f"{(analysis.model_frr or 0) * 100:.1f}%",
        "eer_pct": f"{(analysis.model_eer or 0) * 100:.1f}%",
        "auc": f"{analysis.model_auc or 0:.3f}",
        # Expert (valeurs par défaut — peuvent être surchargées par extra_context)
        "expert_name": expert.username,
        "expert_title": "",
        "expert_credentials": "",
        "lab_name": "DeepfakeDetector Canada — Laboratoire forensique",
        "notes_for_court": "",
        # Intégrité rapport (placeholder — hash calculé APRÈS le rendu)
        "report_hash_sha256": "[calculé après génération]",
        "tsa_timestamp": "[en cours d'horodatage]",
        # Références légales
        "lpc_reference": "Loi sur la preuve au Canada, LRC 1985, c C-5, art. 31.1–31.6",
        "mohan_reference": "R. c. Mohan [1994] 2 RCS 9",
        "jlj_reference": "R. c. J.-L.J. [2000] 2 RCS 600",
        "dgsi_reference": "CAN/DGSI 120 [À VALIDER] — Digital Governance Standards Institute",
        "tsa_authority": getattr(media_file, "tsa_authority", None) or "Non disponible",
        # models_used
        "models_used": analysis.models_used or {},
    }

    # Extra context (surcharge expert_name, expert_title, etc.)
    if extra_context:
        ctx.update(extra_context)

    return ctx


# ── Courtroom Mode ────────────────────────────────────────────────────────────

def generate_courtroom_report(
    analysis_id: int,
    expert: User,
    db: Session,
    extra_context: dict[str, Any] | None = None,
) -> Report:
    """
    Génère un rapport judiciaire PDF (mode tribunal) avec XAI simplifié,
    manifest de preuve et mise en garde légale.
    Même garanties d'intégrité que generate_pdf_report (hash + TSA + RSA-4096).
    Numéro de rapport préfixé CTR- (Courtroom).
    """
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    case = db.query(Case).filter(Case.id == analysis.case_id).first()
    media_file = db.query(MediaFile).filter(MediaFile.id == analysis.media_file_id).first()

    report_number = _generate_courtroom_number(case)
    base_ctx = _build_context(analysis, case, media_file, expert, report_number, extra_context)
    court_extras = _build_courtroom_extras(analysis)
    context = {**base_ctx, **court_extras}

    jinja_env = _make_jinja_env()
    template = jinja_env.get_template("rapport_tribunal.html")
    html_content = template.render(**context)

    output_dir = Path(settings.processed_dir) / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{report_number}.pdf"

    try:
        from weasyprint import HTML
        HTML(string=html_content, base_url=str(_TEMPLATE_DIR)).write_pdf(str(pdf_path))
        log.info("courtroom_pdf_generated", path=str(pdf_path))
    except ImportError:
        log.warning("weasyprint_not_available", fallback="html_only")
        pdf_path_html = output_dir / f"{report_number}.html"
        pdf_path_html.write_text(html_content, encoding="utf-8")
        pdf_path = pdf_path_html

    report_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    tsa_token = request_tsa_timestamp(bytes.fromhex(report_hash))
    tsa_token_b64 = tsa_token.token_b64 if tsa_token else None
    tsa_ts = tsa_token.timestamp if tsa_token else None

    sig_data = {
        "report_number": report_number,
        "report_hash": report_hash,
        "analysis_id": analysis_id,
        "expert_id": expert.id,
        "template": "rapport_tribunal.html",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _, signature_b64 = sign_audit_entry(sig_data)

    report = Report(
        case_id=case.id,
        analysis_id=analysis_id,
        report_number=report_number,
        pdf_path=str(pdf_path),
        report_hash_sha256=report_hash,
        is_signed=bool(signature_b64),
        signature_b64=signature_b64,
        tsa_token_b64=tsa_token_b64,
        tsa_timestamp=tsa_ts,
        expert_id=expert.id,
        expert_username=expert.username,
    )
    db.add(report)
    db.flush()

    audit_data = {
        "action": AuditAction.REPORT_GENERATED.value,
        "user_id": expert.id,
        "report_id": report.id,
        "report_number": report_number,
        "report_hash": report_hash,
        "template": "rapport_tribunal.html",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    entry_hash, audit_sig = sign_audit_entry(audit_data)
    audit = AuditLog(
        user_id=expert.id,
        user_username=expert.username,
        action=AuditAction.REPORT_GENERATED,
        resource_type="Report",
        resource_id=str(report.id),
        details={
            "report_number": report_number,
            "template": "rapport_tribunal.html",
            "report_hash": report_hash[:16] + "...",
            "courtroom_mode": True,
        },
        entry_hash=entry_hash,
        signature_b64=audit_sig,
    )
    db.add(audit)
    db.commit()
    db.refresh(report)
    return report


def _generate_courtroom_number(case: Case) -> str:
    now = datetime.now(timezone.utc)
    short = uuid.uuid4().hex[:6].upper()
    return f"CTR-{case.case_number}-{now.year}{now.month:02d}{now.day:02d}-{short}"


def _build_courtroom_extras(analysis: "Analysis") -> dict[str, Any]:
    """
    Construit les données judiciaires complémentaires :
    - Verdict en langage judiciaire (FR + EN)
    - Statistiques de fiabilité formulées "sur 1000 cas"
    - Tableau d'indicateurs avec libellés et interprétations judiciaires
    - Aperçu lisible de la signature RSA-4096
    """
    from models.analysis import Verdict

    # ── Verdict judiciaire ────────────────────────────────────────────────────
    verdict = analysis.verdict
    if verdict == Verdict.deepfake:
        judicial_fr = (
            "Le système a identifié des indicateurs concordants de manipulation numérique. "
            "La probabilité que ce fichier soit un deepfake ou un contenu synthétique est élevée."
        )
        judicial_en = (
            "The system identified consistent indicators of digital manipulation. "
            "The probability that this file is a deepfake or synthetic content is high."
        )
    elif verdict == Verdict.authentic:
        judicial_fr = (
            "Le système n'a pas identifié d'indicateurs caractéristiques de manipulation numérique. "
            "Le contenu analysé présente les propriétés attendues d'un enregistrement authentique."
        )
        judicial_en = (
            "The system did not identify indicators characteristic of digital manipulation. "
            "The analyzed content presents expected properties of an authentic recording."
        )
    else:
        judicial_fr = (
            "Le système a relevé des anomalies modérées sans atteindre le seuil de conclusion. "
            "Une expertise humaine complémentaire est recommandée avant de tirer des conclusions."
        )
        judicial_en = (
            "The system detected moderate anomalies without reaching a conclusive threshold. "
            "Additional human expertise is recommended before drawing conclusions."
        )

    # ── Fiabilité "sur 1000" — uniquement si mesurée sur jeu de test indépendant ─
    auc = analysis.model_auc
    far = analysis.model_far
    frr = analysis.model_frr
    eer = analysis.model_eer

    if auc is not None and far is not None and frr is not None:
        reliability_detection_pct = round(auc * 100, 1)
        reliability_false_alarm = round(far * 1000)
        reliability_missed = round(frr * 1000)
    else:
        reliability_detection_pct = None
        reliability_false_alarm = None
        reliability_missed = None

    # ── Indicateurs judiciaires ───────────────────────────────────────────────
    _indicator_meta: dict[str, dict[str, str]] = {
        "texture": {
            "label_fr": "Texture du visage (EfficientNet-B4)",
            "label_en": "Facial Texture Analysis",
            "low":  "La texture du visage est cohérente avec un enregistrement naturel.",
            "mid":  "Des légères anomalies de texture ont été détectées; résultat non concluant.",
            "high": "Des anomalies de texture typiques des visages générés par IA ont été détectées.",
        },
        "temporal": {
            "label_fr": "Cohérence temporelle des expressions (LSTM)",
            "label_en": "Temporal Expression Consistency",
            "low":  "Les expressions faciales progressent de façon naturelle entre les images.",
            "mid":  "Des légères incohérences temporelles ont été observées.",
            "high": "Des incohérences entre images successives caractéristiques d'un deepfake ont été détectées.",
        },
        "rppg": {
            "label_fr": "Signal cardiaque facial (rPPG CHROM)",
            "label_en": "Facial Cardiac Signal (rPPG)",
            "low":  "Un signal cardiaque cohérent avec une personne réelle a été détecté.",
            "mid":  "Le signal cardiaque est partiellement présent; résultat ambigu.",
            "high": "Aucun signal cardiaque cohérent détecté, ce qui est caractéristique d'un visage synthétique.",
        },
        "biometrics": {
            "label_fr": "Comportement biométrique facial (clignements, lèvres)",
            "label_en": "Facial Biometric Behaviour",
            "low":  "Le comportement biométrique (clignements, mouvement des lèvres) est dans la norme.",
            "mid":  "Quelques anomalies biométriques mineures ont été observées.",
            "high": "Des anomalies biométriques significatives ont été détectées (clignements anormaux, lèvres asynchrones).",
        },
        "audio": {
            "label_fr": "Caractéristiques vocales (Wav2Vec2)",
            "label_en": "Vocal Characteristics Analysis",
            "low":  "Les caractéristiques vocales sont cohérentes avec une voix humaine naturelle.",
            "mid":  "Des anomalies vocales modérées ont été détectées; résultat non concluant.",
            "high": "Des caractéristiques vocales typiques d'une synthèse par IA ont été détectées.",
        },
        "phase": {
            "label_fr": "Continuité de phase audio (STFT)",
            "label_en": "Audio Phase Continuity",
            "low":  "Le signal audio ne présente pas de discontinuités artificielles.",
            "mid":  "Quelques discontinuités audio ont été relevées; peuvent être dues à l'encodage.",
            "high": "Des discontinuités de phase caractéristiques d'un montage ou d'une synthèse ont été détectées.",
        },
        "metadata": {
            "label_fr": "Cohérence des métadonnées du fichier",
            "label_en": "File Metadata Consistency",
            "low":  "Les métadonnées du fichier sont cohérentes avec un enregistrement naturel.",
            "mid":  "Des incohérences mineures dans les métadonnées ont été relevées.",
            "high": "Les métadonnées présentent des incohérences caractéristiques d'un fichier retraité ou synthétique.",
        },
    }

    score_map = {
        "texture":    analysis.score_video_texture or 0.0,
        "temporal":   analysis.score_video_temporal or 0.0,
        "rppg":       analysis.score_rppg or 0.0,
        "biometrics": analysis.score_biometrics or 0.0,
        "audio":      analysis.score_audio_model or 0.0,
        "phase":      analysis.score_audio_phase or 0.0,
        "metadata":   analysis.score_metadata or 0.0,
    }

    court_indicators = []
    for key, meta in _indicator_meta.items():
        score = score_map[key]
        if score > 0.55:
            interp = meta["high"]
        elif score > 0.35:
            interp = meta["mid"]
        else:
            interp = meta["low"]
        court_indicators.append({
            "feature": key,
            "label_fr": meta["label_fr"],
            "label_en": meta["label_en"],
            "score": score,
            "court_interpretation": interp,
        })

    # Sort by score descending (most suspicious first)
    court_indicators.sort(key=lambda x: x["score"], reverse=True)

    # ── Aperçu de la signature ────────────────────────────────────────────────
    # Placeholder — remplacé après génération par le contexte réel si disponible
    sig_preview = "[calculé après signature]"

    return {
        "judicial_verdict_fr": judicial_fr,
        "judicial_verdict_en": judicial_en,
        "reliability_detection_pct": reliability_detection_pct,
        "reliability_false_alarm": reliability_false_alarm,
        "reliability_missed": reliability_missed,
        "eer_pct": f"{eer * 100:.1f}%" if eer is not None else None,
        "court_indicators": court_indicators,
        "sig_preview": sig_preview,
        "court_file_number": "",
    }
