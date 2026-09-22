"""DeepfakeDetector — Script CLI de démonstration.

Usage:
    python demo.py <fichier> [--output rapport.pdf] [--verbose] [--no-color]

Analyse un fichier audio ou vidéo sans Celery ni Redis.
Génère un rapport terminal et optionnellement un PDF WeasyPrint.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ajouter le backend au path
_BACKEND = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(_BACKEND))

# Extensions supportées
_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
_AUDIO_EXT = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a", ".opus"}
_ALL_EXT = _VIDEO_EXT | _AUDIO_EXT

# ── Couleurs ANSI ─────────────────────────────────────────────────────────────

class _Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    DIM     = "\033[2m"

_NO_COLOR = False


def _c(text: str, *codes: str) -> str:
    if _NO_COLOR:
        return text
    return "".join(codes) + text + _Colors.RESET


def _bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    return "[" + "■" * filled + "░" * (width - filled) + "]"


def _verdict_color(verdict: str) -> str:
    if verdict == "DEEPFAKE":
        return _c(f"  ✗  {verdict}", _Colors.BOLD, _Colors.RED)
    if verdict == "AUTHENTIC":
        return _c(f"  ✓  {verdict}", _Colors.BOLD, _Colors.GREEN)
    return _c(f"  ⚠  {verdict}", _Colors.BOLD, _Colors.YELLOW)


# ── Analyse principale ────────────────────────────────────────────────────────

def analyse(file_path: Path, verbose: bool = False) -> dict[str, Any]:
    """Lance l'analyse complète sur le fichier et retourne les résultats."""
    from core.chain_of_custody import compute_hashes
    from engines.metadata_engine import MetadataEngine
    from engines.fusion import fuse_scores

    ext = file_path.suffix.lower()
    is_video = ext in _VIDEO_EXT

    results: dict[str, Any] = {
        "file": file_path.name,
        "size_bytes": file_path.stat().st_size,
        "ext": ext,
        "is_video": is_video,
        "scores": {},
        "findings": [],
        "tool_detected": None,
        "errors": [],
    }

    # 1. Hashes chaîne de possession
    if verbose:
        print(_c("  → Calcul des hashes...", _Colors.DIM))
    try:
        bundle = compute_hashes(file_path)
        results["sha256"] = bundle.sha256
        results["blake3"] = bundle.blake3
        results["md5"] = bundle.md5
    except Exception as exc:
        results["sha256"] = "error"
        results["errors"].append(f"hash: {exc}")

    # 2. Metadata engine (toujours)
    if verbose:
        print(_c("  → Analyse métadonnées...", _Colors.DIM))
    try:
        meta = MetadataEngine().analyze(file_path)
        results["scores"]["metadata"] = meta.score
        results["findings"].extend(meta.findings)
        results["tool_detected"] = meta.tool_detected
    except Exception as exc:
        results["scores"]["metadata"] = 0.0
        results["errors"].append(f"metadata: {exc}")

    # 3. Audio engine
    if verbose:
        print(_c("  → Analyse audio...", _Colors.DIM))
    try:
        from engines.audio_engine import AudioEngine
        audio = AudioEngine().analyze(file_path)
        results["scores"]["audio_model"] = audio.score_model
        results["scores"]["audio_phase"] = audio.score_phase
    except Exception as exc:
        results["scores"]["audio_model"] = 0.0
        results["scores"]["audio_phase"] = 0.0
        results["errors"].append(f"audio: {exc}")

    # 4. Video engine (si vidéo)
    if is_video:
        if verbose:
            print(_c("  → Analyse vidéo...", _Colors.DIM))
        try:
            from engines.video_engine import VideoEngine
            video = VideoEngine().analyze(file_path)
            results["scores"]["video_texture"]  = video.score_texture
            results["scores"]["video_temporal"] = video.score_temporal
            results["scores"]["rppg"]           = video.score_rppg
            results["scores"]["biometrics"]     = video.score_biometrics
        except Exception as exc:
            results["scores"]["video_texture"]  = 0.0
            results["scores"]["video_temporal"] = 0.0
            results["scores"]["rppg"]           = 0.0
            results["scores"]["biometrics"]     = 0.0
            results["errors"].append(f"video: {exc}")
    else:
        results["scores"]["video_texture"]  = 0.0
        results["scores"]["video_temporal"] = 0.0
        results["scores"]["rppg"]           = 0.0
        results["scores"]["biometrics"]     = 0.0

    # 5. Fusion des scores
    fused = fuse_scores(
        score_texture    = results["scores"].get("video_texture", 0.0),
        score_temporal   = results["scores"].get("video_temporal", 0.0),
        score_rppg       = results["scores"].get("rppg", 0.0),
        score_biometrics = results["scores"].get("biometrics", 0.0),
        score_audio      = results["scores"].get("audio_model", 0.0),
        score_phase      = results["scores"].get("audio_phase", 0.0),
        score_metadata   = results["scores"].get("metadata", 0.0),
    )
    results["final_score"]    = fused.final_score
    results["verdict"]        = fused.verdict.value.upper()
    results["confidence_low"] = fused.confidence_low
    results["confidence_high"] = fused.confidence_high
    results["shap"]           = fused.shap_ranking or []
    results["explanation"]    = fused.plain_explanation or ""

    return results


# ── Affichage terminal ────────────────────────────────────────────────────────

def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"


def print_report(results: dict[str, Any], duration: float) -> None:
    w = 56
    sep = "═" * w

    print(_c(f"\n╔{'═' * w}╗", _Colors.CYAN))
    title = "DeepfakeDetector v1.0 — Analyse Forensique"
    pad = (w - len(title)) // 2
    print(_c(f"║{' ' * pad}{title}{' ' * (w - pad - len(title))}║", _Colors.CYAN, _Colors.BOLD))
    print(_c(f"╚{'═' * w}╝\n", _Colors.CYAN))

    sha = results.get("sha256", "N/A")
    sha_short = sha[:16] + "..." if len(sha) > 16 else sha
    print(f"  {_c('Fichier', _Colors.BOLD)}    : {results['file']}")
    print(f"  {_c('Taille', _Colors.BOLD)}     : {_fmt_size(results['size_bytes'])}")
    print(f"  {_c('Type', _Colors.BOLD)}       : {'video' if results['is_video'] else 'audio'}{results['ext']}")
    print(f"  {_c('SHA256', _Colors.BOLD)}     : {sha_short}")
    if results.get("tool_detected"):
        print(f"  {_c('Outil détecté', _Colors.BOLD)}: {_c(results['tool_detected'], _Colors.RED, _Colors.BOLD)}")

    score = results["final_score"]
    verdict = results["verdict"]

    print(f"\n  {_c(sep, _Colors.DIM)}")
    print(f"\n{_c('  VERDICT', _Colors.BOLD)}")
    print(f"\n  {_verdict_color(verdict)}  {_c(f'(score: {score:.2f})', _Colors.DIM)}")
    ci_low  = results.get("confidence_low", 0.0)
    ci_high = results.get("confidence_high", 1.0)
    print(f"  {_c('IC 95%', _Colors.DIM)}: [{ci_low:.2f} – {ci_high:.2f}]")

    print(f"\n  {_c(sep, _Colors.DIM)}\n")
    print(f"  {_c('Scores par composant :', _Colors.BOLD)}\n")

    labels = {
        "video_texture":  "Texture vidéo ",
        "video_temporal": "Temporel vidéo",
        "rppg":           "rPPG          ",
        "biometrics":     "Biométrie     ",
        "audio_model":    "Audio modèle  ",
        "audio_phase":    "Phase audio   ",
        "metadata":       "Métadonnées   ",
    }
    for key, label in labels.items():
        s = results["scores"].get(key, 0.0)
        bar = _bar(s)
        color = _Colors.RED if s > 0.7 else (_Colors.YELLOW if s > 0.4 else _Colors.GREEN)
        print(f"    {label} : {_c(bar, color)} {s:.2f}")

    if results.get("explanation"):
        print(f"\n  {_c('Explication :', _Colors.BOLD)}")
        for line in results["explanation"].split(". "):
            if line.strip():
                print(f"    • {line.strip()}")

    if results.get("findings"):
        print(f"\n  {_c('Findings métadonnées :', _Colors.BOLD)}")
        for f in results["findings"][:5]:
            print(f"    {_c('→', _Colors.YELLOW)} {f}")

    if results.get("errors"):
        print(f"\n  {_c('Avertissements :', _Colors.DIM)}")
        for e in results["errors"]:
            print(f"    {_c('⚠', _Colors.YELLOW)} {e}")

    print(f"\n  {_c(sep, _Colors.DIM)}")
    print(f"  Durée : {_c(f'{duration:.1f} secondes', _Colors.BOLD)}\n")


# ── Génération PDF ────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Rapport DeepfakeDetector</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
  h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; }}
  h2 {{ color: #283593; margin-top: 24px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th {{ background: #e8eaf6; text-align: left; padding: 8px; }}
  td {{ padding: 8px; border-bottom: 1px solid #e0e0e0; }}
  .deepfake {{ color: #c62828; font-weight: bold; }}
  .authentic {{ color: #2e7d32; font-weight: bold; }}
  .undetermined {{ color: #e65100; font-weight: bold; }}
  .finding {{ background: #fff3e0; border-left: 3px solid #fb8c00; padding: 6px 12px; margin: 4px 0; }}
</style>
</head>
<body>
<h1>Rapport d'analyse forensique — DeepfakeDetector v1.0</h1>
<p><strong>Fichier analysé :</strong> {file}</p>
<p><strong>Taille :</strong> {size}</p>
<p><strong>SHA256 :</strong> <code>{sha256}</code></p>
<p><strong>Blake3 :</strong> <code>{blake3}</code></p>

<h2>Verdict</h2>
<p class="{verdict_class}">&#9632; {verdict} (score global : {score:.3f})</p>
<p>Intervalle de confiance 95% : [{ci_low:.2f} – {ci_high:.2f}]</p>

<h2>Scores par composant</h2>
<table>
  <tr><th>Composant</th><th>Score</th><th>Indicateur</th></tr>
  {score_rows}
</table>

{findings_section}

<h2>Informations chaîne de possession</h2>
<table>
  <tr><th>Hash</th><th>Valeur</th></tr>
  <tr><td>SHA256</td><td><code>{sha256}</code></td></tr>
  <tr><td>Blake3</td><td><code>{blake3}</code></td></tr>
  <tr><td>MD5</td><td><code>{md5}</code></td></tr>
</table>

<p><em>Rapport généré le {date} par DeepfakeDetector v1.0 — à des fins de démonstration uniquement.</em></p>
</body>
</html>
"""


def generate_pdf(results: dict[str, Any], output_path: Path) -> None:
    """Génère un rapport PDF via WeasyPrint."""
    from datetime import datetime, timezone

    labels = {
        "video_texture":  "Texture vidéo",
        "video_temporal": "Temporel vidéo",
        "rppg":           "rPPG",
        "biometrics":     "Biométrie",
        "audio_model":    "Audio modèle",
        "audio_phase":    "Phase audio",
        "metadata":       "Métadonnées",
    }
    score_rows = "\n".join(
        f"  <tr><td>{label}</td><td>{results['scores'].get(k, 0.0):.3f}</td>"
        f"<td>{'⚠ Suspect' if results['scores'].get(k, 0.0) > 0.7 else 'Normal'}</td></tr>"
        for k, label in labels.items()
    )

    findings_html = ""
    if results.get("findings"):
        items = "\n".join(f'  <div class="finding">{f}</div>' for f in results["findings"])
        findings_html = f"<h2>Findings métadonnées</h2>\n{items}"

    verdict = results["verdict"]
    verdict_class = verdict.lower()

    html = _HTML_TEMPLATE.format(
        file    = results["file"],
        size    = _fmt_size(results["size_bytes"]),
        sha256  = results.get("sha256", "N/A"),
        blake3  = results.get("blake3", "N/A"),
        md5     = results.get("md5", "N/A"),
        verdict = verdict,
        verdict_class = verdict_class,
        score   = results["final_score"],
        ci_low  = results.get("confidence_low", 0.0),
        ci_high = results.get("confidence_high", 1.0),
        score_rows = score_rows,
        findings_section = findings_html,
        date    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )

    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(output_path))
        print(_c(f"  PDF généré : {output_path}", _Colors.GREEN))
    except ImportError:
        # Fallback HTML si WeasyPrint absent
        html_path = output_path.with_suffix(".html")
        html_path.write_text(html, encoding="utf-8")
        print(_c(f"  ⚠ WeasyPrint absent — HTML généré : {html_path}", _Colors.YELLOW))
    except Exception as exc:
        print(_c(f"  ⚠ Erreur PDF : {exc}", _Colors.YELLOW))


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main() -> int:
    global _NO_COLOR

    parser = argparse.ArgumentParser(
        description="DeepfakeDetector v1.0 — Analyse forensique CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exemple : python demo.py video.mp4 --output rapport.pdf --verbose",
    )
    parser.add_argument("file", help="Fichier audio ou vidéo à analyser")
    parser.add_argument("--output", "-o", help="Chemin du rapport PDF (optionnel)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Afficher les étapes détaillées")
    parser.add_argument("--no-color", action="store_true", help="Désactiver les couleurs ANSI")
    args = parser.parse_args()

    if args.no_color:
        _NO_COLOR = True

    file_path = Path(args.file)

    # Validation
    if not file_path.exists():
        print(_c(f"Erreur : fichier introuvable — {file_path}", _Colors.RED), file=sys.stderr)
        return 1

    if not file_path.is_file():
        print(_c(f"Erreur : le chemin n'est pas un fichier — {file_path}", _Colors.RED), file=sys.stderr)
        return 1

    ext = file_path.suffix.lower()
    if ext not in _ALL_EXT:
        print(
            _c(f"Erreur : extension non supportée — {ext}\n"
               f"Formats acceptés : {', '.join(sorted(_ALL_EXT))}", _Colors.RED),
            file=sys.stderr,
        )
        return 1

    print(_c(f"\nAnalyse de {file_path.name}...\n", _Colors.CYAN))

    try:
        t0 = time.monotonic()
        results = analyse(file_path, verbose=args.verbose)
        duration = time.monotonic() - t0
    except KeyboardInterrupt:
        print(_c("\nAnalyse interrompue.", _Colors.YELLOW))
        return 130
    except Exception as exc:
        print(_c(f"Erreur fatale : {exc}", _Colors.RED), file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2

    print_report(results, duration)

    if args.output:
        output_path = Path(args.output)
        generate_pdf(results, output_path)

    # Code de sortie basé sur le verdict
    return 0 if results["verdict"] in ("AUTHENTIC", "UNDETERMINED") else 1


if __name__ == "__main__":
    sys.exit(main())
