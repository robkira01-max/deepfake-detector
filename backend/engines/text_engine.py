"""TextEngine — Détection de texte généré par IA (v2.0).

Trois approches complémentaires :
1. Modèle RoBERTa fine-tuné (roberta-base-openai-detector)
2. Perplexité via GPT-2 (faible perplexité = texte IA)
3. Analyse statistique burstiness + entropie

Lazy imports : les modèles sont chargés au premier appel.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# ── Lazy globals ──────────────────────────────────────────────────────────────
_transformers = None
_torch = None

_DETECTOR_MODEL_ID = "roberta-base-openai-detector"
_PERPLEXITY_MODEL_ID = "gpt2"

# Seuils de verdict
_DEEPFAKE_THRESHOLD = 0.70
_AUTHENTIC_THRESHOLD = 0.35
_MIN_WORDS = 50  # en dessous = analyse non fiable


@dataclass
class TextScores:
    score_model: float        # 0=humain, 1=IA — RoBERTa
    score_perplexity: float   # 0=humain, 1=IA — PPL GPT-2 normalisé
    score_burstiness: float   # 0=humain, 1=IA — variance phrases
    score_entropy: float      # 0=humain, 1=IA — entropie tokens
    final_score: float
    word_count: int
    flagged_passages: list[dict] = field(default_factory=list)
    models_used: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def verdict_label(self) -> str:
        if self.final_score >= _DEEPFAKE_THRESHOLD:
            return "TEXTE IA DÉTECTÉ"
        if self.final_score <= _AUTHENTIC_THRESHOLD:
            return "RÉDACTION HUMAINE"
        return "INDÉTERMINÉ"


def _load_transformers():
    global _transformers, _torch
    if _transformers is None:
        import transformers as _t
        import torch as _to
        _transformers = _t
        _torch = _to


# ── Analyse statistique (sans modèle ML) ─────────────────────────────────────

def _tokenize_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s.split()) >= 3]


def _burstiness_score(sentences: list[str]) -> float:
    """Les humains ont des phrases très variées en longueur ; les LLM sont réguliers.

    Burstiness = coefficient de variation inversé.
    Valeur élevée = régulier = IA.
    """
    if len(sentences) < 5:
        return 0.5
    lengths = [len(s.split()) for s in sentences]
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.5
    stdev = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    cv = stdev / mean  # coefficient de variation
    # cv élevé = variable = humain → score bas
    # cv bas = régulier = IA → score élevé
    score = max(0.0, min(1.0, 1.0 - cv))
    return round(score, 4)


def _entropy_score(text: str) -> float:
    """Entropie de Shannon sur les mots (log2).

    Texte humain : entropie élevée (mots variés).
    Texte IA : entropie légèrement plus faible (vocabulaire limité mais précis).
    Faible discriminant seul — combiné avec les autres.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.5
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    total = len(words)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    # Normaliser sur une échelle typique (0–12 bits)
    normalized = min(entropy / 12.0, 1.0)
    # Entropie élevée = humain → score bas pour "IA"
    score = max(0.0, min(1.0, 1.0 - normalized + 0.3))
    return round(score, 4)


# ── Perplexité GPT-2 ──────────────────────────────────────────────────────────

def _compute_perplexity(text: str, max_length: int = 512) -> float | None:
    """Perplexité via GPT-2. Faible PPL = texte IA probable."""
    try:
        _load_transformers()
        tokenizer = _transformers.AutoTokenizer.from_pretrained(_PERPLEXITY_MODEL_ID)
        model = _transformers.AutoModelForCausalLM.from_pretrained(_PERPLEXITY_MODEL_ID)
        model.eval()

        inputs = tokenizer(
            text[:2000],  # limiter pour la mémoire
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        with _torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        ppl = math.exp(loss)
        return ppl
    except Exception:
        return None


def _normalize_perplexity(ppl: float) -> float:
    """Normalise la perplexité brute en score 0–1 (1 = très probablement IA).

    GPT-2 sur texte humain typique : PPL ~30–100
    GPT-2 sur texte LLM : PPL ~10–25 (LLM est plus "prévisible" que les humains)
    """
    if ppl <= 15:
        return 0.85
    if ppl <= 25:
        return 0.70
    if ppl <= 40:
        return 0.50
    if ppl <= 70:
        return 0.30
    return 0.15


# ── Classifieur RoBERTa ───────────────────────────────────────────────────────

def _run_roberta_detector(text: str) -> tuple[float, list[dict]]:
    """Classifie le texte avec roberta-base-openai-detector.

    Retourne (score_ia, passages_signalés).
    """
    try:
        _load_transformers()
        classifier = _transformers.pipeline(
            "text-classification",
            model=_DETECTOR_MODEL_ID,
            truncation=True,
            max_length=512,
        )

        # Analyse globale
        result = classifier(text[:1000])
        label = result[0]["label"]
        confidence = result[0]["score"]

        # Label "Fake" = IA, "Real" = humain
        score_ia = confidence if "fake" in label.lower() else 1.0 - confidence

        # Analyse par passage (paragraphes)
        flagged: list[dict] = []
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.split()) >= 20]
        for i, para in enumerate(paragraphs[:10]):  # max 10 paragraphes
            r = classifier(para[:512])
            lbl = r[0]["label"]
            conf = r[0]["score"]
            para_score = conf if "fake" in lbl.lower() else 1.0 - conf
            if para_score >= 0.65:
                flagged.append({
                    "paragraph_index": i,
                    "text_preview": para[:100] + "..." if len(para) > 100 else para,
                    "score_ia": round(para_score, 3),
                })

        return round(score_ia, 4), flagged

    except Exception as exc:
        return 0.5, [{"error": str(exc)}]


# ── Interface principale ───────────────────────────────────────────────────────

class TextEngine:
    """Moteur de détection de texte généré par IA."""

    def analyze(self, text: str) -> TextScores:
        """Analyse un texte et retourne les scores de détection IA."""
        words = text.split()
        word_count = len(words)

        if word_count < _MIN_WORDS:
            return TextScores(
                score_model=0.5,
                score_perplexity=0.5,
                score_burstiness=0.5,
                score_entropy=0.5,
                final_score=0.5,
                word_count=word_count,
                error=f"Texte trop court ({word_count} mots, minimum {_MIN_WORDS})",
                models_used={"status": "insufficient_text"},
            )

        sentences = _tokenize_sentences(text)

        # Analyse statistique (CPU rapide, sans modèle)
        score_burstiness = _burstiness_score(sentences)
        score_entropy = _entropy_score(text)

        # Perplexité GPT-2
        ppl = _compute_perplexity(text)
        score_perplexity = _normalize_perplexity(ppl) if ppl else 0.5

        # RoBERTa classifieur
        score_model, flagged_passages = _run_roberta_detector(text)

        # Fusion pondérée
        # RoBERTa est le signal le plus fort
        final_score = round(
            0.50 * score_model
            + 0.25 * score_perplexity
            + 0.15 * score_burstiness
            + 0.10 * score_entropy,
            4,
        )

        return TextScores(
            score_model=score_model,
            score_perplexity=score_perplexity,
            score_burstiness=score_burstiness,
            score_entropy=score_entropy,
            final_score=final_score,
            word_count=word_count,
            flagged_passages=flagged_passages,
            models_used={
                "roberta": _DETECTOR_MODEL_ID,
                "perplexity_model": _PERPLEXITY_MODEL_ID,
                "gpt2_perplexity": round(ppl, 2) if ppl else None,
                "sentences_analyzed": len(sentences),
            },
        )

    def analyze_file(self, path: Path) -> TextScores:
        """Extrait le texte d'un fichier .txt ou .docx et l'analyse."""
        try:
            text = self._extract_text(path)
            return self.analyze(text)
        except Exception as exc:
            return TextScores(
                score_model=0.0, score_perplexity=0.0,
                score_burstiness=0.0, score_entropy=0.0,
                final_score=0.0, word_count=0,
                error=str(exc), models_used={},
            )

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix in (".docx", ".doc"):
            import docx
            doc = docx.Document(str(path))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if suffix == ".pdf":
            import pymupdf
            doc = pymupdf.open(str(path))
            return "\n\n".join(page.get_text() for page in doc)
        raise ValueError(f"Format non supporté pour extraction texte : {suffix}")
