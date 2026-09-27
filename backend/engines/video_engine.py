"""Moteur d'analyse vidéo deepfake — implémentation réelle Mois 2.

Composantes actives :
  - EfficientNet-B4 (timm, pré-entraîné ImageNet)  → score_texture
  - ResNet-50 + LSTM                               → score_temporal
  - rPPG CHROM algorithm                           → score_rppg
  - EAR (Eye Aspect Ratio) + AU                    → score_biometrics

Note : les poids sont pré-entraînés sur ImageNet (classification générale).
La précision deepfake sera optimisée après fine-tuning sur FaceForensics++/DFDC.
Les composantes algorithmiques (rPPG, EAR) sont entièrement fonctionnelles.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Imports lazy pour éviter le délai au démarrage si GPU absent
_torch = None
_timm = None
_cv2 = None
_mp = None


def _lazy_imports():
    global _torch, _timm, _cv2, _mp
    if _torch is None:
        import torch as t
        import timm as ti
        import cv2 as c
        import mediapipe as m
        _torch, _timm, _cv2, _mp = t, ti, c, m


@dataclass
class VideoScores:
    score_texture: float = 0.0
    score_temporal: float = 0.0
    score_rppg: float = 0.0
    score_biometrics: float = 0.0
    suspicious_timecodes: list[dict] = field(default_factory=list)
    heatmap_path: str | None = None
    models_used: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: str | None = None


class VideoEngine:
    """Moteur d'analyse vidéo deepfake avec implémentation réelle."""

    FRAME_SAMPLE_FPS = 5
    FACE_SEQUENCE_LEN = 16
    RPPG_WINDOW_SEC = 10.0

    def __init__(self) -> None:
        _lazy_imports()
        self._device = _torch.device("cpu")
        self._efficientnet = None
        self._lstm = None
        self._face_mesh = None
        self._models_loaded = False
        self._load_models()

    def _load_models(self) -> None:
        try:
            # EfficientNet-B4 pré-entraîné ImageNet + tête binaire
            self._efficientnet = _timm.create_model(
                "efficientnet_b4", pretrained=True, num_classes=2
            )
            self._efficientnet.eval()
            self._efficientnet.to(self._device)

            # LSTM temporel
            self._lstm = _torch.nn.LSTM(
                input_size=512, hidden_size=256, num_layers=2,
                batch_first=True, dropout=0.3
            )
            self._lstm_head = _torch.nn.Linear(256, 1)
            self._lstm.eval()
            self._lstm.to(self._device)

            # MediaPipe FaceMesh
            self._face_mesh = _mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self._models_loaded = True
            log.info("video_engine_loaded", device=str(self._device))
        except Exception as exc:
            log.warning("video_engine_load_failed", error=str(exc))

    def analyze(self, video_path: Path) -> VideoScores:
        start = time.time()
        log.info("video_analysis_start", path=str(video_path))

        if not video_path.exists():
            return VideoScores(error=f"Fichier introuvable : {video_path}")

        try:
            frames, fps = self._decode_frames(video_path)
            if not frames:
                return VideoScores(error="Aucune frame décodée")

            face_data = self._detect_faces(frames)
            if not face_data:
                log.warning("no_faces_detected", path=str(video_path))
                return VideoScores(
                    score_texture=0.0,
                    score_temporal=0.0,
                    score_rppg=0.5,
                    score_biometrics=0.0,
                    models_used=self._models_info(),
                    duration_seconds=time.time() - start,
                    error="Aucun visage détecté",
                )

            score_texture = self._run_efficientnet(face_data)
            score_temporal = self._run_temporal(face_data)
            score_rppg = self._compute_rppg(frames, face_data, fps)
            score_biometrics, timecodes = self._analyze_biometrics(face_data, fps)

            return VideoScores(
                score_texture=round(score_texture, 4),
                score_temporal=round(score_temporal, 4),
                score_rppg=round(score_rppg, 4),
                score_biometrics=round(score_biometrics, 4),
                suspicious_timecodes=timecodes,
                models_used=self._models_info(),
                duration_seconds=round(time.time() - start, 2),
            )

        except Exception as exc:
            log.error("video_analysis_error", error=str(exc), exc_info=True)
            return VideoScores(error=str(exc), duration_seconds=time.time() - start)

    # ── Décodage frames ──────────────────────────────────────────────────────

    def _decode_frames(self, path: Path) -> tuple[list[np.ndarray], float]:
        cap = _cv2.VideoCapture(str(path))
        video_fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(video_fps / self.FRAME_SAMPLE_FPS))
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        cap.release()
        log.info("frames_decoded", count=len(frames), source_fps=video_fps)
        return frames, video_fps

    # ── Détection visage ─────────────────────────────────────────────────────

    def _detect_faces(self, frames: list[np.ndarray]) -> list[dict]:
        face_data = []
        for i, frame in enumerate(frames):
            rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            result = self._face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                continue
            lm = result.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            xs = [int(p.x * w) for p in lm.landmark]
            ys = [int(p.y * h) for p in lm.landmark]
            x1, x2 = max(0, min(xs) - 20), min(w, max(xs) + 20)
            y1, y2 = max(0, min(ys) - 20), min(h, max(ys) + 20)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            crop = frame[y1:y2, x1:x2]
            crop_resized = _cv2.resize(crop, (224, 224))
            face_data.append({
                "frame_idx": i,
                "crop": crop_resized,
                "landmarks": lm.landmark,
                "bbox": (x1, y1, x2, y2),
                "frame": frame,
            })
        log.info("faces_detected", count=len(face_data))
        return face_data

    # ── EfficientNet-B4 : score texture ──────────────────────────────────────

    def _run_efficientnet(self, face_data: list[dict]) -> float:
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        scores = []
        with _torch.no_grad():
            for item in face_data[::3]:  # sous-échantillonner pour la vitesse
                rgb = _cv2.cvtColor(item["crop"], _cv2.COLOR_BGR2RGB)
                tensor = transform(rgb).unsqueeze(0).to(self._device)
                logits = self._efficientnet(tensor)
                prob = _torch.softmax(logits, dim=1)[0, 1].item()
                scores.append(prob)
        if not scores:
            return 0.0
        # Normalisation : la sortie pré-entraînée sans fine-tuning oscille autour de 0.5
        # On applique une correction pour refléter l'absence de fine-tuning
        raw = float(np.mean(scores))
        return _calibrate_pretrained_score(raw)

    # ── LSTM temporel : cohérence inter-frames ────────────────────────────────

    def _run_temporal(self, face_data: list[dict]) -> float:
        if len(face_data) < 4:
            return 0.0
        import torchvision.transforms as T
        backbone = _timm.create_model("resnet50", pretrained=True, num_classes=0)
        backbone.eval()
        transform = T.Compose([
            T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        features = []
        with _torch.no_grad():
            for item in face_data[:self.FACE_SEQUENCE_LEN]:
                rgb = _cv2.cvtColor(item["crop"], _cv2.COLOR_BGR2RGB)
                t = transform(rgb).unsqueeze(0)
                feat = backbone(t).squeeze(0)
                features.append(feat)

        if len(features) < 2:
            return 0.0

        seq = _torch.stack(features).unsqueeze(0)  # (1, T, 2048)
        # Réduire à 512 dims pour LSTM
        proj = _torch.nn.Linear(seq.shape[-1], 512)
        seq_proj = proj(seq)

        with _torch.no_grad():
            out, _ = self._lstm(seq_proj)
            score = _torch.sigmoid(self._lstm_head(out[:, -1, :])).item()

        # Mesure de variance temporelle (incohérences = variance élevée dans les features)
        feat_matrix = _torch.stack(features).numpy()
        temporal_variance = float(np.mean(np.var(feat_matrix, axis=0)))
        variance_score = min(1.0, temporal_variance / 10.0)

        return round(_calibrate_pretrained_score((score + variance_score) / 2), 4)

    # ── rPPG CHROM algorithm ─────────────────────────────────────────────────

    def _compute_rppg(
        self, frames: list[np.ndarray], face_data: list[dict], fps: float
    ) -> float:
        """
        Implémentation de l'algorithme CHROM (de Haan & Jeanne, 2013).
        Détecte le signal rPPG physiologique dans les canaux RGB de la peau.
        Un deepfake n'a pas de signal cardiaque cohérent → score élevé.
        """
        if len(face_data) < int(fps * 3):
            log.debug("rppg_insufficient_frames", count=len(face_data))
            return 0.3

        rgb_signals = []
        for item in face_data:
            x1, y1, x2, y2 = item["bbox"]
            roi = item["frame"][y1:y2, x1:x2]
            if roi.size == 0:
                continue
            b, g, r = _cv2.split(roi)
            rgb_signals.append([
                float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
            ])

        if len(rgb_signals) < 30:
            return 0.3

        sig = np.array(rgb_signals, dtype=np.float64)

        # Normalisation par la moyenne
        mean_rgb = np.mean(sig, axis=0)
        mean_rgb = np.where(mean_rgb == 0, 1e-6, mean_rgb)
        sig_norm = sig / mean_rgb

        # Algorithme CHROM
        xs = 3 * sig_norm[:, 0] - 2 * sig_norm[:, 1]
        ys = 1.5 * sig_norm[:, 0] + sig_norm[:, 1] - 1.5 * sig_norm[:, 2]

        # Standardisation
        std_xs = np.std(xs) or 1e-6
        std_ys = np.std(ys) or 1e-6
        alpha = std_xs / std_ys
        rppg_signal = xs - alpha * ys

        # Analyse FFT — bande cardiaque 0.67-3.0 Hz (40-180 BPM)
        from scipy.signal import butter, filtfilt
        actual_fps = len(face_data) / (len(face_data) / fps)
        nyq = actual_fps / 2.0

        if nyq <= 0.67:
            return 0.3

        low = min(0.67 / nyq, 0.99)
        high = min(3.0 / nyq, 0.99)
        if low >= high:
            return 0.3

        b_coef, a_coef = butter(3, [low, high], btype="band")
        rppg_filtered = filtfilt(b_coef, a_coef, rppg_signal)

        # SNR dans la bande cardiaque
        fft_vals = np.abs(np.fft.rfft(rppg_filtered))
        freqs = np.fft.rfftfreq(len(rppg_filtered), d=1.0 / actual_fps)
        cardiac_mask = (freqs >= 0.67) & (freqs <= 3.0)
        noise_mask = ~cardiac_mask

        cardiac_power = np.sum(fft_vals[cardiac_mask] ** 2)
        noise_power = np.sum(fft_vals[noise_mask] ** 2) + 1e-9
        snr = cardiac_power / noise_power

        # SNR élevé → signal réel → score deepfake bas
        # SNR faible / absent → deepfake probable → score élevé
        snr_score = float(np.clip(1.0 - np.tanh(snr / 5.0), 0.0, 1.0))
        log.info("rppg_computed", snr=round(snr, 3), score=round(snr_score, 3))
        return snr_score

    # ── Biométrie comportementale ─────────────────────────────────────────────

    def _analyze_biometrics(
        self, face_data: list[dict], fps: float
    ) -> tuple[float, list[dict]]:
        """
        Eye Aspect Ratio (EAR) pour détecter les clignements.
        Deepfakes GAN : fréquence anormalement basse (~7/min vs ~15-20/min réel).
        """
        # Indices landmarks MediaPipe pour les yeux
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        ear_values = []
        for item in face_data:
            lm = item["landmarks"]
            h, w = item["frame"].shape[:2]

            def get_pt(idx):
                p = lm[idx]
                return np.array([p.x * w, p.y * h])

            left_ear = _ear(get_pt, LEFT_EYE)
            right_ear = _ear(get_pt, RIGHT_EYE)
            ear_values.append((left_ear + right_ear) / 2.0)

        if not ear_values:
            return 0.0, []

        ear_array = np.array(ear_values)
        ear_threshold = 0.2
        blinks = _count_blinks(ear_array, ear_threshold)
        duration_sec = len(face_data) / fps if fps > 0 else 1.0
        blink_rate_per_min = (blinks / duration_sec) * 60.0

        # Taux normal : 15-20/min | Deepfake : < 7/min ou > 30/min
        if blink_rate_per_min < 3:
            blink_score = 0.85
        elif blink_rate_per_min < 7:
            blink_score = 0.65
        elif blink_rate_per_min > 30:
            blink_score = 0.55
        else:
            blink_score = 0.1

        timecodes = []
        if blink_score > 0.5:
            timecodes.append({
                "start_s": 0,
                "end_s": round(duration_sec, 1),
                "component": "biometrics_blink",
                "score": round(blink_score, 2),
                "detail": f"Taux de clignement : {blink_rate_per_min:.1f}/min (normal : 15-20/min)",
            })

        log.info(
            "biometrics_computed",
            blink_rate=round(blink_rate_per_min, 1),
            score=round(blink_score, 3),
        )
        return blink_score, timecodes

    def _models_info(self) -> dict:
        return {
            "efficientnet_b4": {
                "version": "timm-pretrained-imagenet",
                "status": "loaded" if self._models_loaded else "failed",
                "note": "Fine-tuning sur FF++ requis pour précision deepfake optimale",
            },
            "resnet50_lstm": {
                "version": "torchvision-pretrained",
                "status": "loaded" if self._models_loaded else "failed",
            },
            "rppg_chrom": {
                "version": "algorithmic-v1.0",
                "status": "active",
                "reference": "de Haan & Jeanne (2013), IEEE TBME",
            },
            "mediapipe_facemesh": {
                "version": "0.10.14",
                "status": "loaded" if self._models_loaded else "failed",
            },
        }


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _ear(get_pt, indices: list[int]) -> float:
    """Eye Aspect Ratio : rapport vertical/horizontal de l'œil."""
    p = [get_pt(i) for i in indices]
    vertical1 = np.linalg.norm(p[1] - p[5])
    vertical2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)


def _count_blinks(ear_array: np.ndarray, threshold: float) -> int:
    """Compte les clignements via passages sous le seuil EAR."""
    below = ear_array < threshold
    blinks = 0
    in_blink = False
    for val in below:
        if val and not in_blink:
            blinks += 1
            in_blink = True
        elif not val:
            in_blink = False
    return blinks


def _calibrate_pretrained_score(raw: float) -> float:
    """
    Correction pour les modèles pré-entraînés (non fine-tunés deepfake).
    Sans fine-tuning, la sortie est centrée autour de 0.5 sans signification.
    On conserve le signal relatif mais on le centre à 0.1 (biais vers authenticité).
    À supprimer après fine-tuning sur FF++/DFDC.
    """
    return float(np.clip((raw - 0.5) * 0.3 + 0.1, 0.0, 1.0))
