"""Seed script — insère des données de test réalistes pour le dashboard analytics."""
from __future__ import annotations

import random
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")

from database import SessionLocal, init_db
from models.analysis import Analysis, AnalysisStatus, Verdict
from models.case import Case, CaseStatus, Jurisdiction
from models.media_file import MediaFile, MediaStatus, MediaType
from models.user import User, UserRole
from core.security import hash_password

random.seed(42)
init_db()
db = SessionLocal()
NOW = datetime.now(timezone.utc)


# ── Utilisateurs ──────────────────────────────────────────────────────────────
def _get_or_create_user(username: str, role: UserRole, password: str = "Admin1234!") -> User:
    u = db.query(User).filter(User.username == username).first()
    if not u:
        u = User(
            username=username,
            email=f"{username}@deepfake.ca",
            hashed_password=hash_password(password),
            role=role,
            is_active=True,
        )
        db.add(u)
        db.flush()
    return u


admin   = _get_or_create_user("admin",   UserRole.admin)
analyst1 = _get_or_create_user("jdupont",  UserRole.analyst)
analyst2 = _get_or_create_user("mkumar",   UserRole.analyst)
db.commit()


# ── Données dossiers ──────────────────────────────────────────────────────────
CASE_DATA = [
    ("2026-FED-0041", "Vidéo discours politique altérée",        Jurisdiction.federal,          "NDP c. Bloc Québécois",         "Pierre Vallée",    "Sophie Martin"),
    ("2026-ON-0112",  "Témoignage vidéo contrefait",             Jurisdiction.ontario,          "R c. Zhao Wei",                 "James O'Brien",    "Emily Torres"),
    ("2026-QC-0089",  "Enregistrement audio frauduleux",         Jurisdiction.quebec,           "Langlois c. Assurances XYZ",    "Henri Beaumont",   "Marie-Claire Roy"),
    ("2026-BC-0034",  "Deepfake CEO pour fraude financière",     Jurisdiction.british_columbia, "AMF c. TechCorp Inc.",          "Linda Park",       "Raj Mehta"),
    ("2026-AB-0067",  "Vidéo réclamation assurance falsifiée",   Jurisdiction.alberta,          "Industrial Trust c. Riopel",    "Frank Olsen",      "Nasrin Hosseini"),
    ("2026-FED-0055", "Propagande électorale synthétique",       Jurisdiction.federal,          "CCES c. Parti Vert",            "Anne Tremblay",    "Mathieu Gagnon"),
    ("2026-ON-0143",  "Faux témoin oculaire tribunal",           Jurisdiction.ontario,          "R c. Marcus Ibe",               "Daniel Kim",       "Yuki Nakamura"),
    ("2026-QC-0097",  "Audio contrat vocal manipulé",            Jurisdiction.quebec,           "Immeubles PQR c. Lacombe",      "Éric Bouchard",    "Isabelle Côté"),
    ("2026-FED-0078", "Simulation réunion cabinet ministre",     Jurisdiction.federal,          "GRC — Enquête DFI-2026",        "Cmdt Sarah Burns", "Me Paul Dubois"),
    ("2026-BC-0061",  "Vidéo consentement médical falsifié",     Jurisdiction.british_columbia, "Hôpital St-Paul c. famille",    "Dr Anita Sharma",  "Bryan Collins"),
    ("2026-ON-0167",  "Clip réseaux sociaux candidat",           Jurisdiction.ontario,          "Parti Libéral Ontario",         "Diane Moreau",     "Scott Harris"),
    ("2026-AB-0089",  "Appel audio frauduleux PDG",              Jurisdiction.alberta,          "Synergy Oil c. Investisseurs",  "Mike Thompson",    "Priya Patel"),
]

# (verdict, score, txt, tmp, rppg, bio, aud, pha, meta, days_ago)
SCENARIOS = [
    (Verdict.deepfake,      0.91, 0.93, 0.89, 0.95, 0.87, 0.88, 0.91, 0.82,  5),
    (Verdict.deepfake,      0.87, 0.85, 0.91, 0.88, 0.90, 0.83, 0.86, 0.79,  8),
    (Verdict.authentic,     0.12, 0.11, 0.09, 0.13, 0.15, 0.10, 0.08, 0.14, 12),
    (Verdict.deepfake,      0.95, 0.97, 0.93, 0.96, 0.92, 0.94, 0.95, 0.88,  3),
    (Verdict.authentic,     0.08, 0.07, 0.10, 0.06, 0.12, 0.09, 0.05, 0.11, 15),
    (Verdict.deepfake,      0.78, 0.81, 0.74, 0.82, 0.76, 0.75, 0.80, 0.71, 22),
    (Verdict.undetermined,  0.48, 0.52, 0.44, 0.51, 0.47, 0.45, 0.50, 0.43, 18),
    (Verdict.authentic,     0.15, 0.13, 0.17, 0.11, 0.19, 0.16, 0.12, 0.18, 31),
    (Verdict.deepfake,      0.83, 0.86, 0.80, 0.87, 0.84, 0.81, 0.85, 0.78, 45),
    (Verdict.deepfake,      0.72, 0.75, 0.69, 0.74, 0.70, 0.71, 0.73, 0.68, 52),
    (Verdict.authentic,     0.19, 0.18, 0.21, 0.15, 0.22, 0.20, 0.17, 0.23, 60),
    (Verdict.undetermined,  0.53, 0.55, 0.50, 0.54, 0.51, 0.52, 0.56, 0.48, 37),
]

cases_created = []
for i, (num, title, juris, descr, plain, def_) in enumerate(CASE_DATA):
    if db.query(Case).filter(Case.case_number == num).first():
        print(f"  skip existing case {num}")
        continue

    sc = SCENARIOS[i % len(SCENARIOS)]
    days_ago = sc[9]
    created = NOW - timedelta(days=days_ago + 2)
    owner = analyst1 if i % 2 == 0 else analyst2
    verdict = sc[0]
    status = CaseStatus.completed if verdict != Verdict.undetermined else CaseStatus.in_analysis

    case = Case(
        case_number=num,
        title=title,
        description=descr,
        jurisdiction=juris,
        status=status,
        plaintiff=plain,
        defendant=def_,
        created_by_id=owner.id,
        created_at=created,
        updated_at=created,
        closed_at=created + timedelta(days=1) if status == CaseStatus.completed else None,
    )
    db.add(case)
    db.flush()
    cases_created.append((case, sc, owner))

db.commit()


# ── Médias + Analyses ─────────────────────────────────────────────────────────
def _noise(v: float) -> float:
    return round(min(0.99, max(0.01, v + random.uniform(-0.03, 0.03))), 3)


MEDIA_NAMES = ["evidence_{}.mp4", "testimony_{}.mp4", "recording_{}.wav",
               "clip_{}.mp4", "audio_{}.wav", "interview_{}.mp4"]

analyses_created = 0
for case, sc, owner in cases_created:
    verdict, score, txt, tmp, rppg, bio, aud, pha, meta, days_ago = sc
    completed_at = NOW - timedelta(days=days_ago)
    started_at = completed_at - timedelta(minutes=random.randint(3, 12))
    fname = random.choice(MEDIA_NAMES).format(case.id)
    mtype = MediaType.audio if fname.endswith(".wav") else MediaType.video

    mf = MediaFile(
        uuid=str(uuid.uuid4()),
        case_id=case.id,
        original_filename=fname,
        media_type=mtype,
        mime_type="audio/wav" if mtype == MediaType.audio else "video/mp4",
        file_size_bytes=random.randint(1_500_000, 45_000_000),
        status=MediaStatus.verified,
        hash_sha256="a" * 64,
        hash_blake3="b" * 64,
        hash_md5="c" * 32,
        ingested_by_id=owner.id,
        storage_key=f"/data/uploads/{fname}",
    )
    db.add(mf)
    db.flush()

    a = Analysis(
        case_id=case.id,
        media_file_id=mf.id,
        status=AnalysisStatus.completed,
        verdict=verdict,
        final_score=score,
        confidence_low=round(score - 0.05, 2),
        confidence_high=round(score + 0.05, 2),
        score_video_texture=_noise(txt) if mtype == MediaType.video else None,
        score_video_temporal=_noise(tmp) if mtype == MediaType.video else None,
        score_rppg=_noise(rppg) if mtype == MediaType.video else None,
        score_biometrics=_noise(bio) if mtype == MediaType.video else None,
        score_audio_model=_noise(aud),
        score_audio_phase=_noise(pha),
        score_metadata=_noise(meta),
        duration_seconds=random.randint(25, 180),
        requested_by_id=owner.id,
        started_at=started_at,
        completed_at=completed_at,
        xai_plain_explanation=(
            f"Score de synthèse: {score:.2f}. "
            + ("Artefacts GAN détectés sur 3 scènes." if verdict == Verdict.deepfake
               else "Aucune anomalie détectée." if verdict == Verdict.authentic
               else "Résultats inconcluants, qualité insuffisante.")
        ),
    )
    db.add(a)
    analyses_created += 1

db.commit()


# ── Analyses supplémentaires pour les tendances (6 mois de données) ───────────
EXTRA_ANALYSES = [
    (Verdict.deepfake,     0.89,  75),
    (Verdict.authentic,    0.11,  70),
    (Verdict.deepfake,     0.77,  65),
    (Verdict.deepfake,     0.93,  58),
    (Verdict.authentic,    0.14,  55),
    (Verdict.undetermined, 0.51,  50),
    (Verdict.deepfake,     0.82,  42),
    (Verdict.authentic,    0.09,  38),
    (Verdict.deepfake,     0.96,  30),
    (Verdict.authentic,    0.17,  25),
    (Verdict.deepfake,     0.74,  92),
    (Verdict.authentic,    0.22,  88),
    (Verdict.deepfake,     0.88, 110),
    (Verdict.authentic,    0.13, 105),
    (Verdict.deepfake,     0.84, 125),
    (Verdict.authentic,    0.16, 118),
    (Verdict.undetermined, 0.49, 140),
    (Verdict.deepfake,     0.91, 155),
    (Verdict.authentic,    0.10, 170),
    (Verdict.deepfake,     0.79, 185),
]

# Attacher aux dossiers créés ci-dessus ou existants
anchor_case = db.query(Case).first()
anchor_mf   = db.query(MediaFile).filter(MediaFile.case_id == anchor_case.id).first() if anchor_case else None

if anchor_case and anchor_mf:
    for verdict, score, days_ago in EXTRA_ANALYSES:
        completed_at = NOW - timedelta(days=days_ago)
        a = Analysis(
            case_id=anchor_case.id,
            media_file_id=anchor_mf.id,
            status=AnalysisStatus.completed,
            verdict=verdict,
            final_score=score,
            score_video_texture=round(score + 0.02, 3) if score > 0.3 else round(score + 0.01, 3),
            score_audio_model=round(score - 0.01, 3),
            score_metadata=round(score + 0.01, 3),
            duration_seconds=random.randint(20, 90),
            requested_by_id=analyst1.id,
            started_at=completed_at - timedelta(minutes=5),
            completed_at=completed_at,
        )
        db.add(a)
        analyses_created += 1
    db.commit()

total_cases = db.query(Case).count()
total_analyses = db.query(Analysis).count()
print(f"Seed terminé: {len(cases_created)} nouveaux dossiers, {analyses_created} analyses créées.")
print(f"Total DB: {total_cases} dossiers, {total_analyses} analyses.")
db.close()
