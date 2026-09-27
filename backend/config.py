"""Configuration centralisée via variables d'environnement."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_name: str = "DeepfakeDetector"
    app_version: str = "3.1.0"
    app_env: str = "development"
    debug: bool = False
    allowed_origins: str = "http://localhost:3000"
    trusted_hosts: list[str] = ["localhost", "127.0.0.1"]

    # Database
    database_url: str = "postgresql://deepfake:changeme@postgres:5432/deepfake_db"

    # Redis / Celery
    celery_broker_url: str = "redis://:changeme@redis:6379/0"
    celery_result_backend: str = "redis://:changeme@redis:6379/1"
    # DB 2 for token blocklist (separate from Celery DBs 0 and 1)
    redis_url: str = "redis://127.0.0.1:6379/2"

    # Encryption at-rest (Fernet symmetric key)
    # Generate with: python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    fernet_key: str = ""

    # MinIO
    minio_endpoint: str = "minio:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "changeme123"
    minio_bucket_cases: str = "deepfake-cases"
    minio_bucket_reports: str = "deepfake-reports"
    minio_use_ssl: bool = False

    # JWT
    jwt_private_key_path: str = "/app/keys/private.pem"
    jwt_public_key_path: str = "/app/keys/public.pem"
    # Audit signing key (séparée des clés JWT)
    audit_signing_key_path: str = "/app/keys/audit_private.pem"
    jwt_algorithm: str = "RS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # MFA
    mfa_issuer: str = "DeepfakeDetector-Canada"
    mfa_required: bool = True

    # TSA
    tsa_url: str = "https://freetsa.org/tsr"
    tsa_cert_path: str = ""

    # Storage paths
    upload_dir: str = "/data/uploads"
    quarantine_dir: str = "/data/quarantine"
    processed_dir: str = "/data/processed"

    # File limits
    max_file_size_mb: int = 500
    allowed_video_extensions: str = ".mp4,.mov,.avi,.mkv,.webm"
    allowed_audio_extensions: str = ".wav,.mp3,.aac,.flac,.ogg,.m4a"
    allowed_document_extensions: str = ".pdf,.docx,.doc,.jpg,.jpeg,.png,.tiff,.tif"

    # Legal
    retention_years: int = 10

    # KYC (v3.1)
    kyc_face_match_threshold_insightface: float = 0.40
    kyc_face_match_threshold_deepface: float = 0.68
    kyc_document_deepfake_threshold_fail: float = 0.70
    kyc_document_deepfake_threshold_review: float = 0.50

    # HuggingFace (Model Registry — v2.0)
    hf_cache_dir: str = "/data/models"
    hf_token: str = ""

    # ML Training / Évaluation (Brief v3 §0)
    # allow_experimental_engines : si False, seuls les engines "validated" contribuent au score
    allow_experimental_engines: bool = True
    metrics_dir: str = "/data/metrics"

    @property
    def allowed_extensions(self) -> set[str]:
        video = set(self.allowed_video_extensions.split(","))
        audio = set(self.allowed_audio_extensions.split(","))
        return video | audio

    @property
    def allowed_document_exts(self) -> set[str]:
        return set(self.allowed_document_extensions.split(","))

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def jwt_private_key(self) -> str:
        path = Path(self.jwt_private_key_path)
        if path.exists():
            return path.read_text()
        return ""

    @property
    def jwt_public_key(self) -> str:
        path = Path(self.jwt_public_key_path)
        if path.exists():
            return path.read_text()
        return ""

    @property
    def fernet(self):
        """Retourne une instance Fernet si fernet_key est configurée, sinon None."""
        if not self.fernet_key:
            return None
        from cryptography.fernet import Fernet
        return Fernet(self.fernet_key.encode())


settings = Settings()
