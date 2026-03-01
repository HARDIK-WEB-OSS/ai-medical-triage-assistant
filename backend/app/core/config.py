from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    # App
    app_name: str = "AI Medical Triage Assistant"
    environment: str = "development"
    debug: bool = True

    # Database
    database_url: str = "postgresql://triage_user:triage_pass@localhost:5432/triage_db"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Security
    secret_key: str = "dev-secret-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # AI/ML
    model_path: str = "ml/models/triage_classifier.joblib"
    chroma_persist_dir: str = "data/chroma_db"
    openai_api_key: str = ""
    confidence_threshold: float = 0.70  # Below this → flag for human review

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
