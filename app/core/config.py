import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True

    # Celery & Redis Settings
    REDIS_URL: str
    CELERY_ENABLED: bool = True

    # Storage Settings
    STORAGE_TYPE: Literal["local", "s3"] = "local"
    LOCAL_STORAGE_DIR: str = "./data/storage"
    S3_BUCKET: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_ENDPOINT_URL: str = ""

    # Model Settings
    MODEL_PATHS_DIR: str = "./data/models"
    ALLOW_MODEL_DOWNLOADS: bool = False

    # OpenRouter Settings
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "openai/gpt-4o"

    # Database Settings
    DATABASE_URL: str

    # Cloudinary Settings
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str

    @property
    def local_storage_path(self) -> Path:
        path = Path(self.LOCAL_STORAGE_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_paths_path(self) -> Path:
        path = Path(self.MODEL_PATHS_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

settings = Settings()
