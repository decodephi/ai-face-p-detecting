import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    api_title: str = "Facial Skin Analysis API"
    api_version: str = "1.0.0"
    target_size: int = 512
    max_detections: int = 25
    allowed_origins: list[str] = Field(
        default_factory=lambda: _parse_allowed_origins(
            os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173")
        )
    )


def _parse_allowed_origins(raw_origins: str) -> list[str]:
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


settings = Settings()
FIXED_IMAGE_SIZE = (settings.target_size, settings.target_size)
