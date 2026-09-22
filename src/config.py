"""Runtime configuration for CogniShield services."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    environment: str
    api_token: str | None
    database_path: Path
    feed_path: Path
    allow_csv_fallback: bool

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


def load_settings() -> Settings:
    environment = os.getenv("COGNISHIELD_ENV", "development").strip().lower()
    api_token = os.getenv("COGNISHIELD_API_TOKEN")
    database_path = Path(os.getenv("COGNISHIELD_DATABASE_PATH", str(PROJECT_ROOT / "data" / "cognishield.db")))
    feed_path = Path(os.getenv("COGNISHIELD_FEED_PATH", str(PROJECT_ROOT / "data" / "processed" / "threat_intelligence_feed.csv")))
    allow_csv_fallback = os.getenv("COGNISHIELD_ALLOW_CSV_FALLBACK", "false").strip().lower() == "true"

    if environment not in {"development", "test", "production"}:
        raise ValueError("COGNISHIELD_ENV must be development, test, or production")
    if environment == "production" and not api_token:
        raise ValueError("COGNISHIELD_API_TOKEN is required in production")

    return Settings(
        environment=environment,
        api_token=api_token,
        database_path=database_path,
        feed_path=feed_path,
        allow_csv_fallback=allow_csv_fallback,
    )


settings = load_settings()
