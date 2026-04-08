"""Application configuration loaded from environment variables."""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    # Fallback model used automatically when the primary model is throttled.
    groq_fallback_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_FALLBACK_MODEL")

    model_config = {"populate_by_name": True, "env_file": ".env"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
