"""Centralized application configuration using Pydantic Settings."""

import os
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    
    class Settings(BaseSettings):
        """Application settings loaded from environment variables."""
        
        # Application Settings
        APP_NAME: str = "Senti Market Intelligence"
        ENVIRONMENT: str = "development"
        LOG_LEVEL: str = Field(default="INFO")
        CACHE_TTL_SECONDS: int = Field(default=300)
        
        # Market Data Settings
        MARKET_DATA_PROVIDER: str = Field(default="yfinance")
        DEFAULT_SYMBOL: str = Field(default="AAPL")
        
        # Future Feature Placeholders (None by default, loaded securely via env)
        NEWS_API_KEY: Optional[str] = Field(default=None)
        LLM_PROVIDER: Optional[str] = Field(default=None)
        LLM_API_KEY: Optional[str] = Field(default=None)
        DATABASE_URL: Optional[str] = Field(default=None)
        VECTOR_DB_URL: Optional[str] = Field(default=None)
        MLFLOW_TRACKING_URI: Optional[str] = Field(default=None)
        
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False
        )

except ImportError:
    from pydantic import BaseModel, Field

    class Settings(BaseModel):  # type: ignore
        """Fallback Application settings when pydantic-settings is unavailable."""
        
        APP_NAME: str = "Senti Market Intelligence"
        ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
        LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
        
        MARKET_DATA_PROVIDER: str = os.getenv("MARKET_DATA_PROVIDER", "yfinance")
        DEFAULT_SYMBOL: str = os.getenv("DEFAULT_SYMBOL", "AAPL")
        
        NEWS_API_KEY: Optional[str] = os.getenv("NEWS_API_KEY", None)
        LLM_PROVIDER: Optional[str] = os.getenv("LLM_PROVIDER", None)
        LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY", None)
        DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", None)
        VECTOR_DB_URL: Optional[str] = os.getenv("VECTOR_DB_URL", None)
        MLFLOW_TRACKING_URI: Optional[str] = os.getenv("MLFLOW_TRACKING_URI", None)


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached instance of the application settings."""
    return Settings()
