from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = Field(default="development", alias="ENVIRONMENT")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=1, alias="WORKERS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    cartesia_api_key: Optional[str] = Field(default=None, alias="CARTESIA_API_KEY")
    cartesia_voice_id: Optional[str] = Field(default=None, alias="CARTESIA_VOICE_ID")
    cartesia_agent_id: Optional[str] = Field(default=None, alias="CARTESIA_AGENT_ID")
    cartesia_model: Optional[str] = Field(default=None, alias="CARTESIA_MODEL")
    cartesia_tts_url: Optional[str] = Field(default=None, alias="CARTESIA_TTS_URL")
    cartesia_ws_url: Optional[str] = Field(default=None, alias="CARTESIA_WS_URL")
    cartesia_token_url: Optional[str] = Field(default=None, alias="CARTESIA_TOKEN_URL")

    token_secret: str = Field(default="change-this-secret", alias="TOKEN_SECRET")

    llm_timeout_seconds: float = Field(default=5.0, alias="LLM_TIMEOUT_SECONDS")
    idle_timeout_seconds: int = Field(default=25, alias="IDLE_TIMEOUT_SECONDS")
    ws_idle_timeout_seconds: float = Field(default=25.0, alias="WS_IDLE_TIMEOUT_SECONDS")

    max_concurrent_calls: int = Field(default=25, alias="MAX_CONCURRENT_CALLS")
    connection_timeout: float = Field(default=30.0, alias="CONNECTION_TIMEOUT")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_delay: float = Field(default=1.0, alias="RETRY_DELAY")

    max_conversation_turns: int = Field(default=12, alias="MAX_CONVERSATION_TURNS")
    enable_guardrails: bool = Field(default=True, alias="ENABLE_GUARDRAILS")
    voice_model: str = Field(default="sonic-3", alias="VOICE_MODEL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
