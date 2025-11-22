from __future__ import annotations

import base64
import os
import time
from typing import Optional

import httpx
from loguru import logger


class CartesiaTTS:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        voice_id: Optional[str] = None,
        timeout: float = 8.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        model: Optional[str] = None,
        endpoint: Optional[str] = "/tts/bytes",
    ) -> None:
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        configured_url = url or os.getenv("CARTESIA_TTS_URL")
        base_url = (configured_url or "").rstrip("/")
        endpoint = endpoint or "/tts/bytes"
        endpoint = "/" + endpoint.lstrip("/")
        if base_url and not base_url.lower().startswith("http"):
            base_url = f"https://{base_url}"
        if base_url and base_url.lower() == "https://api.cartesia.ai":
            self.url = f"{base_url}{endpoint}"
        elif configured_url:
            self.url = configured_url
        else:
            self.url = f"https://api.cartesia.ai{endpoint}"
        self.voice_id = voice_id or os.getenv("CARTESIA_VOICE_ID")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.retry_delay = max(0.0, retry_delay)
        self.model = model or os.getenv("CARTESIA_MODEL")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.url)

    def synthesize(self, text: str, *, voice_id: Optional[str] = None, fmt: str = "wav") -> Optional[bytes]:
        """
        Synthesize speech using Cartesia TTS.
        Uses the Cartesia API format with proper voice specification and output format.
        """
        if not self.is_configured():
            logger.warning("CartesiaTTS not configured; returning None")
            return None
        
        headers = {
            "X-API-Key": self.api_key,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json"
        }
        
        # Map format to output configuration
        output_format = {
            "container": fmt,
            "encoding": "pcm_s16le",
            "sample_rate": 44100
        }
        
        payload = {
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": voice_id or self.voice_id
            },
            "output_format": output_format,
        }
        if self.model:
            payload["model_id"] = self.model

        attempts = 0
        last_error: Optional[Exception] = None
        while attempts <= self.max_retries:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self.url, headers=headers, json=payload)
                    resp.raise_for_status()
                    return resp.content
            except Exception as exc:
                last_error = exc
                attempts += 1
                if attempts > self.max_retries:
                    logger.exception("Cartesia TTS request failed: %s", exc)
                    break
                logger.warning("Cartesia TTS attempt %d/%d failed: %s", attempts, self.max_retries, exc)
                if self.retry_delay:
                    time.sleep(self.retry_delay)

        if last_error:
            logger.debug("Cartesia TTS exhausted retries after error: %s", last_error)
        return None

    @staticmethod
    def to_b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")
