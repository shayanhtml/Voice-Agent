from __future__ import annotations

import base64
import hmac
import json
import time
from hashlib import sha256
from typing import Dict, Optional, Tuple

from .config import settings


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64u_decode(s: str) -> bytes:
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


def _secret() -> bytes:
    return settings.token_secret.encode("utf-8")


def generate_token(*, session_id: str, agent_id: Optional[str] = None, ttl_seconds: int = 1800) -> Tuple[str, int]:
    now = int(time.time())
    exp = now + int(ttl_seconds)
    payload: Dict[str, object] = {"sid": session_id, "iat": now, "exp": exp}
    if agent_id:
        payload["aid"] = agent_id

    header = {"alg": "HS256", "typ": "JWT"}
    head = _b64u(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    bod = _b64u(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{head}.{bod}".encode("utf-8")
    sig = hmac.new(_secret(), signing_input, sha256).digest()
    token = f"{head}.{bod}.{_b64u(sig)}"
    return token, exp


def verify_token(token: str) -> Dict[str, object]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("invalid token format")
        head_b64, bod_b64, sig_b64 = parts
        signing_input = f"{head_b64}.{bod_b64}".encode("utf-8")
        expected = hmac.new(_secret(), signing_input, sha256).digest()
        provided = _b64u_decode(sig_b64)
        if not hmac.compare_digest(expected, provided):
            raise ValueError("invalid signature")
        payload = json.loads(_b64u_decode(bod_b64).decode("utf-8"))
        now = int(time.time())
        if int(payload.get("exp", 0)) < now:
            raise ValueError("token expired")
        return payload
    except Exception as exc:
        raise ValueError(f"token verification failed: {exc}")
