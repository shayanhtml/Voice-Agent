from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

_POLICY_PATH = Path(__file__).resolve().parent / "policy.yaml"


@lru_cache(maxsize=1)
def _load_policy() -> Dict[str, object]:
    if not _POLICY_PATH.exists():
        return {}
    try:
        with _POLICY_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _age_to_bucket(meta: Dict[str, object]) -> Optional[str]:
    # Accept explicit bucket "4-6"/"7-9", or infer from integer age
    bucket = str(meta.get("age_range") or meta.get("ageRange") or meta.get("age_bucket") or "").strip()
    if bucket:
        return bucket
    age = meta.get("age")
    try:
        age_int = int(age) if age is not None else None
    except (ValueError, TypeError):
        age_int = None
    if age_int is None:
        return None
    if 4 <= age_int <= 6:
        return "4-6"
    if 7 <= age_int <= 9:
        return "7-9"
    return None


def get_max_intensity(meta: Optional[Dict[str, object]] = None, default: int = 2) -> int:
    policy = _load_policy()
    table: Dict[str, int] = policy.get("max_intensity_by_age", {}) or {}
    if not isinstance(table, dict):
        return default
    if not meta:
        return default
    bucket = _age_to_bucket(meta) or ""
    try:
        return int(table.get(bucket, default))
    except (ValueError, TypeError):
        return default


_PROFANITY = {
    "fuck",
    "shit",
    "bitch",
    "asshole",
    "bastard",
}

_GORE = {"blood", "guts", "gory", "severed", "decapitate"}
_SEXUAL = {"sex", "nude", "naked", "porn"}
_MEDICAL = {"diagnose", "prescription", "dose", "surgery"}
_PII_PHRASES = {"your address", "your phone", "what is your name"}

_THREAT_PATTERNS = [
    re.compile(r"\bcome\s+get\b", re.I),
    re.compile(r"\bsnatch(\w+)?\b", re.I),
    re.compile(r"\bget\s+you\b", re.I),
    re.compile(r"\bcome\s+over\s+to\s+.*room\b", re.I),
]


def _remove_forbidden(text: str) -> Tuple[bool, Optional[str], str]:
    lowered = text.lower()
    violations = []
    # Simple word list checks
    for word in _PROFANITY:
        if word in lowered:
            lowered = lowered.replace(word, "****")
            violations.append("profanity")
    for word in _GORE:
        if word in lowered:
            lowered = lowered.replace(word, "[scary]")
            violations.append("gore")
    for word in _SEXUAL:
        if word in lowered:
            lowered = lowered.replace(word, "[redacted]")
            violations.append("sexual")
    for word in _MEDICAL:
        if word in lowered:
            lowered = lowered.replace(word, "[health]")
            violations.append("medical")
    for phrase in _PII_PHRASES:
        if phrase in lowered:
            lowered = lowered.replace(phrase, "personal info")
            violations.append("personal data collection")

    # De-threaten per policy "Never threaten."
    cleaned = lowered
    threatened = False
    for pat in _THREAT_PATTERNS:
        if pat.search(cleaned):
            threatened = True
            cleaned = pat.sub("check on", cleaned)
    if threatened:
        violations.append("threatening language")

    ok = len(violations) == 0
    reason = ", ".join(sorted(set(violations))) if violations else None
    # Try to restore original capitalization of first letter
    if text and cleaned:
        if text[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]
    return ok, reason, cleaned


def check_input(text: str, meta: dict):
    # Basic passthrough for now; could add parent-only command gating
    return True, None, text


def check_output(text: str, meta: dict):
    # Enforce forbidden content and soften threatening phrasing
    ok, reason, cleaned = _remove_forbidden(text)
    return ok, reason, cleaned
