from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

from loguru import logger
from ..config import settings
from ..guardrails import enforcer

try:
    from openai import APIError, OpenAI
except ImportError:  # pragma: no cover - OpenAI client optional for local runs
    OpenAI = None  # type: ignore[assignment]
    APIError = Exception  # type: ignore[assignment]

Phase = Literal[
    "introduction",
    "situation",
    "consequence",
    "escalate",
    "de_escalate",
    "result_positive",
    "result_negative",
    "goodbye",
]

Classification = Literal[
    "ongoing",
    "escalation",
    "praise",
    "resolved_positive",
    "resolved_negative",
]


DEFAULT_CLASSIFICATION_FOR_PHASE: Dict[Phase, Classification] = {
    "introduction": "ongoing",
    "situation": "ongoing",
    "consequence": "ongoing",
    "escalate": "escalation",
    "de_escalate": "praise",
    "result_positive": "resolved_positive",
    "result_negative": "resolved_negative",
    "goodbye": "resolved_positive",
}


@dataclass
class MonsterTurn:
    """Structured output for a monster turn."""

    sentences: List[str]
    control: Dict[str, object]
    phase: Phase
    classification: Classification
    full_text: str
    source_segment_label: Optional[str] = None


@dataclass
class ScriptSegment:
    """Represents a single scripted utterance we can adapt per turn."""

    phase: Phase
    text: str
    label: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ScriptSegment":
        try:
            phase = data["phase"].strip()
            text = data["text"].strip()
        except KeyError as exc:
            raise ValueError(f"Script segment missing key: {exc}") from exc

        label = data.get("label")
        if label:
            label = label.strip()

        if phase not in DEFAULT_CLASSIFICATION_FOR_PHASE:
            raise ValueError(f"Unsupported phase '{phase}' for script segment.")

        return cls(phase=phase, text=text, label=label)


@dataclass
class TurnRecord:
    """Conversation memory for the controller."""

    speaker: Literal["monster", "participant", "system"]
    text: str
    phase: Optional[Phase] = None
    classification: Optional[Classification] = None


class MonsterConversationController:
    """Adapts scripted content using GPT-5 Nano while tracking phases and memory."""

    def __init__(
        self,
        monster_name: str,
        voice_id: str,
        *,
        model: str = "gpt-5-nano",
        temperature: float = 1.0,
        max_history_turns: int = 12,
        prompt_dir: Optional[Path] = None,
        script_plan: Optional[Iterable[ScriptSegment]] = None,
        client: Optional[object] = None,
    ) -> None:
        self.monster_name = monster_name
        self.voice_id = voice_id
        self.model = model
        self.temperature = temperature
        self.max_history_turns = max_history_turns
        self.script_plan: List[ScriptSegment] = list(script_plan or [])
        self.current_index: int = 0
        self.history: List[TurnRecord] = []
        # Timeout for LLM responses; fallback to scripted after this
        self.request_timeout_seconds = settings.llm_timeout or settings.llm_timeout_seconds

        self.prompt_dir = prompt_dir or Path(__file__).resolve().parent.parent / "prompts"
        self.system_prompt = self._build_system_prompt()

        self.client = client
        if self.client is None and OpenAI is not None:
            api_key = settings.openai_api_key
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                logger.warning(
                    "OPENAI_API_KEY not found; controller will fall back to scripted responses."
                )

        logger.info(
            "MonsterConversationController initialized for %s with model %s",
            self.monster_name,
            self.model,
        )
        self._pending_phase_override: Optional[Phase] = None
        self._pending_classification_override: Optional[Classification] = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def set_script_plan(self, segments: Iterable[ScriptSegment]) -> None:
        """Replace the current script plan and reset the cursor."""
        self.script_plan = list(segments)
        self.current_index = 0
        logger.debug("Loaded %d script segments for %s", len(self.script_plan), self.monster_name)

    def reset_conversation(self) -> None:
        """Clear history and rewind script pointer."""
        self.history.clear()
        self.current_index = 0

    def queue_phase_override(
        self,
        phase: Phase,
        *,
        classification: Optional[Classification] = None,
    ) -> None:
        """Force the next generated turn to adopt a specific phase."""
        self._pending_phase_override = phase
        self._pending_classification_override = classification

    def remember_turn(self, record: TurnRecord) -> None:
        """Persist a turn into rolling history."""
        self.history.append(record)
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns :]

    def generate_turn(
        self,
        participant_text: Optional[str],
        *,
        participant_role: Literal["parent", "child", "system"] = "parent",
        extra_context: Optional[Dict[str, str]] = None,
    ) -> MonsterTurn:
        """
        Produce the monster's next turn in a structured format.

        Args:
            participant_text: Latest microphone transcription from the caller.
            participant_role: Who spoke (parent, child, or system hint).
            extra_context: Additional metadata (custom scenario, callback, etc).
        """
        cleaned_text = participant_text.strip() if participant_text else None
        if cleaned_text:
            self.remember_turn(
                TurnRecord(
                    speaker="participant",
                    text=cleaned_text,
                )
            )

        override_phase, override_classification = self._consume_phase_override()
        forced_segment = self._segment_for_phase(override_phase) if override_phase else None
        current_segment = forced_segment or self._current_segment()

        speech, control = self._run_model_or_fallback(
            participant_text=participant_text or "",
            participant_role=participant_role,
            extra_context=extra_context or {},
            current_segment=current_segment,
            forced_phase=override_phase,
            forced_classification=override_classification,
        )
        # Basic guardrail pass on output text (forbidden content, soften threats)
        if settings.enable_guardrails:
            ok, reason, cleaned = enforcer.check_output(speech, extra_context or {})
            if not ok and reason:
                logger.warning("Guardrails adjusted output due to: %s", reason)
            speech = cleaned

        sentences = self._split_into_sentences(speech)
        full_text = " ".join(sentences).strip() or speech.strip()

        phase = control.get("phase")
        if phase not in DEFAULT_CLASSIFICATION_FOR_PHASE:
            phase = override_phase or (current_segment.phase if current_segment else "introduction")
            control["phase"] = phase
        classification = control.get("classification")
        if classification not in DEFAULT_CLASSIFICATION_FOR_PHASE.values():
            classification = (
                override_classification
                or DEFAULT_CLASSIFICATION_FOR_PHASE[phase]  # type: ignore[index]
            )
            control["classification"] = classification

        turn = MonsterTurn(
            sentences=sentences,
            control=control,
            phase=phase,  # type: ignore[arg-type]
            classification=classification,  # type: ignore[arg-type]
            full_text=full_text or speech,
            source_segment_label=current_segment.label if current_segment else None,
        )

        self.remember_turn(
            TurnRecord(
                speaker="monster",
                text=turn.full_text,
                phase=turn.phase,
                classification=turn.classification,
            )
        )

        logger.debug(
            "Generated turn for %s (phase=%s, sentences=%d)",
            self.monster_name,
            turn.phase,
            len(turn.sentences),
        )
        return turn

    # --------------------------------------------------------------------- #
    # Core logic
    # --------------------------------------------------------------------- #
    def _run_model_or_fallback(
        self,
        *,
        participant_text: str,
        participant_role: str,
        extra_context: Dict[str, str],
        current_segment: Optional[ScriptSegment],
        forced_phase: Optional[Phase],
        forced_classification: Optional[Classification],
    ) -> Tuple[str, Dict[str, object]]:
        if self.client is None:
            speech, control = self._fallback_response(
                current_segment,
                participant_text,
                forced_phase=forced_phase,
                forced_classification=forced_classification,
            )
            self._advance_cursor(control.get("phase"))
            return speech, control

        messages = self._build_messages(
            current_segment=current_segment,
            participant_text=participant_text,
            participant_role=participant_role,
            extra_context=extra_context,
        )

        try:
            logger.debug("Sending %d messages to OpenAI", len(messages))
            content = self._call_llm_with_timeout(messages)
            speech, control = self._parse_model_output(content, current_segment)
        except APIError as error:  # pragma: no cover - network failure path
            logger.exception("OpenAI API error: %s", error)
            speech, control = self._fallback_response(
                current_segment,
                participant_text,
                forced_phase=forced_phase,
                forced_classification=forced_classification,
            )
        except TimeoutError:
            logger.warning("LLM timeout after %.2fs; using scripted fallback", self.request_timeout_seconds)
            speech, control = self._fallback_response(
                current_segment,
                participant_text,
                forced_phase=forced_phase,
                forced_classification=forced_classification,
            )
        except (KeyError, ValueError, IndexError) as parse_error:
            logger.exception("Failed to parse OpenAI response: %s", parse_error)
            speech, control = self._fallback_response(
                current_segment,
                participant_text,
                forced_phase=forced_phase,
                forced_classification=forced_classification,
            )

        control = self._validate_control(
            control,
            current_segment=current_segment,
            forced_phase=forced_phase,
            forced_classification=forced_classification,
            extra_context=extra_context,
        )
        self._advance_cursor(control.get("phase"))

        return speech, control

    def _call_llm_with_timeout(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI with a per-request timeout, returning content or raising TimeoutError."""
        if self.client is None:
            raise TimeoutError("Client not available")
        # Try client.with_options(timeout=..)
        try:
            client_with_timeout = getattr(self.client, "with_options", None)
            if callable(client_with_timeout):
                c = client_with_timeout(timeout=self.request_timeout_seconds)
                completion = c.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=messages,
                )
            else:
                # Fallback: pass timeout directly if supported
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=messages,
                    timeout=self.request_timeout_seconds,
                )
            return completion.choices[0].message.content or ""
        except Exception as e:
            # Re-map generic timeouts to TimeoutError where possible
            text = str(e).lower()
            if "timeout" in text or "timed out" in text:
                raise TimeoutError(text)
            raise

    def _build_messages(
        self,
        *,
        current_segment: Optional[ScriptSegment],
        participant_text: str,
        participant_role: str,
        extra_context: Dict[str, str],
    ) -> List[Dict[str, str]]:
        """Compose system + history + latest user payload for GPT-5 Nano."""
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]

        for record in self.history:
            role = "assistant" if record.speaker == "monster" else "user"
            prefix = f"{record.speaker.upper()}"
            if record.phase:
                prefix += f" [{record.phase}]"
            if record.classification:
                prefix += f" ({record.classification})"
            messages.append({"role": role, "content": f"{prefix}: {record.text}"})

        context_lines: List[str] = [
            f"Monster name: {self.monster_name}",
        ]

        if current_segment:
            context_lines.append(
                f"Baseline script segment ({current_segment.phase}): {current_segment.text}"
            )
        else:
            context_lines.append("Baseline script segment: NONE (improvise kindly).")

        remaining = self._remaining_segments(after=current_segment)
        if remaining:
            remaining_text = "; ".join(
                f"{segment.phase}: {segment.text}" for segment in remaining[:5]
            )
            context_lines.append(f"Upcoming segments: {remaining_text}")

        for key, value in extra_context.items():
            context_lines.append(f"{key}: {value}")

        participant_line = (
            f"Latest {participant_role} message: {participant_text.strip() or '[silence]'}"
        )
        context_lines.append(participant_line)

        guidance = (
            "Use the baseline script as the spine of your reply, but you may lightly adjust "
            "wording to acknowledge the latest message. Keep it playful, short, and kid-safe."
        )

        messages.append(
            {
                "role": "user",
                "content": "\n".join(context_lines + ["", guidance]),
            }
        )

        return messages

    def _parse_model_output(
        self,
        content: str,
        current_segment: Optional[ScriptSegment],
    ) -> Tuple[str, Dict[str, object]]:
        """Split model output into speech + control JSON."""
        speech_lines: List[str] = []
        control_lines: List[str] = []
        in_control_block = False

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("SPEECH:"):
                remainder = line.split(":", 1)[1].strip()
                if remainder:
                    speech_lines.append(remainder)
                continue

            if upper.startswith("CONTROL:"):
                inline = line.split(":", 1)[1].strip()
                if inline.startswith("{") and inline.endswith("}"):
                    control_lines.append(inline)
                    in_control_block = False
                else:
                    in_control_block = True
                continue

            if line.startswith("```"):
                in_control_block = not in_control_block
                continue
            
            # Check if this line looks like JSON (starts with { or contains JSON-like content)
            if line.startswith("{") or ('"phase"' in line or '"classification"' in line):
                in_control_block = True
                control_lines.append(raw_line)
                continue
            
            # If we see a closing brace, we're likely ending a JSON block
            if line.endswith("}") and in_control_block:
                control_lines.append(raw_line)
                in_control_block = False
                continue

            if in_control_block:
                control_lines.append(raw_line)
            else:
                speech_lines.append(raw_line)

        speech = " ".join(speech_lines).strip()
        
        # Additional cleanup: remove any remaining JSON-like content from speech
        if speech and "{" in speech:
            # Split by { and take only the part before it
            speech = speech.split("{")[0].strip()
        
        # Remove any prefix patterns like "Monster [consequence] (ongoing):"
        speech = re.sub(r'^(?:Monster|MONSTER)\s*\[.*?\]\s*\(.*?\)\s*:\s*', '', speech, flags=re.IGNORECASE)
        speech = re.sub(r'^(?:Monster|MONSTER)\s*\(.*?\)\s*:\s*', '', speech, flags=re.IGNORECASE)
        speech = re.sub(r'^(?:Monster|MONSTER)\s*\[.*?\]\s*:\s*', '', speech, flags=re.IGNORECASE)
        speech = re.sub(r'^(?:Monster|MONSTER)\s*:\s*', '', speech, flags=re.IGNORECASE)
        
        if not speech:
            speech = current_segment.text if current_segment else "I'm here to help. Tell me more."

        control_text = "\n".join(control_lines).strip()
        if not control_text:
            # fallback: attempt to grab first JSON object in content
            match = re.search(r"\{[\s\S]+\}", content)
            if match:
                control_text = match.group(0)
        control_data: Dict[str, object] | None = None
        if control_text:
            try:
                control_data = json.loads(control_text)
            except json.JSONDecodeError:
                logger.warning(
                    "Invalid CONTROL JSON; falling back to scripted control. Raw control block: %s | Speech lines: %s",
                    control_text,
                    speech_lines,
                )
        else:
            logger.warning(
                "Model output missing CONTROL block; falling back to scripted control. Speech lines: %s",
                speech_lines,
            )

        if control_data is None:
            control_data = {}

        return speech, control_data

    def _validate_control(
        self,
        control: Dict[str, object],
        current_segment: Optional[ScriptSegment],
        *,
        forced_phase: Optional[Phase],
        forced_classification: Optional[Classification],
        extra_context: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        """Ensure control payload has required keys and defaults."""
        phase = control.get("phase")
        if phase not in DEFAULT_CLASSIFICATION_FOR_PHASE:
            if forced_phase:
                phase = forced_phase
            elif current_segment:
                logger.debug(
                    "Control missing/invalid phase '%s'; defaulting to current segment %s",
                    phase,
                    current_segment.phase,
                )
                phase = current_segment.phase
            else:
                phase = "introduction"

        classification = control.get("classification")
        allowed_classifications = set(DEFAULT_CLASSIFICATION_FOR_PHASE.values())
        if forced_classification:
            classification = forced_classification
        elif classification not in allowed_classifications:
            classification = DEFAULT_CLASSIFICATION_FOR_PHASE[phase]  # type: ignore[index]

        # Clamp intensity based on policy/age bucket
        if settings.enable_guardrails:
            max_intensity = enforcer.get_max_intensity(extra_context or {})
            intensity = min(max_intensity, self._clamp_intensity(control.get("intensity", 0)))
        else:
            intensity = self._clamp_intensity(control.get("intensity", 0))
        validated = {
            "phase": phase,
            "classification": classification,
            "intensity": intensity,
            "defer_to_parent": bool(control.get("defer_to_parent", False)),
            "end_call": bool(control.get("end_call", False)),
        }

        return validated

    def _advance_cursor(self, phase: Optional[str]) -> None:
        if not self.script_plan or phase is None:
            return

        if self.current_index >= len(self.script_plan):
            return

        current_phase = self.script_plan[self.current_index].phase
        if phase == current_phase:
            self.current_index += 1
            logger.debug("Advanced script cursor to index %d", self.current_index)

    def _current_segment(self) -> Optional[ScriptSegment]:
        if self.current_index < len(self.script_plan):
            return self.script_plan[self.current_index]
        return None

    def _remaining_segments(
        self,
        after: Optional[ScriptSegment],
    ) -> List[ScriptSegment]:
        if not self.script_plan:
            return []
        start = self.current_index + 1 if after else self.current_index
        return self.script_plan[start:]

    @staticmethod
    def _clamp_intensity(value: object) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = 0
        return max(0, min(numeric, 3))

    def _fallback_response(
        self,
        current_segment: Optional[ScriptSegment],
        participant_text: str,
        *,
        forced_phase: Optional[Phase] = None,
        forced_classification: Optional[Classification] = None,
    ) -> Tuple[str, Dict[str, object]]:
        """Graceful degradation when OpenAI is unavailable."""
        base_text = current_segment.text if current_segment else "I'm here to help you behave kindly."
        if participant_text:
            speech = f"{base_text} I hear you saying \"{participant_text.strip()}\"."
        else:
            speech = base_text

        phase = forced_phase or (current_segment.phase if current_segment else "introduction")
        control = {
            "phase": phase,
            "classification": forced_classification or DEFAULT_CLASSIFICATION_FOR_PHASE[phase],
            "intensity": 0,
            "defer_to_parent": False,
            "end_call": False,
        }

        return speech, control

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Break a response into short sentences for streaming with graceful fallback.
        """
        if not text:
            return ["I'm here to help."]

        pattern = r"[^.!?]+[.!?]"
        matches = re.findall(pattern, text)
        sentences = [chunk.strip() for chunk in matches if chunk.strip()]

        remainder = text.strip()
        if sentences:
            consumed = "".join(matches).strip()
            if len(remainder) > len(consumed):
                tail = remainder[len(consumed) :].strip()
                if tail:
                    sentences.append(tail)
        else:
            sentences = [remainder]

        return sentences

    def _segment_for_phase(self, phase: Optional[Phase]) -> Optional[ScriptSegment]:
        if phase is None:
            return None
        for segment in self.script_plan:
            if segment.phase == phase:
                return segment
        return None

    def _consume_phase_override(self) -> Tuple[Optional[Phase], Optional[Classification]]:
        phase, classification = self._pending_phase_override, self._pending_classification_override
        self._pending_phase_override = None
        self._pending_classification_override = None
        return phase, classification

    def _build_system_prompt(self) -> str:
        prompt_parts: List[str] = []

        def _read_prompt(filename: str) -> str:
            path = self.prompt_dir / filename
            if not path.exists():
                return ""
            return path.read_text(encoding="utf-8").strip()

        prompt_parts.append(_read_prompt("system_boogeyman.md"))
        content_rules = _read_prompt("content_rules.md")
        if content_rules:
            prompt_parts.append("Content rules:\n" + content_rules)
        parent_guide = _read_prompt("parent_guide.md")
        if parent_guide:
            prompt_parts.append("Parent guidance:\n" + parent_guide)

        prompt_parts.append(
            (
                "You speak as the Boogeyman using Cartesia voice id "
                f"{self.voice_id}. Keep lines playful, 1–2 sentences, and under 25 words.\n"
                "Always honor parent overrides immediately.\n"
            )
        )

        format_instructions = """
    Always respond EXACTLY with this template:

    SPEECH:
    <one or two sentences of dialogue - NO PREFIXES, just the spoken words>

    CONTROL:
    ```json
    {
      "phase": "introduction|situation|consequence|escalate|de_escalate|result_positive|result_negative|goodbye",
      "classification": "ongoing|escalation|praise|resolved_positive|resolved_negative",
      "intensity": 0-3,
      "defer_to_parent": true/false,
      "end_call": true/false
    }
    ```

    IMPORTANT: Do NOT include any speaker labels, phase markers like [consequence], or classification markers like (ongoing) in your SPEECH output. Only include the actual words you want to speak.
    Do not emit any other sections, markdown, or commentary.
    """
        prompt_parts.append(format_instructions.strip())

        prompt_parts.append(
            "When adapting a scripted line, keep the core meaning but feel free to add a short nod to "
            "the latest participant message. Mention callbacks if context flag 'is_callback' is true."
        )

        return "\n\n".join(part for part in prompt_parts if part)
