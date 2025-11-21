from __future__ import annotations

import asyncio
import re
import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union

from line.bus import Message
from line.events import AgentResponse, LogMetric
from line.nodes.conversation_context import ConversationContext
from line.nodes.reasoning import ReasoningNode

from ..config import settings

from .boogeyman_scripts import (
    build_boogeyman_plan,
    get_display_names,
)
from .controller import (
    Classification,
    DEFAULT_CLASSIFICATION_FOR_PHASE,
    MonsterConversationController,
    MonsterTurn,
    Phase,
    ScriptSegment,
)
from .events import StageSignal

BOOGEYMAN_VOICE_ID = settings.cartesia_voice_id or "2bfe5f2c-db54-4a8e-80fb-898298c0f0e6"
logger = logging.getLogger(__name__)


class BoogeymanNode:
    """Wrapper around the conversation controller for non-Line integrations."""

    def __init__(
        self,
        *,
        script_plan: Optional[Iterable[Union[ScriptSegment, Dict[str, str]]]] = None,
        model: Optional[str] = None,
    ) -> None:
        model = model or settings.llm_model
        if script_plan:
            segments = self._ensure_segments(script_plan)
        else:
            segments = build_boogeyman_plan(
                scenario="bedtime",
                consequence="getYou",
                is_high_fear=False,
                is_callback=False,
            )
        self.controller = MonsterConversationController(
            monster_name="Boogeyman",
            voice_id=BOOGEYMAN_VOICE_ID,
            model=model,
            script_plan=segments,
            max_history_turns=settings.max_conversation_turns,
        )
        self._last_config: Optional[tuple] = None

    def update_script_plan(
        self, script_plan: Iterable[Union[ScriptSegment, Dict[str, str]]]
    ) -> None:
        segments = self._ensure_segments(script_plan)
        self.controller.set_script_plan(segments)
        self._last_config = None

    def reset(self) -> None:
        """Clears conversation history while keeping the current script plan."""
        self.controller.reset_conversation()
        self._last_config = None

    def generate(self, context: Optional[Dict[str, object]] = None):
        """
        Generate a single structured response.

        Expected context keys:
            - script_plan: optional iterable of script segments (dict or ScriptSegment)
            - participant_text / latest_user_message / user_text: newest utterance
            - participant_role: 'parent', 'child', or 'system' (defaults to parent)
            - extra_context: mapping of additional strings (e.g., child_name, scenario)
        """
        context = context or {}

        if "script_plan" in context and context["script_plan"]:
            self.update_script_plan(context["script_plan"])  # type: ignore[arg-type]

        scenario_raw = context.get("scenario") or context.get("scenario_id")
        consequence_raw = context.get("consequence") or context.get("consequence_id")
        fear_flag = context.get("is_high_fear") or context.get("fear_level")
        is_high_fear = str(fear_flag).lower() in {"1", "true", "yes", "high"} if fear_flag else False
        is_callback = bool(context.get("is_callback"))

        config_key = (
            self._as_key(scenario_raw),
            self._as_key(consequence_raw),
            bool(is_high_fear),
            bool(is_callback),
        )

        if (scenario_raw or consequence_raw) and config_key != self._last_config:
            plan = build_boogeyman_plan(
                scenario=scenario_raw if isinstance(scenario_raw, str) else None,
                consequence=consequence_raw if isinstance(consequence_raw, str) else None,
                is_high_fear=is_high_fear,
                is_callback=is_callback,
            )
            self.controller.set_script_plan(plan)
            self._last_config = config_key

        participant_text = self._extract_text(context)
        participant_role = str(context.get("participant_role", "parent")).lower() or "parent"
        if participant_role not in {"parent", "child", "system"}:
            participant_role = "parent"
        extra_context = context.get("extra_context") or {}
        if not isinstance(extra_context, dict):
            extra_context = {}

        scenario_display, consequence_display = get_display_names(
            scenario_raw if isinstance(scenario_raw, str) else None,
            consequence_raw if isinstance(consequence_raw, str) else None,
        )

        enriched_context = {k: str(v) for k, v in extra_context.items()}
        if scenario_display:
            enriched_context["scenario"] = scenario_display
        if consequence_display:
            enriched_context["consequence"] = consequence_display
        if isinstance(scenario_raw, str):
            enriched_context["scenario_id"] = scenario_raw
        if isinstance(consequence_raw, str):
            enriched_context["consequence_id"] = consequence_raw
        enriched_context["fear_level"] = "high" if is_high_fear else "low"
        if is_callback:
            enriched_context["is_callback"] = "true"

        response = self.controller.generate_turn(
            participant_text,
            participant_role=participant_role,  # type: ignore[arg-type]
            extra_context=enriched_context,
        )
        yield response

    @staticmethod
    def _ensure_segments(
        segments: Iterable[Union[ScriptSegment, Dict[str, str]]]
    ) -> List[ScriptSegment]:
        result: List[ScriptSegment] = []
        for segment in segments:
            if isinstance(segment, ScriptSegment):
                result.append(segment)
            elif isinstance(segment, dict):
                result.append(ScriptSegment.from_dict(segment))
            else:
                raise TypeError(f"Unsupported script segment type: {type(segment)}")
        return result

    @staticmethod
    def _extract_text(context: Dict[str, object]) -> Optional[str]:
        for key in ("participant_text", "latest_user_message", "user_text", "transcript"):
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    @staticmethod
    def _as_key(value: Optional[object]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip().lower()
        return str(value).strip().lower()


class BoogeymanReasoningNode(ReasoningNode):
    """
    Line-compatible reasoning node that streams Boogeyman responses using the shared controller.
    """

    PHASE_TO_STAGE: Dict[Phase, str] = {
        "introduction": "intro",
        "situation": "intro",
        "consequence": "intro",
        "escalate": "escalate",
        "de_escalate": "praise",
        "result_positive": "resolved",
        "result_negative": "resolved",
        "goodbye": "resolved",
    }

    HINT_TO_PHASE: Dict[str, Tuple[Phase, Classification]] = {
        "escalate": ("escalate", "escalation"),
        "praise": ("de_escalate", "praise"),
        "result_positive": ("result_positive", "resolved_positive"),
        "result_negative": ("result_negative", "resolved_negative"),
    }

    PARENT_OVERRIDE_TO_PHASE: Dict[str, Tuple[Phase, Classification]] = {
        "lighter": ("de_escalate", "praise"),
        "lighten": ("de_escalate", "praise"),
        "too_scary": ("de_escalate", "praise"),
        "back_off": ("de_escalate", "praise"),
        "calmer": ("de_escalate", "praise"),
        "softer": ("de_escalate", "praise"),
        "end": ("result_positive", "resolved_positive"),
        "end_call": ("result_positive", "resolved_positive"),
        "wrap_up": ("result_positive", "resolved_positive"),
    }

    def __init__(
        self,
        *,
        metadata: Optional[Dict[str, object]] = None,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(system_prompt="Call the Monster — Boogeyman reasoning node")
        self.metadata: Dict[str, object] = metadata or {}
        self.driver = BoogeymanNode(model=model or settings.llm_model)
        self._intro_sent = False
        self._base_context = self._build_base_context(self.metadata)

        # Setup wizard state: collect scenario, consequence, fear level, tone, and child name
        has_scenario = bool(self._as_str(self.metadata.get("scenario")) or self._as_str(self.metadata.get("scenario_id")))
        has_consequence = bool(self._as_str(self.metadata.get("consequence")) or self._as_str(self.metadata.get("consequence_id")))
        has_fear = (self.metadata.get("is_high_fear") is not None) or (self.metadata.get("fear_level") is not None)
        self._config_mode: bool = not (has_scenario and has_consequence and has_fear)
        self._config_step: int = 0
        self._config_data: Dict[str, object] = {
            "scenario": None,
            "consequence": None,
            "is_high_fear": None,
            "tone": None,
            "child_name": None,
        }

        plan = build_boogeyman_plan(
            scenario=self._as_str(self.metadata.get("scenario")),
            consequence=self._as_str(self.metadata.get("consequence")),
            is_high_fear=self._is_high_fear(self.metadata.get("is_high_fear") or self.metadata.get("fear_level")),
            is_callback=bool(self.metadata.get("is_callback")),
        )
        self.driver.controller.set_script_plan(plan)

    def update_metadata(self, metadata: Dict[str, object]) -> None:
        self.metadata.update(metadata)
        self._base_context = self._build_base_context(self.metadata)
        plan = build_boogeyman_plan(
            scenario=self._as_str(self.metadata.get("scenario")),
            consequence=self._as_str(self.metadata.get("consequence")),
            is_high_fear=self._is_high_fear(self.metadata.get("is_high_fear") or self.metadata.get("fear_level")),
            is_callback=bool(self.metadata.get("is_callback")),
        )
        self.driver.controller.set_script_plan(plan)

    def _prepare_intro_turn(self) -> Optional[MonsterTurn]:
        if self._intro_sent:
            return None
        turn = self._generate_turn(participant_text="", participant_role="system")
        if not turn:
            logger.warning("Intro turn missing; using scripted fallback.")
            turn = self._default_intro_turn()
        self._intro_sent = True
        return turn

    def generate_intro_prompt(self) -> Optional[str]:
        turn = self._prepare_intro_turn()
        if not turn:
            return None
        self._schedule_stage_signal(turn)
        return turn.full_text

    async def stream_intro_prompt(self) -> Optional[str]:
        """
        Broadcast the intro turn so clients hear audio before any transcription arrives.
        If setup wizard is enabled, ask the first question instead of the intro.
        """
        if self._config_mode:
            question = self._config_question(self._config_step)
            await self._broadcast_text(question)
            return question
        turn = self._prepare_intro_turn()
        if not turn:
            return None
        await self._broadcast_turn(turn)
        return turn.full_text
    
    def _default_intro_turn(self) -> MonsterTurn:
        name = None
        extras = self._base_context.get("extra_context") if isinstance(self._base_context, dict) else None
        if isinstance(extras, dict):
            raw = extras.get("child_name") or extras.get("name")
            if isinstance(raw, str) and raw.strip():
                name = raw.strip()
        if not name and isinstance(self.metadata, dict):
            raw = self.metadata.get("child_name") or self.metadata.get("name")
            if isinstance(raw, str) and raw.strip():
                name = raw.strip()
        if name:
            text = f"Hi {name}! I'm the Boogeyman helper checking in."
        else:
            text = "Hi there! I'm the Boogeyman helper checking in."
        control = {
            "phase": "introduction",
            "classification": DEFAULT_CLASSIFICATION_FOR_PHASE["introduction"],
            "intensity": 0,
            "defer_to_parent": False,
            "end_call": False,
        }
        return MonsterTurn(
            sentences=[text],
            control=control,
            phase="introduction",
            classification=control["classification"],
            full_text=text,
        )

    async def process_context(self, context: ConversationContext):
        participant_text = context.get_latest_user_transcript_message() or ""
        if not participant_text.strip() and self._intro_sent and not self._config_mode:
            return

        # Handle setup wizard before normal flow
        if self._config_mode:
            text = (participant_text or "").strip()
            if not text:
                return
            advanced = await self._handle_config_answer(text)
            if advanced:
                for event in advanced:
                    yield event
            return

        turn = self._generate_turn(
            participant_text=participant_text,
            participant_role="parent",
        )

        if turn:
            self._intro_sent = True
            for event in self._events_for_turn(turn):
                yield event

    def add_event(self, event):
        if isinstance(event, StageSignal):
            return
        super().add_event(event)

    def _generate_turn(self, *, participant_text: str, participant_role: str) -> Optional[MonsterTurn]:
        # Heuristic steering: infer phase from participant text when no explicit override is given
        self._maybe_queue_heuristic_override(participant_text)
        runtime_context = dict(self._base_context)
        runtime_context["participant_text"] = participant_text
        runtime_context["participant_role"] = participant_role

        generator = self.driver.generate(runtime_context)
        try:
            turn = next(generator)
        except StopIteration:
            return None
        if isinstance(turn, MonsterTurn):
            return turn
        if isinstance(turn, str):
            sentences = [segment.strip() for segment in turn.splitlines() if segment.strip()]
            full_text = " ".join(sentences) if sentences else turn
            return MonsterTurn(
                sentences=sentences or [turn],
                control={
                    "phase": "introduction",
                    "classification": "ongoing",
                    "intensity": 0,
                    "defer_to_parent": False,
                    "end_call": False,
                },
                phase="introduction",
                classification="ongoing",
                full_text=full_text,
            )
        return None

    def _maybe_queue_heuristic_override(self, text: str) -> None:
        if not text or not text.strip():
            return
        t = text.strip().lower()
        # Positive compliance cues
        praise_cues = [
            r"\b(i'm|i am) (doing|starting|getting) (it|this)\b",
            r"\b(done|finished|all done)\b",
            r"\b(clean(ing)?|brushing|ready|packed)\b",
            r"\b(ok|okay|alright|will do|i will)\b",
            r"\b(sorry)\b",
            r"\byes\b",
        ]
        # Refusal/defiance cues
        escalate_cues = [
            r"\b(no|never)\b",
            r"\b(won't|will not|not going to)\b",
            r"\b(don't want to|refuse)\b",
            r"\b(not listening|you can't)\b",
        ]
        # Resolution cues
        positive_result = [r"\b(done|finished|in bed|room is clean)\b"]
        negative_result = [r"\b(still not|not doing it|refuse)\b"]

        def any_match(patterns):
            return any(re.search(p, t) for p in patterns)

        ctrl = self.driver.controller
        try:
            if any_match(positive_result):
                ctrl.queue_phase_override("result_positive", classification="resolved_positive")
                return
            if any_match(negative_result):
                ctrl.queue_phase_override("result_negative", classification="resolved_negative")
                return
            if any_match(praise_cues):
                ctrl.queue_phase_override("de_escalate", classification="praise")
                return
            if any_match(escalate_cues):
                ctrl.queue_phase_override("escalate", classification="escalation")
                return
        except Exception:
            # Heuristic is best-effort; ignore on failure
            return

    def _phase_to_stage(self, phase: Phase) -> str:
        return self.PHASE_TO_STAGE.get(phase, "intro")

    def _stage_signal_for_turn(self, turn: MonsterTurn) -> StageSignal:
        return StageSignal(
            phase=turn.phase,
            classification=turn.classification,
            stage=self._phase_to_stage(turn.phase),
        )

    def _events_for_turn(self, turn: MonsterTurn) -> List[object]:
        stage_signal = self._stage_signal_for_turn(turn)
        events: List[object] = [
            stage_signal,
            LogMetric(name="stage_signal", value=stage_signal.as_metadata()),
        ]
        for sentence in turn.sentences:
            events.append(AgentResponse(content=sentence))
        return events

    def _schedule_stage_signal(self, turn: MonsterTurn) -> None:
        bridge = getattr(self, "_bridge", None)
        if bridge is None or bridge.bus is None:
            return
        signal = self._stage_signal_for_turn(turn)

        async def _broadcast():
            await bridge.bus.broadcast(Message(source=self.id, event=signal))

        asyncio.create_task(_broadcast())

    async def _broadcast_turn(self, turn: MonsterTurn) -> None:
        bridge = getattr(self, "_bridge", None)
        if bridge is None or bridge.bus is None:
            return

        for event in self._events_for_turn(turn):
            await bridge.bus.broadcast(Message(source=self.id, event=event))
            if isinstance(event, AgentResponse):
                super().add_event(event)

    async def _broadcast_text(self, text: str) -> None:
        bridge = getattr(self, "_bridge", None)
        if bridge is None or bridge.bus is None:
            return
        await bridge.bus.broadcast(Message(source=self.id, event=AgentResponse(content=text)))

    def _config_question(self, step: int) -> str:
        prompts = [
            "Quick setup: what's the scenario? bedtime, not listening, or mess.",
            "Pick a consequence. get you, monster's choice, or take something away.",
            "Fear level: say high or low.",
            "Tone preference:  Choose escalate or de escalate.",
            "What is the child's name?",
        ]
        return prompts[min(step, len(prompts) - 1)]

    def _llm_parse_config(self, step: int, user_input: str) -> Optional[str]:
        """Use LLM to parse ambiguous config answers when keyword matching fails."""
        if not self.driver.controller.client:
            return None
        
        prompts = {
            0: "User is choosing a scenario. Valid options: 'bedtime', 'not_listening', or 'mess'. User said: '{input}'. Reply with ONLY the exact option name (bedtime, not_listening, or mess) or 'unknown' if unclear.",
            1: "User is choosing a consequence. Valid options: 'getYou', 'monsterChoice', or 'takeSomethingAway'. User said: '{input}'. Reply with ONLY the exact option name (getYou, monsterChoice, or takeSomethingAway) or 'unknown' if unclear.",
            2: "User is choosing fear level. Valid options: 'high' or 'low'. User said: '{input}'. Reply with ONLY 'high' or 'low' or 'unknown' if unclear.",
            3: "User is choosing tone. Valid options: 'escalate' or 'de_escalate'. User said: '{input}'. Reply with ONLY 'escalate' or 'de_escalate' or 'unknown' if unclear.",
        }
        
        prompt = prompts.get(step)
        if not prompt:
            return None
        
        try:
            response = self.driver.controller.client.chat.completions.create(
                model=settings.llm_model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You are a classification assistant. Reply with ONLY the exact option or 'unknown'."},
                    {"role": "user", "content": prompt.format(input=user_input)}
                ],
                timeout=3.0
            )
            result = (response.choices[0].message.content or "").strip().lower()
            return result if result != "unknown" else None
        except Exception as e:
            logger.warning(f"LLM config parsing failed: {e}")
            return None

    async def _handle_config_answer(self, raw: str) -> List[object]:
        """Process a wizard answer; return a list of events to emit (AgentResponse/StageSignal)."""
        events: List[object] = []
        # Normalize but keep the original for names, etc.
        text = raw.strip().lower()
        if not text:
            # Do not advance on pure silence; gently reprompt.
            events.append(AgentResponse(content=self._config_question(self._config_step)))
            return events
        
        # Deduplicate repeated words: "high high" → "high", "low low low" → "low"
        words = text.split()
        if len(words) > 1 and len(set(words)) == 1:
            # All words are the same, collapse to single word
            text = words[0]
        
        step = self._config_step
        # Step 0: scenario
        if step == 0:
            # Extensive ASR-friendly keywords for scenario recognition
            bedtime_keywords = {
                "bedtime", "time to sleep", "go to sleep", "sleep time", "go to bed", "lay down",
                "sleep now", "night time", "it's night", "bed now", "bad time", "bed time",
                "bed-time", "bed side", "bed line", "bet time", "bed sign", "bed dime",
                "bed", "sleep", "night", "laying down"
            }
            not_listening_keywords = {
                "not listening", "won't listen", "doesn't listen", "ignoring me", "not paying attention",
                "he doesn't hear me", "she won't listen", "won't follow", "not following", "not obeying",
                "not listing", "not glistening", "not listen", "not listen ing", "no listening",
                "not hearing", "don't listen", "listen", "ignoring", "disobey", "won't mind"
            }
            mess_keywords = {
                "mess", "messy", "making a mess", "room is a mess", "everything messy", "dirty room",
                "clean up", "won't clean", "not cleaning", "clean your toys", "mass", "message",
                "messs", "met", "messed", "lesson", "clean", "dirty", "toys", "room", "mett"
            }
            
            is_bedtime = any(kw in text for kw in bedtime_keywords)
            is_not_listening = any(kw in text for kw in not_listening_keywords)
            is_mess = any(kw in text for kw in mess_keywords)
            
            # Prioritize most specific match
            if is_not_listening and not (is_bedtime or is_mess):
                val = "not_listening"
            elif is_bedtime and not is_mess:
                val = "bedtime"
            elif is_mess:
                val = "mess"
            else:
                val = None
            
            # If keyword matching failed, try LLM parsing
            if not val:
                llm_result = self._llm_parse_config(step, raw)
                if llm_result in {"bedtime", "not_listening", "mess"}:
                    val = llm_result
            
            if not val:
                events.append(AgentResponse(content="Please choose: bedtime, not listening, or mess."))
                return events
            self._config_data["scenario"] = val
            self._config_step += 1
            events.append(AgentResponse(content=self._config_question(self._config_step)))
            return events
        # Step 1: consequence
        if step == 1:
            mapping = {
                "getyou": "getYou",
                "get you": "getYou",
                "monsterchoice": "monsterChoice",
                "monster's choice": "monsterChoice",
                "monsters choice": "monsterChoice",
                "take something away": "takeSomethingAway",
                "takesomethingaway": "takeSomethingAway",
            }
            key = text.replace("'", "")
            val = mapping.get(key)
            if not val:
                if "monster" in key or "scare" in key:
                    val = "monsterChoice"
                elif "take" in key or "away" in key or "toy" in key:
                    val = "takeSomethingAway"
                elif "get" in key or "come" in key:
                    val = "getYou"
            # If keyword matching failed, try LLM parsing
            if not val:
                llm_result = self._llm_parse_config(step, raw)
                if llm_result == "getyou":
                    val = "getYou"
                elif llm_result == "monsterchoice":
                    val = "monsterChoice"
                elif llm_result == "takesomethingaway":
                    val = "takeSomethingAway"
            
            if not val:
                events.append(AgentResponse(content="Choose: getYou, monsterChoice, or takeSomethingAway."))
                return events
            self._config_data["consequence"] = val
            self._config_step += 1
            events.append(AgentResponse(content=self._config_question(self._config_step)))
            return events
        # Step 2: fear level
        if step == 2:
            # Allow numeric options and phrases like "keep it really scary" or "not too scary, keep it low".
            # Convention: 1 = high, 2 = low.
            # Extensive ASR-friendly keywords for high and low
            high_keywords = {
                "high", "hi", "hie", "hight", "height", "hy", "hye", "hai", "hey", "heigh", "haye",
                "haai", "haa", "i", "ay", "aye", "why", "sky-high", "too high", "very high",
                "h", "1", "true", "yes", "really scary", "very scary", "scary"
            }
            low_keywords = {
                "low", "lo", "loo", "lowe", "slow", "lowa", "lough", "below", "no", "not high",
                "little", "a little", "small", "less", "minimum", "tiny", "fine", "normal", "okay",
                "l", "0", "2", "false", "gentle", "not too scary", "not scary"
            }
            
            is_high = any(kw in text for kw in high_keywords)
            is_low = any(kw in text for kw in low_keywords)
            
            # If keyword matching failed, try LLM parsing
            if not (is_high or is_low):
                llm_result = self._llm_parse_config(step, raw)
                if llm_result == "high":
                    is_high = True
                elif llm_result == "low":
                    is_low = True
            
            if not (is_high or is_low):
                events.append(AgentResponse(content="Say 'high' or 'low'."))
                return events
            self._config_data["is_high_fear"] = bool(is_high and not is_low)
            self._config_step += 1
            events.append(AgentResponse(content=self._config_question(self._config_step)))
            return events
        # Step 3: tone
        if step == 3:
            # Check exact matches and simple cases first with priority logic
            tone = None
            
            # Numeric options
            if text in {"2", "two"}:
                tone = "de_escalate"
            elif text in {"1", "one"}:
                tone = "escalate"
            # Check for "de" prefix variations (high priority for de-escalate)
            elif "de " in text or "de-" in text or "de_" in text or text.startswith("de"):
                tone = "de_escalate"
            else:
                # Keyword matching with lists for better control
                escalate_keywords = [
                    "escalate", "escalator", "esculate", "excalate", "escalation", "escape late",
                    "extra late", "scale it", "scale up", "speed up", "pick up"
                ]
                deescalate_keywords = [
                    "deescalate", "cool down", "reduce", "make it low", "low it down", "go lower",
                    "calm it", "less scary", "lower level", "go down", "reduce it", "slow down",
                    "soften it", "tone it down", "undo", "back off", "calm", "praise", "gentle",
                    "softer", "calmer", "lighter", "ease up"
                ]
                
                # Check de-escalate first (more specific)
                if any(kw in text for kw in deescalate_keywords):
                    tone = "de_escalate"
                # Only check escalate if no de-escalate match
                elif any(kw in text for kw in escalate_keywords):
                    tone = "escalate"
            
            # If keyword matching failed, try LLM parsing
            if not tone:
                llm_result = self._llm_parse_config(step, raw)
                if llm_result == "escalate":
                    tone = "escalate"
                elif llm_result and "escalate" in llm_result:
                    # Handles "de_escalate", "de-escalate", "deescalate"
                    tone = "de_escalate"
            
            if not tone:
                events.append(AgentResponse(content="Say 'escalate' or 'de escalate'."))
                return events
            self._config_data["tone"] = tone
            self._config_step += 1
            events.append(AgentResponse(content=self._config_question(self._config_step)))
            return events
        # Step 4: child name
        if step == 4:
            name = raw.strip()
            self._config_data["child_name"] = name if name else None
            # Apply configuration and exit wizard
            self._config_mode = False
            # Merge into metadata and base context
            self.metadata["scenario"] = self._config_data["scenario"]
            self.metadata["consequence"] = self._config_data["consequence"]
            self.metadata["is_high_fear"] = self._config_data["is_high_fear"]
            self._base_context = self._build_base_context(self.metadata)
            # Rebuild plan
            plan = build_boogeyman_plan(
                scenario=self._as_str(self.metadata.get("scenario")),
                consequence=self._as_str(self.metadata.get("consequence")),
                is_high_fear=self._is_high_fear(self.metadata.get("is_high_fear") or self.metadata.get("fear_level")),
                is_callback=bool(self.metadata.get("is_callback")),
            )
            self.driver.controller.set_script_plan(plan)
            # Apply tone preference
            tone = self._config_data.get("tone")
            if tone == "escalate":
                self.driver.controller.queue_phase_override("escalate", classification="escalation")
            elif tone == "de_escalate":
                self.driver.controller.queue_phase_override("de_escalate", classification="praise")
            # Personalized intro
            extra = {}
            if self._config_data.get("child_name"):
                extra["child_name"] = str(self._config_data["child_name"])
            turn = self._generate_turn(participant_text="", participant_role="system")
            if turn:
                # Prepend a personalized salutation if not present
                if extra.get("child_name") and not any(extra["child_name"] in s for s in turn.sentences):
                    greet = f"Hi {extra['child_name']}! I'm the Boogeyman helper checking in."
                    events.append(AgentResponse(content=greet))
                for ev in self._events_for_turn(turn):
                    events.append(ev)
                self._intro_sent = True
            return events
        # Fallback
        events.append(AgentResponse(content=self._config_question(self._config_step)))
        return events

    async def _force_phase(self, phase: Phase, classification: Classification) -> None:
        self.driver.controller.queue_phase_override(phase, classification=classification)
        turn = self._generate_turn(participant_text="", participant_role="system")
        if turn:
            self._intro_sent = True
            await self._broadcast_turn(turn)

    async def handle_ui_hint(self, hint: str) -> None:
        mapping = self.HINT_TO_PHASE.get(hint.lower())
        if not mapping:
            return
        await self._force_phase(*mapping)

    async def handle_parent_override(self, command: str) -> None:
        mapping = self.PARENT_OVERRIDE_TO_PHASE.get(command.lower())
        if not mapping:
            return
        await self._force_phase(*mapping)

    @staticmethod
    def _build_base_context(metadata: Dict[str, object]) -> Dict[str, object]:
        scenario = metadata.get("scenario") or metadata.get("scenario_id")
        consequence = metadata.get("consequence") or metadata.get("consequence_id")
        fear_flag = metadata.get("is_high_fear") or metadata.get("fear_level")
        is_high_fear = BoogeymanReasoningNode._is_high_fear(fear_flag)

        base_context: Dict[str, object] = {
            "scenario": scenario,
            "consequence": consequence,
            "is_high_fear": is_high_fear,
            "is_callback": metadata.get("is_callback", False),
        }

        extra_fields = {}
        for key in ("child_name", "custom_scenario", "customScenario", "notes"):
            if key in metadata:
                extra_fields[key] = metadata[key]

        if extra_fields:
            base_context["extra_context"] = extra_fields
        return base_context

    @staticmethod
    def _is_high_fear(flag: Optional[object]) -> bool:
        if isinstance(flag, bool):
            return flag
        if isinstance(flag, (int, float)):
            return flag >= 1
        if isinstance(flag, str):
            return flag.strip().lower() in {"1", "true", "yes", "high"}
        return False

    @staticmethod
    def _as_str(value: Optional[object]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)
