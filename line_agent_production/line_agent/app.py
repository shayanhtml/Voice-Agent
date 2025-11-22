from __future__ import annotations

import json
import logging
from typing import Dict

from line import Bridge, CallRequest, VoiceAgentApp, VoiceAgentSystem
from line.bus import Message
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi import Body, Header
import os
import time
import uuid
import asyncio
from line.events import (
    UserStartedSpeaking,
    UserStoppedSpeaking,
    UserTranscriptionReceived,
    UserUnknownInputReceived,
    AgentResponse,
)
from line.user_bridge import register_observability_event

from .config import settings
from .nodes import StageSignal
from .nodes.boogeyman_node import BoogeymanReasoningNode, BoogeymanNode
from .nodes.boogeyman_scripts import build_boogeyman_plan
from .auth import generate_token, verify_token
from .integrations.cartesia_tts import CartesiaTTS

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)


async def handle_unknown_input(system: VoiceAgentSystem, node: BoogeymanReasoningNode, message):
    event = message.event
    raw = getattr(event, "input_data", None)
    if not isinstance(raw, str):
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return

    payload_type = str(payload.get("type", "")).lower()
    if payload_type == "ui_hint":
        hint = payload.get("hint") or payload.get("value")
        if isinstance(hint, str) and hint:
            await node.handle_ui_hint(hint)
    elif payload_type in {"parent_override", "override"}:
        command = payload.get("command") or payload.get("value")
        if isinstance(command, str) and command:
            await node.handle_parent_override(command)
    elif payload_type == "session_control":
        command = str(payload.get("command") or payload.get("action") or payload.get("value") or "").lower()
        if command in {"end", "end_call", "hangup"}:
            logger.info("Session control received: %s", command)
            await system.harness.end_call()

def attach_routes(system: VoiceAgentSystem, node: BoogeymanReasoningNode, bridge: Bridge) -> None:
    def log_and_buffer_transcription(message):
        event = message.event
        if isinstance(event, UserTranscriptionReceived):
            logger.info("📝 Transcription received: %s", event.content)
        node.add_event(event)

    bridge.on(UserTranscriptionReceived).map(log_and_buffer_transcription)

    (
        bridge.on(UserStoppedSpeaking)
        .interrupt_on(UserStartedSpeaking)
        .stream(node.generate)
        .broadcast()
    )

    bridge.on(UserStartedSpeaking).map(node.on_interrupt_generate)
    bridge.on(UserUnknownInputReceived).map(lambda message: handle_unknown_input(system, node, message))

    register_observability_event(bridge, system.harness, StageSignal)

    # Idle timeout watchdog (Line runtime)
    idle_seconds = settings.idle_timeout_seconds
    last_activity = {"t": asyncio.get_event_loop().time()}

    def _touch_activity(_msg=None):
        last_activity["t"] = asyncio.get_event_loop().time()

    bridge.on(UserTranscriptionReceived).map(lambda m: _touch_activity(m))
    bridge.on(UserStartedSpeaking).map(lambda m: _touch_activity(m))
    bridge.on(UserStoppedSpeaking).map(lambda m: _touch_activity(m))

    async def _idle_watchdog():
        try:
            while True:
                await asyncio.sleep(2)
                now = asyncio.get_event_loop().time()
                if now - last_activity["t"] > idle_seconds:
                    logger.info("Idle %ss reached; sending closing line and ending call", idle_seconds)
                    try:
                        closing = "I have other kids to check on. I’ll come back."
                        await bridge.bus.broadcast(
                            Message(source=node.id, event=AgentResponse(content=closing))
                        )
                    except Exception:
                        logger.warning("Could not broadcast closing line; ending call anyway")
                    await system.harness.end_call()
                    return
        except Exception as exc:
            logger.exception("Idle watchdog error: %s", exc)

    asyncio.create_task(_idle_watchdog())

async def handle_new_call(system: VoiceAgentSystem, call_request: CallRequest):
    """Configure the Boogeyman reasoning node for the active call."""
    metadata: Dict[str, object] = call_request.metadata or {}
    node = BoogeymanReasoningNode(metadata=metadata)
    bridge = Bridge(node)

    system.with_speaking_node(node, bridge=bridge)
    attach_routes(system, node, bridge)

    await system.start()

    intro = await node.stream_intro_prompt()
    if intro:
        logger.info("Intro prompt streamed (%s chars)", len(intro))
    else:
        fallback = "Hi there! I'm the Boogeyman helper checking in."
        logger.warning("Intro turn missing; sending fallback via harness.")
        await system.send_initial_message(fallback)

    await system.wait_for_shutdown()


voice_agent = VoiceAgentApp(handle_new_call)
app = voice_agent.fastapi_app


def main():
    voice_agent.run()


if __name__ == "__main__":
    main()
# ----------------------------
# Token issuance for WebSocket
# ----------------------------

@app.post("/token")
async def issue_token(
    payload: Dict[str, object] = Body(default={}),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
):
    expected = settings.cartesia_api_key
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="unauthorized")

    session_id = str(payload.get("session_id") or payload.get("sid") or uuid.uuid4())
    agent_id = payload.get("agent_id") or payload.get("aid")
    ttl = int(payload.get("ttl_seconds") or 1800)
    token, exp = generate_token(session_id=session_id, agent_id=str(agent_id) if agent_id else None, ttl_seconds=ttl)
    return {"token": token, "expires": exp, "session_id": session_id}


# ----------------------------
# Health/Version endpoints
# ----------------------------

@app.get("/healthz")
async def healthz():
    # Basic readiness check; extend with dependency checks as needed
    ok = True
    details = {
        "openai": bool(settings.openai_api_key),
        "token_secret": bool(settings.token_secret),
        "cartesia": bool(settings.cartesia_api_key),
    }
    return {"ok": ok, "details": details}


@app.get("/health")
async def health():
    # Alias for health checks expecting /health
    return await healthz()


@app.get("/status")
async def status():
    # Minimal readiness endpoint expected by the platform
    return {"status": "ok"}


@app.get("/version")
async def version():
    return {"name": "boogeyman-agent", "version": "0.1.0"}


# ----------------------------
# WebSocket endpoint (Cartesia)
# ----------------------------

def _phase_to_stage(phase: str) -> str:
    try:
        return BoogeymanReasoningNode.PHASE_TO_STAGE.get(phase, "intro")
    except Exception:
        return "intro"


@app.websocket("/ws/boogeyman")
async def boogeyman_ws(websocket: WebSocket):
    # Token verification (Authorization: Bearer <token> or ?token=)
    token = websocket.query_params.get("token")
    if not token:
        auth_header = websocket.headers.get("authorization") or ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1].strip()
    if not token:
        await websocket.close(code=1008)
        return
    try:
        claims = verify_token(token)
    except Exception:
        await websocket.close(code=1008)
        return
    await websocket.accept()

    # Track token expiry for mid-connection refresh
    token_state = {"exp": int(claims.get("exp", 0)) if isinstance(claims, dict) else 0}

    # Instance using simpler non-Line wrapper
    node = BoogeymanNode(model=settings.llm_model)
    tts = CartesiaTTS(
        api_key=settings.cartesia_api_key,
        url=settings.cartesia_tts_url,
        voice_id=settings.cartesia_voice_id,
        model=settings.cartesia_model,
        timeout=settings.connection_timeout,
        max_retries=settings.max_retries,
        retry_delay=settings.retry_delay,
    )

    # Bring over override maps from reasoning node
    HINT_TO_PHASE = BoogeymanReasoningNode.HINT_TO_PHASE
    PARENT_OVERRIDE_TO_PHASE = BoogeymanReasoningNode.PARENT_OVERRIDE_TO_PHASE

    # Track agent speaking state to prevent interruptions
    agent_speaking = {"active": False, "queued_user_input": None}

    # Idle timeout for WS: prefer header override, else env default
    try:
        WS_IDLE_SECONDS = float(
            websocket.headers.get("x-idle-timeout")
            or os.getenv("WS_IDLE_TIMEOUT_SECONDS", "25")
        )
    except Exception:
        WS_IDLE_SECONDS = 25.0
    last_activity = {"t": asyncio.get_event_loop().time()}

    def _touch():
        last_activity["t"] = asyncio.get_event_loop().time()

    async def _ws_idle_watchdog():
        try:
            while True:
                await asyncio.sleep(2)
                now = asyncio.get_event_loop().time()
                if now - last_activity["t"] > WS_IDLE_SECONDS:
                    try:
                        await websocket.send_json({
                            "type": "agent_response",
                            "text": "I have other kids to check on. I’ll come back.",
                            "control": {"phase": "goodbye", "classification": "resolved_positive", "intensity": 0, "end_call": True},
                        })
                    except Exception:
                        pass
                    try:
                        await websocket.close(code=1000)
                    finally:
                        return
        except Exception:
            return

    idle_task = asyncio.create_task(_ws_idle_watchdog())

    async def _token_watchdog():
        # Close the socket if token expires and is not refreshed
        # Grace period allows client to refresh slightly after expiry
        GRACE_SECONDS = 15
        try:
            while True:
                await asyncio.sleep(5)
                exp = int(token_state.get("exp", 0) or 0)
                if exp:
                    now = int(time.time())
                    if now > exp + GRACE_SECONDS:
                        try:
                            await websocket.send_json({
                                "type": "error",
                                "error": "token_expired",
                                "message": "WebSocket token expired; please reconnect or refresh token.",
                            })
                        except Exception:
                            pass
                        try:
                            await websocket.close(code=1008)
                        finally:
                            return
        except Exception:
            return

    token_task = asyncio.create_task(_token_watchdog())

    # Try to get metadata first for a personalized intro (wait up to 1s)
    have_intro = False
    pending_meta = None
    message = None
    # Lightweight setup wizard to collect scenario/consequence/fear/tone/child_name
    config_mode = False
    config_step = 0
    config_data = {"scenario": None, "consequence": None, "is_high_fear": None, "tone": None, "child_name": None}

    async def _ask_next_question():
        prompts = [
            "Quick setup: what's the scenario? Choose bedtime, not listening, or doing something messy.",
            "Pick a consequence: Choose get you, take something away, or monster's choice.",
            "Fear level: Choose high or low.",
            "Tone preference: Choose escalate or de escalate.",
            "What's the child's name?",
        ]
        if config_step < len(prompts):
            await websocket.send_json({"type": "agent_response", "text": prompts[config_step], "control": {"phase": "introduction", "classification": "ongoing", "intensity": 0}})
    try:
        first = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
        if isinstance(first, dict) and str(first.get("type") or "").lower() in {"init", "metadata"}:
            pending_meta = first
        else:
            # Not metadata: start a quick setup wizard before speaking to the child
            await websocket.send_json({"type": "ack", "ok": True})
            config_mode = True
            config_step = 0
            await _ask_next_question()
            message = first
    except Exception:
        message = None

    try:
        while True:
            if message is None:
                message = await websocket.receive_json()
            mtype = str(message.get("type") or "").lower()

            # Token refresh for already-open WebSocket
            if mtype in {"auth", "token_refresh", "refresh_token", "refresh"}:
                new_token = (
                    message.get("token")
                    or message.get("access_token")
                    or message.get("bearer")
                    or message.get("value")
                )
                if not new_token:
                    await websocket.send_json({"type": "ack", "ok": False, "error": "missing_token"})
                    _touch()
                    message = None
                    continue
                try:
                    new_claims = verify_token(str(new_token))
                    token_state["exp"] = int(new_claims.get("exp", token_state.get("exp", 0)))
                    await websocket.send_json({"type": "ack", "ok": True, "refreshed": True})
                except Exception:
                    await websocket.send_json({"type": "ack", "ok": False, "error": "invalid_token"})
                _touch()
                message = None
                continue

            # Update plan/metadata
            if mtype in {"init", "metadata"}:
                meta = (pending_meta.get("metadata") if pending_meta else None) or message.get("metadata") or message.get("data") or {}
                scenario = meta.get("scenario") or meta.get("scenario_id")
                consequence = meta.get("consequence") or meta.get("consequence_id")
                is_high_fear = meta.get("is_high_fear") or meta.get("fear_level")
                is_callback = bool(meta.get("is_callback"))
                child_name = meta.get("child_name") or meta.get("name")

                plan = build_boogeyman_plan(
                    scenario=str(scenario) if scenario is not None else None,
                    consequence=str(consequence) if consequence is not None else None,
                    is_high_fear=bool(str(is_high_fear).lower() in {"1", "true", "yes", "high"}) if is_high_fear is not None else False,
                    is_callback=is_callback,
                )
                node.controller.set_script_plan(plan)
                # If no intro yet and we have metadata, send a personalized intro
                if not have_intro:
                    gen = node.generate({
                        "participant_text": "",
                        "participant_role": "system",
                        "extra_context": {"child_name": child_name} if child_name else {},
                    })
                    try:
                        turn = next(gen)
                        if hasattr(turn, "phase"):
                            stage = _phase_to_stage(turn.phase)
                            await websocket.send_json({
                                "type": "stage_signal",
                                "data": {"phase": turn.phase, "classification": turn.classification, "stage": stage},
                            })
                        send_audio = (websocket.query_params.get("audio") or "").lower() in {"1","true","yes"}
                        send_audio = send_audio or ((websocket.headers.get("x-audio") or "").lower() in {"1","true","yes"})
                        
                        # Mark agent as speaking
                        agent_speaking["active"] = True
                        
                        # Send text responses immediately for faster perceived response
                        for sentence in getattr(turn, "sentences", []):
                            await websocket.send_json({
                                "type": "agent_response",
                                "text": sentence,
                                "control": getattr(turn, "control", {}),
                            })
                        
                        # Then generate and send audio if requested
                        if send_audio and tts.is_configured():
                            for sentence in getattr(turn, "sentences", []):
                                audio = tts.synthesize(sentence)
                                if audio:
                                    await websocket.send_json({
                                        "type": "audio_chunk",
                                        "content_type": "audio/wav",
                                        "b64": CartesiaTTS.to_b64(audio),
                                        "text": sentence,
                                        "control": getattr(turn, "control", {}),
                                    })
                        
                        agent_speaking["active"] = False
                        have_intro = True
                    except StopIteration:
                        pass
                await websocket.send_json({"type": "ack", "ok": True})
                _touch()
                message = None
                continue

            # UI hint -> force next phase
            if mtype in {"ui_hint", "hint"}:
                hint = (message.get("hint") or message.get("value") or "").strip().lower()
                mapping = HINT_TO_PHASE.get(hint)
                if mapping:
                    phase, classification = mapping
                    node.controller.queue_phase_override(phase, classification=classification)
                await websocket.send_json({"type": "ack", "ok": True})
                _touch()
                message = None
                continue

            # Parent override
            if mtype in {"parent_override", "override"}:
                cmd = (message.get("command") or message.get("value") or "").strip().lower()
                mapping = PARENT_OVERRIDE_TO_PHASE.get(cmd)
                if mapping:
                    phase, classification = mapping
                    node.controller.queue_phase_override(phase, classification=classification)
                if cmd in {"end", "end_call", "wrap_up"}:
                    # Send final message and close
                    await websocket.send_json({
                        "type": "agent_response",
                        "text": "Thanks. I’ll check back later.",
                        "control": {"phase": "goodbye", "classification": "resolved_positive", "intensity": 0, "end_call": True},
                    })
                    await websocket.close(code=1000)
                    return
                await websocket.send_json({"type": "ack", "ok": True})
                _touch()
                message = None
                continue

            # Wizard: collect initial configuration before normal flow
            if config_mode and mtype in {"transcript", "text", "user", "user_message", "user-message"}:
                raw = str(message.get("text") or message.get("content") or "").strip().lower()
                # Handle empty input during config - re-prompt instead of failing silently
                if not raw or len(raw) < 2:
                    prompts = [
                        "I didn't catch that. Quick setup: what's the scenario? Choose bedtime, not listening, or doing something messy.",
                        "I didn't hear you. Pick a consequence: Choose getYou, takeSomethingAway, or monsterChoice.",
                        "Sorry, I missed that. Fear level: Choose high or low.",
                        "Couldn't hear you. Tone preference: Choose escalate or de-escalate.",
                        "What's the child's name? ",
                    ]
                    await websocket.send_json({"type": "agent_response", "text": prompts[min(config_step, len(prompts)-1)], "control": {"phase": "introduction", "classification": "ongoing", "intensity": 0}})
                    _touch(); message=None; continue
                # Step 0: scenario - expanded keyword matching for better ASR recognition
                if config_step == 0:
                    # Comprehensive keyword lists for ASR variations
                    val = None
                    if any(kw in raw for kw in ["bedtime", "bed time", "bed", "sleep", "night", "bad time", "bet time", "dead time", "best time", "beth time", "sleeping", "sleepy", "nap", "rest"]):
                        val = "bedtime"
                    elif any(kw in raw for kw in ["not listening", "not listen", "listen", "ignoring", "won't listen", "doesn't listen", "disobey", "not lessening", "knot listening", "nat listening", "defiant", "refusing", "rebel"]):
                        val = "not_listening"
                    elif any(kw in raw for kw in ["mess", "messy", "clean", "dirty", "room", "toys", "mass", "mett", "matt", "mat", "matte", "math", "mas", "mace", "mesh", "mash", "match", "messed"]):
                        val = "mess"
                    if not val:
                        await websocket.send_json({"type": "agent_response", "text": "Please choose: bedtime, not listening, or doing something messy.", "control": {"phase": "introduction", "classification": "ongoing", "intensity": 0}})
                        _touch(); message=None; continue
                    config_data["scenario"] = val; config_step += 1; await _ask_next_question(); _touch(); message=None; continue
                # Step 1: consequence - improved keyword matching with phonetic variations
                if config_step == 1:
                    val = None
                    if any(kw in raw for kw in ["monster", "scare", "choice", "decide", "munster", "monster pick", "scary choice", "monster way", "monster voice"]):
                        val = "monsterChoice"
                    elif any(kw in raw for kw in ["take", "away", "toy", "something", "remove", "confiscate", "take it", "cake something", "tape something", "privilege"]):
                        val = "takeSomethingAway"
                    elif any(kw in raw for kw in ["get", "come", "you", "git", "getcha", "get ya", "catch", "grab", "find"]):
                        val = "getYou"
                    if not val:
                        await websocket.send_json({"type": "agent_response", "text": "Choose: getYou, monsterChoice, or takeSomethingAway.", "control": {"phase": "introduction", "classification": "ongoing", "intensity": 0}})
                        _touch(); message=None; continue
                    config_data["consequence"] = val; config_step += 1; await _ask_next_question(); _touch(); message=None; continue
                # Step 2: fear level - expanded recognition for ASR variations
                if config_step == 2:
                    is_high = any(kw in raw for kw in ["high", "hi", "hight", "height", "hy", "hai", "hey", "h", "1", "yes", "scary", "very", "really scary", "very scary"])
                    is_low = any(kw in raw for kw in ["low", "lo", "loo", "lowe", "slow", "l", "0", "2", "no", "gentle", "not scary", "little", "less", "minimum", "fine", "normal", "okay"])
                    if not (is_high or is_low):
                        await websocket.send_json({"type": "agent_response", "text": "Choose 'high' or 'low'.", "control": {"phase": "introduction", "classification": "ongoing", "intensity": 0}})
                        _touch(); message=None; continue
                    config_data["is_high_fear"] = bool(is_high and not is_low)
                    config_step += 1; await _ask_next_question(); _touch(); message=None; continue
                # Step 3: tone preference - improved matching with priority for de-escalate
                if config_step == 3:
                    tone = None
                    # Check for numeric options first
                    if raw in {"2", "two"}:
                        tone = "de_escalate"
                    elif raw in {"1", "one"}:
                        tone = "escalate"
                    # Check for "de" prefix (high priority)
                    elif "de " in raw or "de-" in raw or "de_" in raw or raw.startswith("de") or "dee" in raw:
                        tone = "de_escalate"
                    # Check other de-escalate keywords with phonetic variations
                    elif any(kw in raw for kw in ["calm", "praise", "gentle", "softer", "lighter", "slow", "reduce", "less", "chill", "relax", "mellow", "ease", "dial back", "dial down", "be nice", "be gentle"]):
                        tone = "de_escalate"
                    # Only match escalate if no de-escalate indicators, with phonetic variations
                    elif any(kw in raw for kw in ["escal", "scale", "up", "more", "speed", "escalade", "ramp", "intensify", "turn up", "amp up", "scarier"]):
                        tone = "escalate"
                    if not tone:
                        await websocket.send_json({"type": "agent_response", "text": "Choose 'escalate' or 'de-escalate'.", "control": {"phase": "introduction", "classification": "ongoing", "intensity": 0}})
                        _touch(); message=None; continue
                    config_data["tone"] = tone; config_step += 1; await _ask_next_question(); _touch(); message=None; continue
                # Step 4: child name
                if config_step == 4:
                    name = (message.get("text") or message.get("content") or "").strip()
                    config_data["child_name"] = name if name else None
                    # Apply configuration
                    config_mode = False
                    # Rebuild plan per answers
                    plan = build_boogeyman_plan(
                        scenario=config_data["scenario"],
                        consequence=config_data["consequence"],
                        is_high_fear=bool(config_data["is_high_fear"]),
                        is_callback=False,
                    )
                    node.controller.set_script_plan(plan)
                    # Apply tone preference as phase override for next turn
                    if config_data.get("tone") == "escalate":
                        node.controller.queue_phase_override("escalate", classification="escalation")
                    elif config_data.get("tone") == "de_escalate":
                        node.controller.queue_phase_override("de_escalate", classification="praise")
                    # Personalized intro
                    gen = node.generate({
                        "participant_text": "",
                        "participant_role": "system",
                        "extra_context": {"child_name": config_data.get("child_name")} if config_data.get("child_name") else {},
                    })
                    try:
                        turn = next(gen)
                        if hasattr(turn, "phase"):
                            stage = _phase_to_stage(turn.phase)
                            await websocket.send_json({
                                "type": "stage_signal",
                                "data": {"phase": turn.phase, "classification": turn.classification, "stage": stage},
                            })
                        send_audio = (websocket.query_params.get("audio") or "").lower() in {"1","true","yes"}
                        send_audio = send_audio or ((websocket.headers.get("x-audio") or "").lower() in {"1","true","yes"})
                        
                        agent_speaking["active"] = True
                        
                        for sentence in getattr(turn, "sentences", []):
                            await websocket.send_json({"type": "agent_response", "text": sentence, "control": getattr(turn, "control", {})})
                        if send_audio and tts.is_configured():
                            for sentence in getattr(turn, "sentences", []):
                                audio = tts.synthesize(sentence)
                                if audio:
                                    await websocket.send_json({"type": "audio_chunk", "content_type": "audio/wav", "b64": CartesiaTTS.to_b64(audio), "text": sentence, "control": getattr(turn, "control", {})})
                        
                        agent_speaking["active"] = False
                        have_intro = True
                    except StopIteration:
                        pass
                    _touch(); message=None; continue

            # Transcript -> generate
            if mtype in {"transcript", "text", "user", "user_message", "user-message"}:
                text = message.get("text") or message.get("content") or ""
                
                # If agent is currently speaking, queue the user input instead of interrupting
                if agent_speaking["active"]:
                    logger.info("Agent speaking - queuing user input: %s", text)
                    agent_speaking["queued_user_input"] = text
                    _touch()
                    message = None
                    continue
                
                gen = node.generate({
                    "participant_text": str(text),
                    "participant_role": "parent",
                    "extra_context": message.get("extra_context") or {},
                })
                try:
                    turn = next(gen)
                except StopIteration:
                    await websocket.send_json({"type": "agent_response", "text": "I'm here to help."})
                    _touch()
                    message = None
                    continue

                # Stage signal
                try:
                    stage = _phase_to_stage(turn.phase)
                    await websocket.send_json({
                        "type": "stage_signal",
                        "data": {"phase": turn.phase, "classification": turn.classification, "stage": stage},
                    })
                except Exception:
                    pass

                # Mark agent as speaking
                agent_speaking["active"] = True
                
                # Sentences (with optional audio chunks)
                send_audio = (websocket.query_params.get("audio") or "").lower() in {"1","true","yes"}
                send_audio = send_audio or ((websocket.headers.get("x-audio") or "").lower() in {"1","true","yes"})
                
                # Send text responses immediately for faster perceived response
                for sentence in getattr(turn, "sentences", []):
                    await websocket.send_json({
                        "type": "agent_response",
                        "text": sentence,
                        "control": getattr(turn, "control", {}),
                    })
                
                # Then generate and send audio if requested
                if send_audio and tts.is_configured():
                    for sentence in getattr(turn, "sentences", []):
                        audio = tts.synthesize(sentence)
                        if audio:
                            await websocket.send_json({
                                "type": "audio_chunk",
                                "content_type": "audio/wav",
                                "b64": CartesiaTTS.to_b64(audio),
                                "text": sentence,
                                "control": getattr(turn, "control", {}),
                            })
                
                # Agent finished speaking
                agent_speaking["active"] = False
                
                # Process any queued user input
                if agent_speaking["queued_user_input"]:
                    queued_text = agent_speaking["queued_user_input"]
                    agent_speaking["queued_user_input"] = None
                    logger.info("Processing queued user input: %s", queued_text)
                    
                    # Generate response to queued input
                    gen = node.generate({
                        "participant_text": str(queued_text),
                        "participant_role": "parent",
                        "extra_context": {},
                    })
                    try:
                        queued_turn = next(gen)
                        
                        # Send stage signal
                        try:
                            stage = _phase_to_stage(queued_turn.phase)
                            await websocket.send_json({
                                "type": "stage_signal",
                                "data": {"phase": queued_turn.phase, "classification": queued_turn.classification, "stage": stage},
                            })
                        except Exception:
                            pass
                        
                        # Mark agent as speaking again
                        agent_speaking["active"] = True
                        
                        # Send responses
                        for sentence in getattr(queued_turn, "sentences", []):
                            await websocket.send_json({
                                "type": "agent_response",
                                "text": sentence,
                                "control": getattr(queued_turn, "control", {}),
                            })
                        
                        if send_audio and tts.is_configured():
                            for sentence in getattr(queued_turn, "sentences", []):
                                audio = tts.synthesize(sentence)
                                if audio:
                                    await websocket.send_json({
                                        "type": "audio_chunk",
                                        "content_type": "audio/wav",
                                        "b64": CartesiaTTS.to_b64(audio),
                                        "text": sentence,
                                        "control": getattr(queued_turn, "control", {}),
                                    })
                        
                        agent_speaking["active"] = False
                    except StopIteration:
                        agent_speaking["active"] = False
                
                _touch()
                message = None
                continue

            # Unknown type -> ignore gracefully
            await websocket.send_json({"type": "ack", "ok": False, "error": "unknown_type"})
            message = None

    except WebSocketDisconnect:
        logger.info("Boogeyman WS disconnected")
    except Exception as exc:
        logger.exception("Boogeyman WS error: %s", exc)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        try:
            idle_task.cancel()
        except Exception:
            pass
        try:
            token_task.cancel()
        except Exception:
            pass
        try:
            await asyncio.gather(idle_task, token_task, return_exceptions=True)
        except Exception:
            pass
