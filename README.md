# Boogeyman Voice Agent

A small Line-based voice agent that role‑plays a "Playful Boogeyman" using a scripted plan and optional LLM adaptation.

## Features
- Scripted conversation arc with phases: introduction → situation → consequence → escalate/de-escalate → result.
- Optional adaptation with an OpenAI chat model. Falls back to scripted lines if no API key.
- Basic guardrails: output sanitization (forbidden content, softened threats) and intensity clamping by age bucket from `guardrails/policy.yaml`.
- Stage telemetry events for clients (`StageSignal`).

## Requirements
- Python 3.10+
- Dependencies:
  - `openai` (optional, for LLM adaptation)
  - `loguru`
  - `PyYAML`
  - `python-dotenv`
  - The `line` runtime framework (internal/external dependency; not provided here)

Install dependencies:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Environment
- Copy `.env.example` to `.env` and set values:
  - `OPENAI_API_KEY` (optional)
  - `CARTESIA_API_KEY` (required to mint WS tokens)
  - `TOKEN_SECRET` (required for token signing)
  - `LLM_TIMEOUT_SECONDS` (default 5)
  - `IDLE_TIMEOUT_SECONDS` (default 25)
  - `WS_IDLE_TIMEOUT_SECONDS` (default 25)
  - `CARTESIA_TTS_URL` (optional HTTP TTS endpoint)
  - `CARTESIA_VOICE_ID` (optional default voice)
  - `CARTESIA_WS_URL` (optional WS TTS endpoint)
  - `CARTESIA_TOKEN_URL` (optional token exchange URL)

## Configuration
- Optional: set `OPENAI_API_KEY` to enable model-based adaptation.
- Policy: `line_agent/guardrails/policy.yaml` defines tone, forbidden content tags, and `max_intensity_by_age` mapping (e.g., `"4-6": 2`, `"7-9": 3`).
- Metadata: The call metadata can include `age` or `age_range` (e.g., `"4-6"`) to apply intensity ceilings.

## Running
Using module execution (if `line_agent` is importable):

```powershell
$env:OPENAI_API_KEY="sk-..."   # optional
python -m line_agent.app
```

Or run directly by path:

```powershell
python c:\Users\Sheyan\Desktop\line_agent_new\line_agent\app.py
```

Note: The app depends on the `line` framework and a media harness; ensure your environment provides these.

### WebSocket (Cartesia-style)
- Endpoint: `ws://<host>:<port>/ws/boogeyman`
- Auth: obtain a short-lived token from `POST /token` and pass it as `Authorization: Bearer <token>` or `?token=` query param.
- Message types (JSON):
  - `{"type":"init","metadata": {"scenario":"bedtime","consequence":"getyou","is_callback":false,"age":6}}`
  - `{"type":"transcript","text":"We’re not getting ready for bed."}`
  - `{"type":"ui_hint","hint":"escalate|praise|result_positive|result_negative"}`
  - `{"type":"parent_override","command":"lighter|too_scary|end|wrap_up|end_call"}`
- Server sends back:
  - `{"type":"stage_signal","data": {"phase":"...","classification":"...","stage":"..."}}`
  - `{"type":"agent_response","text":"...","control": {"phase":"...","classification":"...","intensity":0}}`

Run with Uvicorn:

```powershell
uvicorn line_agent.app:app --host 0.0.0.0 --port 8000
```

#### Optional audio over WebSocket
- Set `CARTESIA_TTS_URL` and `CARTESIA_API_KEY` (and optionally `CARTESIA_VOICE_ID`).
- Connect with `?audio=true` (or header `x-audio: true`).
- Server will include `audio_chunk` messages: `{ "type": "audio_chunk", "content_type": "audio/wav", "b64": "...", "text": "...", "control": { ... } }`.
- If synthesis fails, it falls back to `agent_response` (text-only).

### Token issuance
- Set `CARTESIA_API_KEY` on the server; clients call `POST /token` with header `x-api-key: <CARTESIA_API_KEY>`.
- Optional body: `{ "session_id": "...", "agent_id": "...", "ttl_seconds": 1800 }`
- Response: `{ "token": "...", "expires": 173... , "session_id": "..." }`

Example (PowerShell):

```powershell
$env:CARTESIA_API_KEY="your-admin-api-key"
Invoke-RestMethod -Method Post -Uri http://localhost:8000/token -Headers @{"x-api-key"=$env:CARTESIA_API_KEY} -Body '{}' -ContentType 'application/json'
```

Connect WS with token:

```powershell
# Using query param
wscat -c "ws://localhost:8000/ws/boogeyman?token=<TOKEN>"

# Or with Authorization header (example pseudo)
# Some WS clients support: --header "Authorization: Bearer <TOKEN>"
```

### Deploying with Cartesia
- Use your Cartesia CLI to `cartesia deploy` and obtain an `agent_id`.
- Generate tokens per session using `/token` (include `agent_id` in the body if you want it embedded in claims).
- Your app (iOS or server) then connects over WebSockets with the Bearer token for a real-time conversation.

## Notes on Guardrails
- Output text is sanitized against simple forbidden terms and softened to avoid threats.
- Intensity in the CONTROL block is clamped to the policy maximum for the child age bucket when available; otherwise a conservative default is used.

## Development Tips
- Core logic: `line_agent/nodes/controller.py`
- Script plan: `line_agent/nodes/boogeyman_scripts.py`
- Reasoning node (Line integration): `line_agent/nodes/boogeyman_node.py`
- Guardrails: `line_agent/guardrails/enforcer.py`
- Prompts: `line_agent/prompts/`
