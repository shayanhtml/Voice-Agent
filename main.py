"""Cartesia Line entrypoint

Exposes the FastAPI `app` and starts Uvicorn when executed directly.
The platform expects a uvicorn.run() call in main.py.
"""

import os
import uvicorn
from line_agent.app import app  # FastAPI instance


def _int(name: str, default: int) -> int:
	try:
		return int(os.getenv(name, str(default)))
	except Exception:
		return default


if __name__ == "__main__":
	host = os.getenv("HOST", "0.0.0.0")
	port = _int("PORT", 8000)
	log_level = (os.getenv("LOG_LEVEL", "info") or "info").lower()

	# Start Uvicorn programmatically as Cartesia requires
	uvicorn.run(app, host=host, port=port, log_level=log_level)
