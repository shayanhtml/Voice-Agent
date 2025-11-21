from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from .controller import Classification, Phase


@dataclass
class StageSignal:
    """Event emitted when the agent transitions to a new conversational phase."""

    phase: Phase
    classification: Classification
    stage: str

    def to_log_event(self) -> Dict[str, object]:
        """Format payload for logging bridges."""
        return {
            "event": "StageSignal",
            "metadata": {
                "phase": self.phase,
                "classification": self.classification,
                "stage": self.stage,
            },
        }

    def as_metadata(self) -> Dict[str, object]:
        return asdict(self)
