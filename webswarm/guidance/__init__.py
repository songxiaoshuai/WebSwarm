"""Entry point for the WebSwarm guidance subsystem.

guidance manages runtime assistance signals: web_probing records webpage-information
topology probes, while subtask_experience records transferable execution experience
extracted from scout subtasks.
"""

from .event_store import GuidanceEventStore
from .subtask_experience import extract_subtask_experience

__all__ = [
    "GuidanceEventStore",
    "extract_subtask_experience",
]
