"""Splinter Memory — the public facade of the context-curation system.

``splinter/`` holds the flat packages (``cortex``, ``sieve``, ``membrane``,
``retention``, ``focal``, ``backend``, ``auditor``, ``logs``) that the bench and
harness import by name. This module is the *one import* for system integrators:

    from splinter import Splinter, SplinterConfig, UltraSmallDrone, LMStudioBackend

Nothing in ``splinter/`` imports from ``hivebench/`` or ``harness/`` — the system
is self-contained and portable into other projects.
"""

from backend.lmstudio import LMStudioBackend
from backend.openai_compat import OpenAICompatBackend
from cortex.config import SplinterConfig
from cortex.splinter import Splinter
from retention.store import ContextStore
from sieve.medium import MediumDrone
from sieve.ultra_small import UltraSmallDrone

__all__ = [
    "Splinter",
    "SplinterConfig",
    "UltraSmallDrone",
    "MediumDrone",
    "ContextStore",
    "LMStudioBackend",
    "OpenAICompatBackend",
]