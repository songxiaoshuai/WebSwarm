"""WebSwarm package entry point.

Exposes the top-level WebSwarmAgent runner. experiment_config.build_webswarm_config
builds and validates runtime parameters before passing them in as a dict.
"""

from .webswarm_agent import WebSwarmAgent

__all__ = ["WebSwarmAgent"]
