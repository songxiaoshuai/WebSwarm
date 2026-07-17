"""WebSwarm verb-agent package.

Verb agents are specialized capabilities available for root-agent delegation: atom handles
atomic queries, deep performs deep hypothesis search and independent verification, wide
performs fanout decomposition, and entity_collect enumerates entity sets with high recall.
"""
from .registry import VERB_REGISTRY, VALID_VERBS, run_verb

__all__ = ["VERB_REGISTRY", "VALID_VERBS", "run_verb"]
