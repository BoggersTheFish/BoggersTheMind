"""Entities: ingestor, inference, simulation, consolidation, insight."""

from .ingestor import ingest_wikipedia
from .inference import infer
from .simulation import simulate
from .consolidation import consolidate
from .insight import write_insight

__all__ = [
    "ingest_wikipedia",
    "infer",
    "simulate",
    "consolidate",
    "write_insight",
]
