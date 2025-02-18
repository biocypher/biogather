"""Provide top level symbols."""

"""Biogather: A library for biomedical text processing and information extraction.

This library provides tools for extracting structured information from biomedical text,
including named entity recognition, relationship extraction, and event detection.
"""

from biogather.preprocessing import TextPreprocessor
from biogather.linguistic_annotation import LinguisticAnnotator
from biogather.entity_extraction import EntityExtractor
from biogather.relationship_extraction import RelationshipExtractor

__version__ = "0.1.0"

__all__ = [
    "EntityExtractor",
    "LinguisticAnnotator",
    "RelationshipExtractor",
    "TextPreprocessor",
]
