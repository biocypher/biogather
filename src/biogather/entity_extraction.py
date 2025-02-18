"""
Entity extraction module for biogather.

This module provides functionality for extracting and classifying biomedical entities
from text using both general-purpose and domain-specific models.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, ClassVar, Any

import spacy
from spacy.language import Language
from transformers import pipeline

from biogather.utils import is_scispacy_available

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Span


class EntityExtractor:
    """Class for biomedical entity extraction tasks."""

    BIOMEDICAL_MODELS: ClassVar[dict[str, str]] = {
        "ner": "en_core_sci_scibert",
        "linking": "en_core_sci_lg",
        "base": "en_core_sci_sm",
    }

    def __init__(
        self,
        model: str = "en_core_web_sm",
        use_biomedical: bool = False,
        transformer_model: str | None = None,
    ) -> None:
        """
        Initialize the EntityExtractor.

        Args:
            model: The spaCy model to use. If use_biomedical is True and no model
                  is specified, defaults to 'en_core_web_sm'.
            use_biomedical: Whether to use biomedical models. Requires scispacy.
            transformer_model: Optional transformer model for enhanced NER. If None,
                             only spaCy-based NER will be used.

        """
        self.is_biomedical = use_biomedical

        if use_biomedical:
            if not is_scispacy_available():
                msg = (
                    "Biomedical processing requires scispacy. "
                    "Install it with: pip install biogather[bio]"
                )
                raise ImportError(msg)
            if model == "en_core_web_sm":
                model = self.BIOMEDICAL_MODELS["base"]

        try:
            self.nlp: Language = spacy.load(model)
        except OSError as e:
            model_type = "biomedical" if use_biomedical else "basic"
            msg = (
                f"Model '{model}' not found. For {model_type} models, run: "
                f"python -m spacy download {model}"
            )
            raise OSError(msg) from e

        # Initialize transformer pipeline if specified
        self.transformer_ner = None
        if transformer_model:
            self.transformer_ner = pipeline(
                "ner",
                model=transformer_model,
                tokenizer=transformer_model,
                aggregation_strategy="simple",
            )

    def extract_entities(
        self,
        text: str,
        use_transformer: bool = True,
    ) -> list[dict[str, str | float]]:
        """
        Extract named entities from text.

        Args:
            text: Input text to analyze.
            use_transformer: Whether to use transformer model for NER.

        Returns:
            List of entity dictionaries with text, label, and confidence.

        """
        if use_transformer and self.transformer_ner:
            # Use transformer-based NER
            entities = self.transformer_ner(text)
            return [
                {
                    "text": entity["word"],
                    "label": entity["entity_group"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "confidence": entity["score"],
                }
                for entity in entities
            ]

        # Use spaCy NER (biomedical or standard)
        doc = self.nlp(text)
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": None,
            }
            for ent in doc.ents
        ]

    def classify_entity_type(self, text: str, entity: str) -> str:
        """
        Classify the type of a biomedical entity.

        Args:
            text: The context text containing the entity.
            entity: The entity to classify.

        Returns:
            The predicted entity type.

        """
        # Load biomedical linking model if available and needed
        if (
            self.is_biomedical
            and is_scispacy_available()
            and self.nlp.meta["name"] != self.BIOMEDICAL_MODELS["linking"]
        ):
            try:
                linking_nlp = spacy.load(self.BIOMEDICAL_MODELS["linking"])
                return self._classify_with_linking(text, entity, linking_nlp)
            except OSError:
                pass  # Fall back to basic classification

        return self._classify_basic(text, entity)

    def _classify_with_linking(
        self,
        text: str,
        entity: str,
        linking_nlp: Language | None = None,
    ) -> str:
        """Classify entity using biomedical linking."""
        nlp = linking_nlp or self.nlp
        doc = nlp(text)

        for ent in doc.ents:
            if entity.lower() in ent.text.lower():
                return ent.label_

        return "UNKNOWN"

    def _classify_basic(
        self,
        text: str,
        entity: str,
    ) -> str:
        """Classify basic entities using spaCy."""
        doc = self.nlp(text)
        for ent in doc.ents:
            if entity.lower() in ent.text.lower():
                return ent.label_

        return "UNKNOWN"

    def classify_entity_linking(self, text: str, entity: str) -> dict:
        """
        Classify entity using biomedical linking.

        Args:
            text: The context text containing the entity.
            entity: The entity to link.

        Returns:
            Dictionary containing linking information.

        """
        # Load biomedical linking model if available and needed
        if (
            self.is_biomedical
            and is_scispacy_available()
            and self.nlp.meta["name"] != self.BIOMEDICAL_MODELS["linking"]
        ):
            try:
                linking_nlp = spacy.load(self.BIOMEDICAL_MODELS["linking"])
                return self._classify_with_linking(text, entity, linking_nlp)
            except OSError:
                pass  # Fall back to basic classification

        return self._classify_basic(text, entity)

    def classify_entity_basic(self, text: str, entity: str) -> str:
        """
        Classify basic entities using spaCy.

        Args:
            text: The context text containing the entity.
            entity: The entity to classify.

        Returns:
            The predicted entity type.

        """
        doc = self.nlp(text)
        for ent in doc.ents:
            if entity.lower() in ent.text.lower():
                return ent.label_

        return "UNKNOWN"

    def extract_keywords(
        self,
        text: str,
        include_phrases: bool = True,
        min_freq: int = 1,
        custom_patterns: list[list[dict]] | None = None,
    ) -> list[dict]:
        """
        Extract important keywords and phrases.

        Args:
            text: The input text to analyze.
            include_phrases: Whether to include multi-word phrases.
            min_freq: Minimum frequency for inclusion.
            custom_patterns: Custom matcher patterns.

        Returns:
            List of extracted keywords with metadata.

        """
        doc = self.nlp(text)
        keywords = [
            {
                "text": token.text,
                "pos": token.pos_,
                "is_entity": token.ent_type_ != "",
                "importance": 1.0 if token.ent_type_ else 0.5,
            }
            for token in doc
            if len(token.text) >= min_freq
            and not token.is_stop
            and not token.is_punct
            and token.pos_ in ("NOUN", "PROPN", "ADJ")
        ]

        # Include noun phrases if requested
        if include_phrases:
            keywords.extend([
                {
                    "text": chunk.text,
                    "pos": "PHRASE",
                    "is_entity": any(token.ent_type_ for token in chunk),
                    "importance": 1.0 if any(token.ent_type_ for token in chunk) else 0.7,
                }
                for chunk in doc.noun_chunks
                if len(chunk.text.split()) > 1
            ])

        return keywords

    def extract_entity_context(
        self,
        text: str,
        entity: str,
        window_size: int = 3,
        include_pos: bool = True,
        include_deps: bool = True,
    ) -> dict:
        """
        Extract context windows around entity mentions.

        Args:
            text: The input text to analyze.
            entity: The target entity.
            window_size: Number of tokens before/after entity.
            include_pos: Include POS tags in context.
            include_deps: Include dependency info in context.

        Returns:
            Dictionary containing context information.

        """
        doc = self.nlp(text)
        contexts = []

        for sent in doc.sents:
            if entity.lower() in sent.text.lower():
                start = max(0, sent.start_char - window_size)
                end = min(len(text), sent.end_char + window_size)

                contexts.append(
                    {
                        "entity": entity,
                        "sentence": sent.text,
                        "context": text[start:end],
                        "position": (sent.start_char, sent.end_char),
                    },
                )

        return contexts
