"""
Relationship extraction module for biogather.

This module provides functionality for extracting relationships between biomedical
entities, including entity pairs, events, and their roles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import spacy
from transformers import pipeline

from biogather.utils import is_scispacy_available

if TYPE_CHECKING:
    from spacy.language import Language


class RelationshipExtractor:
    """Class for biomedical relationship extraction tasks."""

    BIOMEDICAL_MODELS: ClassVar[dict[str, str]] = {
        "base": "en_core_sci_sm",
        "large": "en_core_sci_lg",
    }

    def __init__(
        self,
        model: str = "en_core_web_sm",
        use_biomedical: bool = False,
        transformer_model: str | None = None,
    ) -> None:
        """
        Initialize the RelationshipExtractor.

        Args:
            model: The spaCy model to use. If use_biomedical is True and no model
                  is specified, defaults to 'en_core_sci_sm'.
            use_biomedical: Whether to use biomedical models. Requires scispacy.
            transformer_model: Optional transformer model for enhanced extraction.

        """
        self.is_biomedical = use_biomedical

        if use_biomedical:
            if not is_scispacy_available():
                msg = "Biomedical processing requires scispacy. Install it with: pip install biogather[bio]"
                raise ImportError(msg)
            if model == "en_core_web_sm":
                model = self.BIOMEDICAL_MODELS["base"]

        try:
            self.nlp: Language = spacy.load(model)
        except OSError as e:
            model_type = "biomedical" if use_biomedical else "basic"
            msg = f"Model '{model}' not found. For {model_type} models, run: python -m spacy download {model}"
            raise OSError(msg) from e

        # Initialize transformer pipeline if specified
        self.transformer = None
        if transformer_model:
            self.transformer = pipeline(
                "ner",
                model=transformer_model,
                tokenizer=transformer_model,
                aggregation_strategy="simple",
            )

    def get_default_patterns(self) -> list[dict[str, str | set[str]]]:
        """
        Get default relationship patterns based on model type.

        Returns:
            List of default patterns for relationship extraction.

        """
        if self.is_biomedical:
            return [
                {
                    "type": "INTERACTS_WITH",
                    "verbs": {
                        "interact",
                        "bind",
                        "inhibit",
                        "activate",
                        "phosphorylate",
                        "methylate",
                        "acetylate",
                        "ubiquitinate",
                        "regulate",
                    },
                },
                {
                    "type": "REGULATES",
                    "verbs": {
                        "regulate",
                        "control",
                        "modulate",
                        "affect",
                        "influence",
                        "mediate",
                        "induce",
                        "suppress",
                        "enhance",
                    },
                },
                {
                    "type": "TREATS",
                    "verbs": {
                        "treat",
                        "cure",
                        "prevent",
                        "alleviate",
                        "improve",
                        "reduce",
                        "increase",
                        "decrease",
                        "modify",
                    },
                },
            ]
        return [
            {
                "type": "INTERACTS_WITH",
                "verbs": {"interact", "bind", "inhibit", "activate"},
            },
            {
                "type": "REGULATES",
                "verbs": {"regulate", "control", "modulate", "affect"},
            },
            {
                "type": "TREATS",
                "verbs": {"treat", "cure", "prevent", "alleviate"},
            },
        ]

    def extract_entity_pairs(
        self,
        text: str,
        max_distance: int = 5,
        require_verb: bool = True,
    ) -> list[dict[str, str | tuple[int, int]]]:
        """
        Extract pairs of entities that might be related.

        Args:
            text: The input text to analyze.
            max_distance: Maximum token distance between entities.
            require_verb: Whether to require a verb between entities.

        Returns:
            List of potential entity pairs with metadata.

        """
        doc = self.nlp(text)
        pairs = []

        for sent in doc.sents:
            entities = list(sent.ents)

            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1 :]:
                    # Check token distance between entities
                    distance = e2.start - e1.end
                    if distance > max_distance:
                        continue

                    # Get text between entities
                    between_tokens = doc[e1.end : e2.start]

                    # Check if there's a verb between entities if required
                    has_verb = not require_verb or any(t.pos_ == "VERB" for t in between_tokens)

                    if has_verb:
                        pairs.append(
                            {
                                "entity1": {
                                    "text": e1.text,
                                    "type": e1.label_,
                                    "span": (e1.start, e1.end),
                                },
                                "entity2": {
                                    "text": e2.text,
                                    "type": e2.label_,
                                    "span": (e2.start, e2.end),
                                },
                                "between": between_tokens.text,
                            },
                        )

        return pairs

    def extract_relationships(
        self,
        text: str,
        patterns: list[dict[str, str]] | None = None,
        include_negation: bool = True,
    ) -> list[dict[str, str | dict[str, str]]]:
        """
        Extract relationships between entities using patterns and rules.

        Args:
            text: The input text to analyze.
            patterns: Custom patterns for relationship extraction.
            include_negation: Whether to detect negated relationships.

        Returns:
            List of extracted relationships with metadata.

        """
        doc = self.nlp(text)
        relationships = []

        # Default patterns for common biomedical relationships
        if not patterns:
            patterns = self.get_default_patterns()

        for sent in doc.sents:
            # Get entities in the sentence
            entities = list(sent.ents)

            # Look for relationship patterns between entities
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1 :]:
                    # Get the text between the entities
                    if e1.end_char < e2.start_char:
                        between_text = doc[e1.end : e2.start].text.lower()
                    else:
                        between_text = doc[e2.end : e1.start].text.lower()

                    # Check for relationship patterns
                    relationships.extend(
                        [
                            {
                                "type": pattern["type"],
                                "entity1": {
                                    "text": e1.text,
                                    "type": e1.label_,
                                },
                                "entity2": {
                                    "text": e2.text,
                                    "type": e2.label_,
                                },
                                "evidence": sent.text,
                            }
                            for pattern in patterns
                            if any(verb in between_text for verb in pattern["verbs"])
                        ],
                    )

        return relationships

    def extract_events(
        self,
        text: str,
        include_triggers: bool = True,
        include_arguments: bool = True,
    ) -> list[dict[str, str | list[dict[str, str]]]]:
        """
        Extract biomedical events from text.

        Args:
            text: The input text to analyze.
            include_triggers: Whether to include event triggers.
            include_arguments: Whether to include event arguments.

        Returns:
            List of extracted events with metadata.

        """
        doc = self.nlp(text)
        events = []

        # Default biomedical event triggers
        default_triggers = {
            "regulation": {"regulate", "control", "modulate"},
            "activation": {"activate", "induce", "stimulate"},
            "inhibition": {"inhibit", "suppress", "block"},
            "binding": {"bind", "interact", "complex"},
            "expression": {"express", "transcribe", "translate"},
        }

        for sent in doc.sents:
            for token in sent:
                # Check if token is a potential event trigger
                is_trigger = False
                trigger_type = None

                for event_type, triggers in default_triggers.items():
                    if token.lemma_.lower() in triggers:
                        is_trigger = True
                        trigger_type = event_type
                        break

                if is_trigger:
                    event = {
                        "type": trigger_type,
                        "trigger": token.text,
                        "arguments": [],
                    }

                    # Find event arguments
                    for ent in sent.ents:
                        role = "Theme" if ent.start < token.i else "Agent"

                        event["arguments"].append(
                            {
                                "text": ent.text,
                                "type": ent.label_,
                                "role": role,
                            },
                        )

                    events.append(event)

        return events

    def determine_argument_role(
        self,
        text: str,
        entity: str,
        trigger: str,
    ) -> str:
        """
        Determine the role of an entity in relation to an event trigger.

        Args:
            text: The input text containing the entity and trigger.
            entity: The entity text to analyze.
            trigger: The trigger word (usually a verb).

        Returns:
            The determined role (e.g., "Agent", "Theme", "Instrument").

        """
        doc = self.nlp(text)

        # Find the entity and trigger spans
        entity_tokens = None
        trigger_token = None

        for token in doc:
            if token.text == trigger:
                trigger_token = token

        for ent in doc.ents:
            if ent.text == entity:
                entity_tokens = ent
                break

        if not entity_tokens or not trigger_token:
            return "UNKNOWN"

        # Check dependency path
        for token in entity_tokens:
            if token.dep_ in ("nsubj", "nsubjpass"):
                return "Agent"
            if token.dep_ in ("dobj", "pobj"):
                return "Theme"
            if token.dep_ in ("prep", "agent"):
                return "Instrument"
        return "Theme"  # Default to Theme if no specific role is found
