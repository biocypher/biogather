"""
Linguistic annotation module for biogather.

This module provides functionality for linguistic analysis of text, including
part-of-speech tagging, dependency parsing, and feature extraction.
"""
import importlib.util
import warnings
from typing import TYPE_CHECKING, Any

import spacy
from spacy.language import Language

from biogather.utils import is_scispacy_available

if TYPE_CHECKING:
    from spacy.language import Language

def is_scispacy_available() -> bool:
    """Check if scispacy is available."""
    return importlib.util.find_spec("scispacy") is not None

class LinguisticAnnotator:
    """Class for linguistic annotation tasks."""

    def __init__(
        self,
        model: str = "en_core_web_sm",
    ) -> None:
        """
        Initialize the LinguisticAnnotator.

        Args:
            model: The spaCy model to use.

        """
        try:
            self.nlp: Language = spacy.load(model)
        except OSError:
            if model.startswith("en_core_sci_") and not is_scispacy_available():
                warnings.warn(
                    f"Could not load model '{model}'. Falling back to 'en_core_web_sm'. "
                    "For biomedical text processing, install scispacy and its models: "
                    "pip install biogather[scispacy]",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"Could not load model '{model}'. Falling back to 'en_core_web_sm'. "
                    "Make sure you have downloaded the model: python -m spacy download en_core_web_sm",
                    stacklevel=2,
                )
            self.nlp: Language = spacy.load("en_core_web_sm")

    def pos_tag(self, text: str) -> list[tuple[str, str]]:
        """Perform part-of-speech tagging."""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

    def dependency_parse(self, text: str) -> list[dict[str, str]]:
        """Perform dependency parsing."""
        doc = self.nlp(text)
        return [
            {
                "text": token.text,
                "dep": token.dep_,
                "head": token.head.text,
                "head_pos": token.head.pos_,
            }
            for token in doc
        ]

    def extract_noun_phrases(self, text: str) -> list[dict]:
        """
        Extract noun phrases using spaCy's noun chunk detection.

        Args:
            text: The input text to analyze.

        Returns:
            List of noun phrases with metadata.

        """
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def analyze_sentence_structure(self, text: str) -> list[dict]:
        """
        Analyze sentence structure including subject, verb, object.

        Args:
            text: The input text to analyze.

        Returns:
            List of sentence structures with metadata.

        """
        doc = self.nlp(text)
        sentences = []

        for sent in doc.sents:
            structure = {
                "text": sent.text,
                "components": [],
            }

            # Find main verb and its arguments
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    structure["components"].append(
                        {"role": "subject", "text": token.text, "pos": token.pos_},
                    )
                elif (token.dep_ == "ROOT" or token.dep_ == "VERB") and token.pos_ == "VERB":
                    structure["components"].append(
                        {"role": "verb", "text": token.text, "pos": token.pos_},
                    )
                elif token.dep_ in ("dobj", "pobj"):
                    structure["components"].append(
                        {"role": "object", "text": token.text, "pos": token.pos_},
                    )

            # If no verb found through ROOT, try finding main verb through other dependencies
            if not any(comp["role"] == "verb" for comp in structure["components"]):
                for token in sent:
                    if token.pos_ == "VERB":
                        structure["components"].append(
                            {"role": "verb", "text": token.text, "pos": token.pos_},
                        )
                        break

            sentences.append(structure)

        return sentences

    def get_token_features(self, text: str) -> list[dict]:
        """
        Get comprehensive linguistic features for each token.

        Args:
            text: The input text to analyze.

        Returns:
            List of token features with metadata.

        """
        doc = self.nlp(text)
        features = []

        for token in doc:
            token_features = {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "shape": token.shape_,
                "is_alpha": str(token.is_alpha),
                "is_stop": str(token.is_stop),
                "is_punct": str(token.is_punct),
                "like_num": str(token.like_num),
            }
            features.append(token_features)

        return features
