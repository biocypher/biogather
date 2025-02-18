"""
Text preprocessing module for biogather.

This module provides functionality for basic text preprocessing tasks including
tokenization, sentence splitting, case normalization, stopword removal, and
lemmatization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import nltk
import spacy
from spacy.language import Language

if TYPE_CHECKING:
    from spacy.language import Language

from biogather.utils import is_scispacy_available

def is_scispacy_available() -> bool:
    """Check if scispacy is available."""
    try:
        import scispacy  # noqa: F401
        return True  # noqa: TRY300
    except ImportError:
        return False


class TextPreprocessor:
    """Class for text preprocessing tasks."""

    def __init__(
        self,
        model: str = "en_core_web_sm",
        use_biomedical: bool = False,
        download_nltk: bool = True,
    ) -> None:
        """
        Initialize the TextPreprocessor.

        Args:
            model: The spaCy model to use. If use_biomedical is True and no model
                  is specified, defaults to 'en_core_sci_sm'.
            use_biomedical: Whether to use biomedical models. Requires scispacy.
            download_nltk: Whether to download required NLTK data.

        """
        if use_biomedical and not is_scispacy_available():
            msg = (
                "Biomedical processing requires scispacy. "
                "Install it with: pip install biogather[bio]"
            )
            raise ImportError(msg)

        if use_biomedical and model == "en_core_web_sm":
            model = "en_core_sci_sm"

        if download_nltk:
            nltk.download("punkt")
            nltk.download("stopwords")
            nltk.download("wordnet")

        try:
            self.nlp: Language = spacy.load(model)
        except OSError as e:
            msg = (
                f"Model '{model}' not found. For basic models, run: "
                f"python -m spacy download {model}. "
                "For biomedical models, install scispacy and run: "
                "python -m spacy download en_core_sci_sm"
            )
            raise OSError(msg) from e

        self.stopwords = set(nltk.corpus.stopwords.words("english"))
        self.is_biomedical = use_biomedical

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text using spaCy.

        Args:
            text: The input text to tokenize.

        Returns:
            List of tokens.

        """
        doc = self.nlp(text)
        return [token.text for token in doc]

    def split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: The input text to split.

        Returns:
            List of sentences.

        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def normalize_case(
        self,
        text: str,
        case: str = "lower",
    ) -> str:
        """
        Normalize text case.

        Args:
            text: The input text to normalize.
            case: The case to normalize to ('lower' or 'upper').

        Returns:
            Normalized text.

        """
        if case == "lower":
            return text.lower()
        if case == "upper":
            return text.upper()
        return text

    def remove_stopwords(
        self,
        text: str | list[str],
        custom_stopwords: list[str] | None = None,
    ) -> str | list[str]:
        """
        Remove stopwords from text.

        Args:
            text: Input text or list of tokens.
            custom_stopwords: Additional stopwords to remove.

        Returns:
            Text or tokens with stopwords removed.

        """
        stopwords = self.stopwords.copy()
        if custom_stopwords:
            stopwords.update(custom_stopwords)

        if isinstance(text, str):
            doc = self.nlp(text)
            return " ".join(
                [token.text for token in doc if token.text.lower() not in stopwords],
            )
        return [token for token in text if token.lower() not in stopwords]

    def lemmatize(self, text: str) -> list[str]:
        """
        Lemmatize text using spaCy.

        Args:
            text: The input text to lemmatize.

        Returns:
            List of lemmatized tokens.

        """
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]
