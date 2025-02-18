"""
Unit tests for the preprocessing module.
"""

import pytest
from unittest.mock import Mock, patch

from biogather.preprocessing import TextPreprocessor
from biogather.utils import is_scispacy_available


def is_scispacy_available() -> bool:
    """Check if scispacy is available."""
    try:
        import scispacy  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Create a basic TextPreprocessor instance for testing."""
    return TextPreprocessor(use_biomedical=False)


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing."""
    return "The protein BRCA1 is involved in DNA repair. It helps prevent cancer."


def test_tokenize_basic(preprocessor, sample_text):
    """Test text tokenization with basic model."""
    tokens = preprocessor.tokenize(sample_text)
    assert len(tokens) > 0
    assert isinstance(tokens[0], str)


@pytest.mark.skipif(not is_scispacy_available(), reason="requires scispacy")
def test_tokenize_biomedical(sample_text):
    """Test text tokenization with biomedical model."""
    preprocessor = TextPreprocessor(use_biomedical=True)
    tokens = preprocessor.tokenize(sample_text)
    assert len(tokens) > 0
    assert "BRCA1" in tokens


def test_split_sentences(preprocessor, sample_text):
    """Test sentence splitting."""
    sentences = preprocessor.split_sentences(sample_text)
    assert len(sentences) == 2
    assert "BRCA1" in sentences[0]


@pytest.mark.skipif(not is_scispacy_available(), reason="requires scispacy")
def test_split_sentences_biomedical():
    """Test sentence splitting with biomedical text."""
    preprocessor = TextPreprocessor(use_biomedical=True)
    text = "p53 regulates cell cycle. It interacts with MDM2."
    sentences = preprocessor.split_sentences(text)
    assert len(sentences) == 2
    assert "p53" in sentences[0]


def test_normalize_case(preprocessor):
    """Test case normalization."""
    text = "BRCA1 and TP53 are important genes."
    lower = preprocessor.normalize_case(text, case="lower")
    upper = preprocessor.normalize_case(text, case="upper")
    assert "brca1" in lower
    assert "BRCA1" in upper


def test_remove_stopwords(preprocessor):
    """Test stopword removal."""
    text = "The protein BRCA1 is important."
    filtered = preprocessor.remove_stopwords(text)
    assert "the" not in filtered.lower()
    assert "is" not in filtered.lower()
    assert "BRCA1" in filtered


@pytest.mark.skipif(not is_scispacy_available(), reason="requires scispacy")
def test_remove_stopwords_biomedical():
    """Test stopword removal with biomedical text."""
    preprocessor = TextPreprocessor(use_biomedical=True)
    text = "The protein p53 is a tumor suppressor."
    filtered = preprocessor.remove_stopwords(text)
    assert "p53" in filtered
    assert "tumor suppressor" in filtered.lower()


def test_lemmatize(preprocessor):
    """Test text lemmatization."""
    text = "proteins are binding to receptors"
    lemmas = preprocessor.lemmatize(text)
    assert "protein" in lemmas
    assert "bind" in lemmas
    assert "receptor" in lemmas


@pytest.mark.skipif(not is_scispacy_available(), reason="requires scispacy")
def test_lemmatize_biomedical():
    """Test lemmatization with biomedical terms."""
    preprocessor = TextPreprocessor(use_biomedical=True)
    text = "kinases are phosphorylating substrates"
    lemmas = preprocessor.lemmatize(text)
    assert "kinase" in lemmas
    assert "phosphorylate" in lemmas
    assert "substrate" in lemmas


@pytest.mark.skipif(not is_scispacy_available(), reason="requires scispacy")
def test_biomedical_error():
    """Test error handling when trying to use biomedical features without scispacy."""
    with pytest.raises(ImportError):
        TextPreprocessor(use_biomedical=True, model="invalid_model")
