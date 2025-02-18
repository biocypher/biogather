"""Unit tests for the linguistic annotation module."""

import pytest

from biogather.linguistic_annotation import LinguisticAnnotator

@pytest.fixture
def annotator() -> LinguisticAnnotator:
    """Create a LinguisticAnnotator instance for testing."""
    return LinguisticAnnotator()


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing."""
    return "The protein BRCA1 regulates cell cycle. It helps prevent cancer."


def test_pos_tagging(annotator, sample_text):
    """Test part-of-speech tagging."""
    doc = annotator.nlp(sample_text)
    pos_tags = [token.pos_ for token in doc]
    assert "NOUN" in pos_tags
    assert "VERB" in pos_tags
    assert "DET" in pos_tags


def test_dependency_parsing(annotator, sample_text):
    """Test dependency parsing."""
    doc = annotator.nlp(sample_text)
    deps = [(token.text, token.dep_, token.head.text) for token in doc]

    # Check for basic dependency relations
    assert any(dep[1] == "nsubj" for dep in deps)  # subject
    assert any(dep[1] == "dobj" for dep in deps)   # direct object
    assert any(dep[1] == "det" for dep in deps)    # determiner


def test_noun_phrase_extraction(annotator, sample_text):
    """Test noun phrase extraction."""
    phrases = annotator.extract_noun_phrases(sample_text)
    assert len(phrases) > 0
    assert any("BRCA1" in phrase for phrase in phrases)
    assert any("cell cycle" in phrase for phrase in phrases)


def test_sentence_structure(annotator, sample_text):
    """Test sentence structure analysis."""
    structures = annotator.analyze_sentence_structure(sample_text)
    assert len(structures) > 0

    # Check first sentence structure
    first_sent = structures[0]
    assert "components" in first_sent
    assert "text" in first_sent

    # Check components
    components = first_sent["components"]
    subject = next((comp for comp in components if comp["role"] == "subject"), None)
    verb = next((comp for comp in components if comp["role"] == "verb"), None)
    obj = next((comp for comp in components if comp["role"] == "object"), None)

    assert subject is not None
    assert verb is not None
    assert obj is not None
    assert subject["text"] == "BRCA1"
    assert "regulates" in verb["text"]
    assert "cycle" in obj["text"]


def test_token_features(annotator, sample_text):
    """Test token feature extraction."""
    features = annotator.get_token_features(sample_text)
    assert len(features) > 0

    # Check feature types
    for token_feat in features:
        assert "text" in token_feat
        assert "pos" in token_feat
        assert "dep" in token_feat
        assert "lemma" in token_feat
