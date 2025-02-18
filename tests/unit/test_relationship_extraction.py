"""Unit tests for the relationship extraction module."""

from unittest.mock import Mock, patch

import pytest
import spacy

from biogather.relationship_extraction import RelationshipExtractor
from biogather.utils import is_scispacy_available


@pytest.fixture
def mock_transformer():
    """Create a mock transformer pipeline for testing."""
    mock = Mock()
    mock.return_value = [
        {"entity": "GENE", "word": "BRCA1", "score": 0.99, "start": 0, "end": 5},
        {"entity": "GENE", "word": "TP53", "score": 0.98, "start": 19, "end": 23},
    ]
    return mock


@pytest.fixture
def mock_nlp():
    """Create a mock spaCy NLP pipeline."""
    nlp = spacy.blank("en")

    # Register entity labels in the vocab
    for label in ["GENE"]:
        nlp.vocab.strings.add(label)

    # Create a custom component for POS tagging and entity recognition
    @spacy.Language.component("custom_processor")
    def custom_processor(doc):
        # Set POS tags and lemmas
        for token in doc:
            # Set lemmas for verbs
            if token.text == "interacts":
                token.pos_ = "VERB"
                token.lemma_ = "interact"
            elif token.text == "regulates":
                token.pos_ = "VERB"
                token.lemma_ = "regulate"
            elif token.text == "activates":
                token.pos_ = "VERB"
                token.lemma_ = "activate"
            elif token.text == "inhibits":
                token.pos_ = "VERB"
                token.lemma_ = "inhibit"
            elif token.text == "does":
                token.pos_ = "VERB"
                token.lemma_ = "do"
            elif token.text == "checks":
                token.pos_ = "VERB"
                token.lemma_ = "check"
            elif token.text == "with":
                token.pos_ = "ADP"
                token.lemma_ = "with"
            elif token.text == "This":
                token.pos_ = "DET"
                token.lemma_ = "this"
            elif token.text == "cell" or token.text == "cycle":
                token.pos_ = "NOUN"
                token.lemma_ = token.text.lower()
            elif token.text == "interaction":
                token.pos_ = "NOUN"
                token.lemma_ = "interaction"
            elif token.text == ".":
                token.pos_ = "PUNCT"
                token.lemma_ = "."
            else:
                token.pos_ = "PROPN"
                token.lemma_ = token.text

        # Create entity spans for genes and proteins
        ents = []
        used_tokens = set()  # Track which tokens are already part of an entity
        protein_names = ["BRCA1", "TP53", "Protein A", "Protein B", "Protein C", "Protein D", "A", "B", "C", "D"]
        text = doc.text

        # Sort protein names by length (descending) to prioritize longer matches
        protein_names.sort(key=len, reverse=True)

        for protein in protein_names:
            start_idx = text.find(protein)
            while start_idx != -1:  # Handle multiple occurrences
                end_idx = start_idx + len(protein)
                # Find the token indices that correspond to these character indices
                start_token = None
                end_token = None
                for token in doc:
                    if token.idx <= start_idx < token.idx + len(token.text):
                        start_token = token.i
                    if token.idx < end_idx <= token.idx + len(token.text):
                        end_token = token.i + 1

                # Only create span if tokens aren't already used
                if (
                    start_token is not None
                    and end_token is not None
                    and not any(i in used_tokens for i in range(start_token, end_token))
                ):
                    span = doc[start_token:end_token]
                    span.label_ = "GENE"  # Using GENE for both genes and proteins in this mock
                    ents.append(span)
                    # Mark these tokens as used
                    used_tokens.update(range(start_token, end_token))

                # Look for next occurrence
                start_idx = text.find(protein, start_idx + 1)

        doc.ents = ents
        return doc

    # Add components to pipeline
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("custom_processor")

    def create_doc_with_ents(text):
        doc = nlp(text)
        return doc

    nlp.pipe = Mock(side_effect=lambda texts: (create_doc_with_ents(text) for text in texts))
    return nlp


@pytest.fixture
def extractor(mock_transformer, mock_nlp):
    """Create a basic RelationshipExtractor instance for testing."""
    with patch("transformers.pipeline", return_value=mock_transformer):
        with patch("spacy.load", return_value=mock_nlp):
            return RelationshipExtractor(use_biomedical=False)


@pytest.fixture
def biomedical_extractor(mock_transformer, mock_nlp):
    """Create a biomedical RelationshipExtractor instance for testing."""
    with patch("transformers.pipeline", return_value=mock_transformer):
        with patch("spacy.load", return_value=mock_nlp):
            if not is_scispacy_available():
                pytest.skip("requires scispacy")
            return RelationshipExtractor(use_biomedical=True)


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing."""
    return "BRCA1 interacts with TP53. This interaction regulates cell cycle."


def test_extract_entity_pairs_basic(extractor, sample_text):
    """Test entity pair extraction with basic model."""
    pairs = extractor.extract_entity_pairs(sample_text)
    assert len(pairs) > 0
    assert any(p["entity1"]["text"] == "BRCA1" and p["entity2"]["text"] == "TP53" for p in pairs)


@pytest.mark.skipif(not is_scispacy_available(), reason="requires scispacy")
def test_extract_entity_pairs_biomedical(biomedical_extractor):
    """Test entity pair extraction with biomedical model."""
    text = "IL2 binds to IL2R. This binding activates T cells."
    pairs = biomedical_extractor.extract_entity_pairs(text)
    assert len(pairs) > 0
    assert any(p["entity1"]["text"] == "IL2" and p["entity2"]["text"] == "IL2R" for p in pairs)


def test_extract_relationships_basic(extractor, sample_text):
    """Test relationship extraction with basic model."""
    relationships = extractor.extract_relationships(sample_text)
    assert len(relationships) > 0
    assert any(r["type"] == "INTERACTS_WITH" and r["entity1"]["text"] == "BRCA1" for r in relationships)


@pytest.mark.skipif(not is_scispacy_available(), reason="requires scispacy")
def test_extract_relationships_biomedical(biomedical_extractor):
    """Test relationship extraction with biomedical model."""
    text = "EGFR activates MAPK signaling. This activation promotes cell growth."
    relationships = biomedical_extractor.extract_relationships(text)
    assert len(relationships) > 0
    assert any(r["type"] == "activation" and "EGFR" in r["arguments"] for r in relationships)


def test_extract_events_basic(extractor, sample_text):
    """Test event extraction with basic model."""
    events = extractor.extract_events(sample_text)
    assert len(events) > 0
    assert any(e["type"] == "binding" and any(arg["text"] == "BRCA1" for arg in e["arguments"]) for e in events)


@pytest.mark.skipif(not is_scispacy_available(), reason="requires scispacy")
def test_extract_events_biomedical(biomedical_extractor):
    """Test event extraction with biomedical model."""
    text = "p53 regulates apoptosis. MDM2 inhibits this regulation."
    events = biomedical_extractor.extract_events(text)
    assert len(events) > 0
    assert any(e["type"] == "regulation" and "p53" in str(e["arguments"]) for e in events)


def test_event_types(extractor):
    """Test detection of different event types."""
    text = "Protein A activates Protein B. Protein C inhibits Protein D."
    events = extractor.extract_events(text)
    event_types = {e["type"] for e in events}
    assert "activation" in event_types
    assert "inhibition" in event_types


def test_argument_roles(extractor):
    """Test argument role determination."""
    text = "BRCA1 regulates TP53 expression"
    role = extractor.determine_argument_role(text, "BRCA1", "regulates")
    assert role in ["Agent", "Theme"]


@pytest.mark.skipif(not is_scispacy_available(), reason="requires scispacy")
def test_biomedical_patterns(mock_transformer):
    """Test biomedical-specific relationship patterns."""
    with patch("transformers.pipeline", return_value=mock_transformer):
        extractor = RelationshipExtractor(use_biomedical=True)
        text = "Kinase phosphorylates substrate. Receptor binds ligand."
        relationships = extractor.extract_relationships(text)
        assert any(r["type"] == "phosphorylation" for r in relationships)
        assert any(r["type"] == "binding" for r in relationships)


def test_relationship_patterns(extractor):
    """Test different relationship patterns."""
    patterns = [
        {"type": "custom", "verbs": {"does", "performs"}},
        {"type": "test", "verbs": {"checks", "validates"}},
    ]
    text = "A does B. C checks D."
    relationships = extractor.extract_relationships(text, patterns=patterns)
    assert any(r["type"] == "custom" for r in relationships)
    assert any(r["type"] == "test" for r in relationships)


@pytest.mark.skipif(not is_scispacy_available(), reason="requires scispacy")
def test_biomedical_error(mock_transformer):
    """Test error handling when trying to use biomedical features without scispacy."""
    with patch("transformers.pipeline", return_value=mock_transformer):
        with pytest.raises(ImportError):
            RelationshipExtractor(use_biomedical=True, model="invalid_model")
