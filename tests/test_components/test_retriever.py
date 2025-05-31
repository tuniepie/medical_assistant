import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.components.retriever import Retriever


@pytest.fixture
def mock_docs():
    return [
        (Document(page_content="Flu is a viral infection", metadata={"source": "doc1"}), 0.5),
        (Document(page_content="Cold symptoms overlap with flu", metadata={"source": "doc2"}), 0.6),
        (Document(page_content="Headache is a common symptom", metadata={"source": "doc3"}), 0.8)
    ]


@pytest.fixture
def mock_store():
    store = MagicMock()
    vector_store = MagicMock()
    store.get_vector_store.return_value = vector_store
    return store


@patch("src.components.retriever.EmbeddingModel")
def test_similarity_search(mock_embedding_model, mock_store, mock_docs):
    mock_store.get_vector_store().similarity_search_with_score.return_value = mock_docs
    retriever = Retriever(store_manager=mock_store)

    results = retriever.similarity_search("What is flu?", k=2)

    assert len(results) == 3
    mock_store.get_vector_store().similarity_search_with_score.assert_called_once()


@patch("src.components.retriever.EmbeddingModel")
def test_similarity_search_with_threshold(mock_embedding_model, mock_store, mock_docs):
    mock_store.get_vector_store().similarity_search_with_score.return_value = mock_docs
    retriever = Retriever(store_manager=mock_store)

    results = retriever.similarity_search_with_threshold("What is flu?", k=2, threshold=0.65)
    assert len(results) == 2  # only the first two are <= 0.65


@patch("src.components.retriever.EmbeddingModel")
def test_get_relevant_context(mock_embedding_model, mock_store, mock_docs):
    mock_store.get_vector_store().similarity_search_with_score.return_value = mock_docs
    retriever = Retriever(store_manager=mock_store)

    context = retriever.get_relevant_context("flu symptoms", k=3, max_context_length=200)
    assert "[Source:" in context
    assert len(context) <= 200


@patch("src.components.retriever.EmbeddingModel")
def test_calculate_similarity_threshold(mock_embedding_model):
    mock_embed = MagicMock()
    mock_embed.embed_query.side_effect = lambda text: np.array([1.0 if "query" in text else 0.5] * 5)
    mock_embedding_model.return_value = mock_embed

    retriever = Retriever(store_manager=MagicMock())
    threshold = retriever.calculate_similarity_threshold(
        query="query sample",
        sample_docs=["doc1", "doc2", "doc3"],
        percentile=75
    )

    assert 0 <= threshold <= 1


@patch("src.components.retriever.EmbeddingModel")
def test_calculate_similarity_threshold_empty(mock_embedding_model):
    mock_embedding_model.return_value.embed_query.return_value = [0.1] * 5
    retriever = Retriever(store_manager=MagicMock())
    threshold = retriever.calculate_similarity_threshold("query", [])
    assert threshold == 0.7  # fallback value
