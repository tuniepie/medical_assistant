import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.core.rag_pipeline import RAGPipeline  
from config.settings import get_settings



@pytest.fixture
def mock_settings():
    mock = MagicMock()
    mock.temperature = 0.7
    mock.retrieval_k = 3
    return mock


@pytest.fixture
def mock_vectorstore():
    return MagicMock()


@pytest.fixture
def mock_retriever(mock_vectorstore):
    with patch('src.core.rag_pipeline.Retriever') as MockRetriever:
        instance = MockRetriever.return_value
        doc = Document(page_content="This is a test document.", metadata={"source": "test.txt"})
        instance.similarity_search.return_value = [(doc, 0.9)]
        yield instance


@pytest.fixture
def mock_generator():
    with patch('src.core.rag_pipeline.Generator') as MockGen:
        instance = MockGen.return_value
        instance.generate.return_value = "This is a generated answer."
        instance.stream.return_value = iter(["This", " is", " streamed."])
        instance.model_name = "mock-model"
        instance.temperature = 0.7
        instance.get_config.return_value = {"model": "mock-model", "temperature": 0.7}
        yield instance


@patch('src.core.rag_pipeline.get_settings')
def test_generate_response(mock_get_settings, mock_generator, mock_retriever, mock_vectorstore, mock_settings):
    mock_get_settings.return_value = mock_settings
    pipeline = RAGPipeline()
    result = pipeline.generate_response("What is COVID-19?", vectorstore=mock_vectorstore)

    assert result["answer"] == "This is a generated answer."
    assert result["metadata"]["retrieved_docs"] == 1
    assert "model" in result["metadata"]


@patch('src.core.rag_pipeline.get_settings')
def test_generate_streaming_response(mock_get_settings, mock_generator, mock_retriever, mock_settings):
    mock_get_settings.return_value = mock_settings
    pipeline = RAGPipeline()
    pipeline.retriever = mock_retriever
    pipeline.generator = mock_generator

    result = list(pipeline.generate_streaming_response("What is COVID-19?"))
    assert result == ["This", " is", " streamed."]


@patch("src.core.rag_pipeline.get_settings")
def test_update_settings_logs_changes(mock_get_settings, mock_generator, mock_retriever, mock_settings, caplog):
    # Mock the settings object returned by get_settings
    mock_settings = MagicMock()
    mock_settings.temperature = 0.7
    mock_settings.retrieval_k = 3
    mock_get_settings.return_value = mock_settings

    pipeline = RAGPipeline()

    with caplog.at_level("INFO"):
        pipeline.update_settings(temperature=0.5, k=10)

    assert "temperature: 0.5" in caplog.text
    assert "retrieval_k: 10" in caplog.text

@patch("src.core.rag_pipeline.get_settings")
def test_evaluate_response_quality(mock_get_settings, mock_generator, mock_retriever, mock_settings):
    mock_settings = MagicMock()
    mock_settings.temperature = 0.7
    mock_settings.retrieval_k = 3
    mock_get_settings.return_value = mock_settings

    pipeline = RAGPipeline()

    doc = Document(page_content="Diabetes is a chronic condition.", metadata={"source": "medical.txt"})
    sources = [(doc, 0.9)]
    response = "Consult your doctor. See medical.txt for details."

    metrics = pipeline.evaluate_response_quality("What is diabetes?", response, sources)

    assert metrics["response_length"] > 10
    assert metrics["contains_disclaimer"]
    assert metrics["cites_sources"]
    assert "quality_score" in metrics

@patch('src.core.rag_pipeline.get_settings')
def test_get_pipeline_info(mock_get_settings, mock_generator, mock_settings):
    mock_get_settings.return_value = mock_settings
    pipeline = RAGPipeline()
    info = pipeline.get_pipeline_info()
    assert info["status"] == "ready"
    assert "model" in info
