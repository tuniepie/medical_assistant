from unittest.mock import patch, MagicMock
from src.components.embedding import EmbeddingModel

@patch("src.components.embedding.get_settings")
@patch("src.components.embedding.CohereEmbeddings")
def test_initialization(mock_cohere, mock_settings):
    mock_settings.return_value.embedding_model_name = "embed-model"
    mock_settings.return_value.cohere_api_key = "fake-api-key"

    mock_embed_instance = MagicMock()
    mock_cohere.return_value = mock_embed_instance

    model = EmbeddingModel()

    mock_settings.assert_called_once()
    mock_cohere.assert_called_once_with(
        cohere_api_key="fake-api-key",
        model="embed-model"
    )
    assert model.get() == mock_embed_instance

@patch("src.components.embedding.get_settings")
@patch("src.components.embedding.CohereEmbeddings")
def test_embed_query(mock_cohere, mock_settings):
    mock_settings.return_value.embedding_model_name = "embed-model"
    mock_settings.return_value.cohere_api_key = "fake-api-key"

    mock_embed_instance = MagicMock()
    mock_embed_instance.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_cohere.return_value = mock_embed_instance

    model = EmbeddingModel()
    result = model.embed_query("hello world")

    mock_embed_instance.embed_query.assert_called_once_with("hello world")
    assert result == [0.1, 0.2, 0.3]
