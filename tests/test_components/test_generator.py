import pytest
from unittest.mock import patch, MagicMock
from src.components.generator import Generator

@pytest.fixture
def mock_settings():
    class Settings:
        cohere_api_key = "fake-key"
        llm = "fake-model"
        temperature = 0.7
        max_tokens = 256
    return Settings()

@patch("src.components.generator.ChatPromptTemplate")
@patch("src.components.generator.ChatCohere")
@patch("src.components.generator.get_settings")
def test_initialization(mock_get_settings, mock_chat_cohere, mock_prompt_template, mock_settings):
    mock_get_settings.return_value = mock_settings
    mock_prompt_template.from_messages.return_value = MagicMock()
    mock_chat_cohere.return_value = MagicMock()

    generator = Generator()

    assert generator.model_name == "fake-model"
    assert generator.temperature == 0.7
    assert generator.max_tokens == 256
    mock_chat_cohere.assert_called_once()
    mock_prompt_template.from_messages.assert_called_once()

@patch("src.components.generator.ChatPromptTemplate")
@patch("src.components.generator.ChatCohere")
@patch("src.components.generator.get_settings")
def test_generate(mock_get_settings, mock_chat_cohere, mock_prompt_template, mock_settings):
    mock_get_settings.return_value = mock_settings
    mock_prompt = MagicMock()
    mock_prompt.format.return_value = "formatted prompt"
    mock_prompt_template.from_messages.return_value = mock_prompt

    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = " Final answer "
    mock_chat_cohere.return_value = mock_llm

    generator = Generator()
    result = generator.generate("What is a migraine?", "Context about migraines")
    
    mock_llm.invoke.assert_called_once_with("formatted prompt")
    assert result == "Final answer"

@patch("src.components.generator.ChatPromptTemplate")
@patch("src.components.generator.ChatCohere")
@patch("src.components.generator.get_settings")
def test_stream(mock_get_settings, mock_chat_cohere, mock_prompt_template, mock_settings):
    mock_get_settings.return_value = mock_settings
    mock_prompt = MagicMock()
    mock_prompt.format.return_value = "formatted prompt"
    mock_prompt_template.from_messages.return_value = mock_prompt

    mock_streaming_llm = MagicMock()
    mock_streaming_llm.stream.return_value = [
        MagicMock(content="Part1"),
        MagicMock(content="Part2"),
        MagicMock(content=None)
    ]
    mock_chat_cohere.side_effect = [MagicMock(), mock_streaming_llm]  # One for init, one for stream()

    generator = Generator()
    chunks = list(generator.stream("What is flu?", "Flu info"))
    
    assert chunks == ["Part1", "Part2"]
    mock_streaming_llm.stream.assert_called_once_with("formatted prompt")

@patch("src.components.generator.ChatPromptTemplate")
@patch("src.components.generator.ChatCohere")
@patch("src.components.generator.get_settings")
def test_update(mock_get_settings, mock_chat_cohere, mock_prompt_template, mock_settings):
    mock_get_settings.return_value = mock_settings
    mock_prompt_template.from_messages.return_value = MagicMock()
    mock_chat_cohere.return_value = MagicMock()

    generator = Generator()
    generator.update(model_name="new-model", temperature=0.2, max_tokens=128)

    assert generator.model_name == "new-model"
    assert generator.temperature == 0.2
    assert generator.max_tokens == 128
    assert mock_chat_cohere.call_count == 2  # one during init, one after update

@patch("src.components.generator.ChatPromptTemplate")
@patch("src.components.generator.ChatCohere")
@patch("src.components.generator.get_settings")
def test_get_config(mock_get_settings, mock_chat_cohere, mock_prompt_template, mock_settings):
    mock_get_settings.return_value = mock_settings
    fake_prompt = MagicMock()
    fake_prompt.__str__.return_value = "PromptTemplate"
    mock_prompt_template.from_messages.return_value = fake_prompt
    mock_chat_cohere.return_value = MagicMock()

    generator = Generator()
    config = generator.get_config()

    assert config["model_name"] == "fake-model"
    assert config["temperature"] == 0.7
    assert config["max_tokens"] == 256
    assert config["prompt_template"] == "PromptTemplate"
