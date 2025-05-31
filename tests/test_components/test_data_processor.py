import os
import pytest
from unittest.mock import patch, MagicMock
from src.components.data_processor import DataProcessor
from langchain_core.documents import Document

@pytest.fixture
def processor():
    return DataProcessor(chunk_size=100, chunk_overlap=20)

def test_preprocess_text(processor):
    text = "   Hello   World\n\n\tTest\x00ing   "
    clean = processor.preprocess_text(text)
    assert clean == "Hello World Testing"

def test_chunk_documents_basic(processor):
    docs = [Document(page_content="This is a test document " * 50)]
    chunks = processor.chunk_documents(docs)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Document) for c in chunks)
    assert len(chunks) > 1  # chunking happened

def test_filter_chunks_by_length(processor):
    chunks = [
        Document(page_content="short"),
        Document(page_content="long enough content here"),
    ]
    filtered = processor.filter_chunks_by_length(chunks, min_length=10)
    assert len(filtered) == 1
    assert filtered[0].page_content == "long enough content here"

def test_get_document_stats(processor):
    chunks = [
        Document(page_content="abc", metadata={"source": "file1"}),
        Document(page_content="defgh", metadata={"source": "file2"}),
        Document(page_content="ijklmn", metadata={"source": "file1"}),
    ]
    stats = processor.get_document_stats(chunks)
    assert stats["total_chunks"] == 3
    assert stats["total_characters"] == 14
    assert stats["chunk_size_distribution"]["min"] == 3
    assert stats["chunk_size_distribution"]["max"] == 6
    assert "file1" in stats["sources"]

@patch("src.components.data_processor.DataLoader")
def test_process_document_mock(mock_loader_class, processor):
    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        Document(page_content="sample content")
    ]
    mock_loader_class.return_value = mock_loader

    chunks = processor.process_document("fake/path.txt", source_name="test-source")
    assert all("test-source" == c.metadata["source"] for c in chunks)
    assert all("fake/path.txt" == c.metadata["file_path"] for c in chunks)
    assert "chunk_id" in chunks[0].metadata

@patch("src.components.data_processor.DataProcessor.process_document")
def test_process_multiple_documents(mock_process, processor):
    mock_process.side_effect = [
        [Document(page_content="doc1")],
        [Document(page_content="doc2"), Document(page_content="doc3")],
    ]
    result = processor.process_multiple_documents(["file1.txt", "file2.txt"])
    assert len(result) == 3
    assert result[0].page_content == "doc1"