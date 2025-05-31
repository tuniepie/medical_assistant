import os
import pytest
from unittest.mock import patch, MagicMock
from src.components.data_loader import DataLoader

@pytest.fixture
def data_loader():
    return DataLoader()

def test_supported_file_types(data_loader):
    assert data_loader.supported_file_types() == ['.pdf', '.txt']

@pytest.mark.parametrize("file_name,expected", [
    ("document.pdf", True),
    ("notes.txt", True),
    ("image.jpg", False),
])
def test_validate_file_type(data_loader, file_name, expected):
    assert data_loader.validate_file_type(file_name) == expected

def test_load_txt_file_real(data_loader):
    path = os.path.abspath("tests/data/test.txt")
    docs = data_loader.load(path)
    assert len(docs) > 0
    assert isinstance(docs[0].page_content, str)

def test_load_pdf_file_real(data_loader):
    path = os.path.abspath("tests/data/test.pdf")
    docs = data_loader.load(path)
    assert len(docs) > 0
    assert isinstance(docs[0].page_content, str)

def test_load_invalid_file_type(data_loader):
    with pytest.raises(ValueError, match="Unsupported file type: .jpg"):
        data_loader.load("image.jpg")

@patch("src.components.data_loader.DataLoader.load")
def test_load_multiple_files(mock_load, data_loader):
    mock_load.side_effect = [["doc1"], ["doc2", "doc3"]]

    result = data_loader.load_multiple(["file1.pdf", "file2.txt"])
    assert result == ["doc1", "doc2", "doc3"]
    assert mock_load.call_count == 2

@patch("os.stat")
def test_get_file_info_success(mock_stat, data_loader):
    mock_stat_result = os.stat_result((0, 0, 0, 0, 0, 0, 1234, 0, 0, 0)) 
    mock_stat.return_value = mock_stat_result

    info = data_loader.get_file_info("data/test.txt")
    assert info["name"] == "test.txt"
    assert info["size"] == 1234
    assert info["extension"] == ".txt"
    assert info["is_supported"] is True


@patch("os.stat", side_effect=FileNotFoundError("File not found"))
def test_get_file_info_error(mock_stat, data_loader):
    info = data_loader.get_file_info("tests/data/missing.txt")
    assert "error" in info
    assert "File not found" in info["error"]
