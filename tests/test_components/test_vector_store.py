import pytest
from unittest.mock import patch, MagicMock, mock_open
from src.components.vector_store import VectorStore
from langchain_core.documents import Document


@pytest.fixture
def dummy_docs():
    return [Document(page_content="Doc 1"), Document(page_content="Doc 2")]


@pytest.fixture
def mock_embeddings():
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 384  # example dim
    return mock


@patch("src.components.vector_store.FAISS")
@patch("src.components.vector_store.EmbeddingModel")
@patch("src.components.vector_store.get_settings")
def test_create_vector_store(mock_get_settings, MockEmbeddingModel, MockFAISS, dummy_docs, mock_embeddings):
    MockEmbeddingModel.return_value.get.return_value = mock_embeddings
    MockFAISS.from_documents.return_value = MagicMock()

    store = VectorStore()
    store.create(dummy_docs)

    assert store.get_document_count() == 2
    MockFAISS.from_documents.assert_called_once()


@patch("src.components.vector_store.FAISS")
@patch("src.components.vector_store.EmbeddingModel")
@patch("src.components.vector_store.get_settings")
def test_add_documents(mock_get_settings, MockEmbeddingModel, MockFAISS, dummy_docs, mock_embeddings):
    MockEmbeddingModel.return_value.get.return_value = mock_embeddings
    mock_store = MagicMock()
    MockFAISS.from_documents.return_value = mock_store

    store = VectorStore()
    store.create([Document(page_content="Base")])
    store.add_documents(dummy_docs)

    assert store.get_document_count() == 3
    mock_store.add_documents.assert_called_once()


@patch("src.components.vector_store.open", new_callable=mock_open)
@patch("src.components.vector_store.pickle.dump")
@patch("src.components.vector_store.FAISS")
@patch("src.components.vector_store.EmbeddingModel")
@patch("src.components.vector_store.get_settings")
def test_save(mock_get_settings, MockEmbeddingModel, MockFAISS, mock_pickle, mock_file, dummy_docs, mock_embeddings):
    MockEmbeddingModel.return_value.get.return_value = mock_embeddings
    mock_store = MagicMock()
    MockFAISS.from_documents.return_value = mock_store

    store = VectorStore()
    store.create(dummy_docs)
    store.save()

    mock_store.save_local.assert_called()
    mock_pickle.assert_called_once()


@patch("src.components.vector_store.open", new_callable=mock_open)
@patch("src.components.vector_store.pickle.load")
@patch("src.components.vector_store.FAISS")
@patch("src.components.vector_store.EmbeddingModel")
@patch("src.components.vector_store.get_settings")
def test_load(mock_get_settings, MockEmbeddingModel, MockFAISS, mock_pickle, mock_file, dummy_docs, mock_embeddings, tmp_path):
    MockEmbeddingModel.return_value.get.return_value = mock_embeddings
    mock_store = MagicMock()
    MockFAISS.load_local.return_value = mock_store
    mock_pickle.return_value = dummy_docs

    store = VectorStore(store_path=str(tmp_path))
    index_path = tmp_path / "faiss_index"
    index_path.mkdir()
    with open(index_path / "documents.pkl", "wb") as f:
        pass  # simulate file creation

    store.load()
    assert store.get_document_count() == 2
    MockFAISS.load_local.assert_called_once()


@patch("src.components.vector_store.shutil.rmtree")
@patch("src.components.vector_store.EmbeddingModel")
@patch("src.components.vector_store.get_settings")
def test_delete(mock_get_settings, MockEmbeddingModel, mock_rmtree, tmp_path):
    MockEmbeddingModel.return_value.get.return_value = MagicMock()
    store = VectorStore(store_path=str(tmp_path))
    path = tmp_path / "faiss_index"
    path.mkdir()

    store.delete()
    mock_rmtree.assert_called_once()


@patch("src.components.vector_store.EmbeddingModel")
@patch("src.components.vector_store.get_settings")
def test_get_embedding_dim_with_query(mock_get_settings, MockEmbeddingModel, mock_embeddings):
    mock_embeddings.embed_query.return_value = [0.0] * 512
    MockEmbeddingModel.return_value.get.return_value = mock_embeddings
    store = VectorStore()
    dim = store.get_embedding_dim()
    assert dim == 512
