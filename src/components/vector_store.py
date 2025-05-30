import os
import pickle
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from src.utils.logger import setup_logger
from src.components.embedding import EmbeddingModel
from config.settings import get_settings


logger = setup_logger()

class VectorStore:
    def __init__(self, store_path: str = "data/vector_db"):
        self.settings = get_settings()
        self.embedding_model = EmbeddingModel()
        self.embeddings = self.embedding_model.get()
        self.embedding_model_name = self.settings.embedding_model_name
        self.vector_store = None
        self.documents = []
        self.store_path = store_path
        os.makedirs(self.store_path, exist_ok=True)
        logger.info(f"VectorStore initialized with model: {self.embedding_model_name}")

    def create(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        self.documents = documents
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info(f"Vector store created with {len(documents)} documents")

    def add_documents(self, documents: List[Document]) -> None:
        if not self.vector_store:
            logger.warning("No existing vector store. Creating a new one.")
            self.create(documents)
            return
        self.vector_store.add_documents(documents)
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents. Total now: {len(self.documents)}")

    def save(self, path: Optional[str] = None) -> None:
        if not self.vector_store:
            raise ValueError("No vector store to save.")
        path = path or os.path.join(self.store_path, "faiss_index")
        self.vector_store.save_local(path)
        with open(os.path.join(path, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Vector store saved at {path}")

    def load(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.store_path, "faiss_index")
        if not os.path.exists(path):
            logger.warning(f"No vector store found at {path}")
            return
        self.vector_store = FAISS.load_local(path, embeddings=self.embeddings, allow_dangerous_deserialization=True)
        docs_path = os.path.join(path, "documents.pkl")
        if os.path.exists(docs_path):
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
        logger.info(f"Loaded vector store from {path}. Docs: {len(self.documents)}")

    def delete(self) -> None:
        self.vector_store = None
        self.documents = []
        path = os.path.join(self.store_path, "faiss_index")
        if os.path.exists(path):
            shutil.rmtree(path)
        logger.info("Vector store deleted.")

    def get_documents(self) -> List[Document]:
        return self.documents

    def get_vector_store(self):
        return self.vector_store

    def get_embedding_dim(self) -> int:
        if self.vector_store:
            return self.vector_store.index.d
        return len(self.embeddings.embed_query("test"))
    def get_document_count(self) -> int:
        return len(self.documents) if self.documents else 0
