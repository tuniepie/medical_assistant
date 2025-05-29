import os
import pickle
import shutil
import numpy as np
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from src.logger import setup_logger

# Load environment variables
load_dotenv()
logger = setup_logger()

class VectorStore:
    """
    Manages a FAISS-based vector store using Cohere embeddings.
    """

    def __init__(self, embedding_model: str = "embed-english-v3.0"):
        self.embedding_model = embedding_model
        self.embeddings = CohereEmbeddings(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            model=embedding_model
        )
        self.vector_store = None
        self.documents = []
        self.store_path = "data/vector_db"
        os.makedirs(self.store_path, exist_ok=True)

        logger.info(f"VectorStore initialized with model: {embedding_model}")

    # -------------------------------
    # Vector store creation & updates
    # -------------------------------
    def create_vector_store(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        self.documents = documents
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info(f"Vector store created with {len(documents)} documents")

    def add_documents(self, documents: List[Document]) -> None:
        if not self.vector_store:
            logger.warning("No existing vector store. Creating a new one.")
            self.create_vector_store(documents)
            return
        self.vector_store.add_documents(documents)
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents. Total now: {len(self.documents)}")

    # def update_embedding_model(self, model_name: str) -> None:
    #     logger.warning("Changing embedding model will recreate vector store.")
    #     self.embedding_model = model_name
    #     self.embeddings = CohereEmbeddings(
    #         cohere_api_key=os.getenv("COHERE_API_KEY"),
    #         model=model_name
    #     )
    #     if self.documents:
    #         self.create_vector_store(self.documents)
    #     logger.info(f"Embedding model updated to: {model_name}")

    # -------------------------------
    # Similarity search
    # -------------------------------
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")
        logger.info(f"Searching top {k} documents for query: {query[:50]}...")
        results = self.vector_store.similarity_search_with_score(query, k=k)
        for i, (doc, score) in enumerate(results):
            logger.debug(f"  - Rank {i+1}: Score={score:.4f}, Source={doc.metadata.get('source', 'Unknown')}")
        return results

    def similarity_search_with_threshold(self, query: str, k: int = 3, threshold: float = 0.7) -> List[Tuple[Document, float]]:
        results = self.similarity_search(query, k=k * 2)
        filtered = [(doc, score) for doc, score in results if score <= threshold][:k]
        logger.info(f"Filtered to {len(filtered)} results under threshold {threshold}")
        return filtered

    def get_relevant_context(self, query: str, k: int = 3, max_context_length: int = 4000) -> str:
        results = self.similarity_search(query, k=k)
        context_parts, total_len = [], 0

        for doc, _ in results:
            content = doc.page_content.strip()
            source = doc.metadata.get('source', 'Unknown')
            block = f"[Source: {source}]\n{content}\n"

            if total_len + len(block) > max_context_length:
                space_left = max_context_length - total_len
                if space_left > 100:
                    context_parts.append(block[:space_left-3] + "...")
                break

            context_parts.append(block)
            total_len += len(block)

        final_context = "\n".join(context_parts)
        logger.info(f"Context length: {len(final_context)} characters from {len(context_parts)} docs")
        return final_context

    # -------------------------------
    # Save/load/delete vector store
    # -------------------------------
    def save_vector_store(self, path: Optional[str] = None) -> None:
        if not self.vector_store:
            raise ValueError("No vector store to save.")
        path = path or os.path.join(self.store_path, "faiss_index")
        self.vector_store.save_local(path)
        with open(os.path.join(path, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Vector store saved at {path}")

    def load_vector_store(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.store_path, "faiss_index")
        if not os.path.exists(path):
            logger.warning(f"No vector store found at {path}")
            return
        self.vector_store = FAISS.load_local(
            path, embeddings=self.embeddings, allow_dangerous_deserialization=True
        )
        docs_path = os.path.join(path, "documents.pkl")
        if os.path.exists(docs_path):
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
        logger.info(f"Loaded vector store from {path}. Docs: {len(self.documents)}")

    def delete_vector_store(self) -> None:
        self.vector_store = None
        self.documents = []
        path = os.path.join(self.store_path, "faiss_index")
        if os.path.exists(path):
            shutil.rmtree(path)
        logger.info("Vector store deleted.")

    # -------------------------------
    # Metadata search & info
    # -------------------------------
    def search_by_metadata(self, metadata_filter: dict, k: int = 10) -> List[Document]:
        results = [
            doc for doc in self.documents
            if all(doc.metadata.get(k) == v for k, v in metadata_filter.items())
        ]
        logger.info(f"{len(results[:k])} documents matched filter: {metadata_filter}")
        return results[:k]

    def get_store_info(self) -> dict:
        if not self.vector_store:
            return {"status": "not_initialized", "document_count": 0, "embedding_model": self.embedding_model}
        avg_len = np.mean([len(doc.page_content) for doc in self.documents]) if self.documents else 0
        sources = list({doc.metadata.get('source', 'Unknown') for doc in self.documents})
        return {
            "status": "ready",
            "document_count": len(self.documents),
            "unique_sources": len(sources),
            "sources": sources,
            "embedding_model": self.embedding_model,
            "average_doc_length": avg_len,
            "total_characters": sum(len(doc.page_content) for doc in self.documents)
        }

    def get_document_count(self) -> int:
        return len(self.documents)

    def get_embedding_dimension(self) -> int:
        if self.vector_store:
            return self.vector_store.index.d
        return len(self.embeddings.embed_query("test"))

# -------------------------------
# Utility function
# -------------------------------
def calculate_similarity_threshold(query: str, embeddings: CohereEmbeddings, sample_docs: List[str], percentile: float = 75) -> float:
    if not sample_docs:
        return 0.7
    try:
        q_vec = embeddings.embed_query(query)
        scores = []
        for doc in sample_docs:
            d_vec = embeddings.embed_query(doc)
            sim = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
            scores.append(sim)
        threshold = np.percentile(scores, percentile)
        return float(threshold)
    except Exception as e:
        logger.error(f"Threshold calculation error: {str(e)}")
        return 0.7
