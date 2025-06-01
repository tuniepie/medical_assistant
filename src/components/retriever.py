import numpy as np
from typing import List, Tuple
from langchain_core.documents import Document
from src.utils.logger import logger
from src.components.embedding import EmbeddingModel
from src.components.vector_store import VectorStore

class Retriever:
    def __init__(self, store_manager: VectorStore):
        self.store = store_manager
        self.embeddings = EmbeddingModel()

    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        vector_store = self.store.get_vector_store()
        if not vector_store:
            raise ValueError("Vector store not initialized.")
        logger.info(f"Searching top {k} documents for query: {query[:50]}...")
        results = vector_store.similarity_search_with_score(query, k=k)
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
                    context_parts.append(block[:space_left - 3] + "...")
                break
            context_parts.append(block)
            total_len += len(block)

        final_context = "\n".join(context_parts)
        logger.info(f"Context length: {len(final_context)} characters from {len(context_parts)} docs")
        return final_context

    def calculate_similarity_threshold(self, query: str, sample_docs: List[str], percentile: float = 75) -> float:
        if not sample_docs:
            return 0.7
        try:
            q_vec = self.embeddings.embed_query(query)
            scores = []
            for doc in sample_docs:
                d_vec = self.embeddings.embed_query(doc)
                sim = np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec))
                scores.append(sim)
            threshold = np.percentile(scores, percentile)
            return float(threshold)
        except Exception as e:
            logger.error(f"Threshold calculation error: {str(e)}")
            return 0.7
