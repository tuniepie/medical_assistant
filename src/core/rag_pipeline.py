import time
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from src.components.generator import Generator
from src.components.retriever import Retriever
from src.components.vector_store import VectorStore
from src.utils.logger import logger
from config.settings import get_settings

class RAGPipeline:
    """Complete RAG pipeline for medical question answering."""

    def __init__(self):
        self.settings = get_settings()
        self.generator = Generator()
        self.temperature = self.settings.temperature
        self.top_k = self.settings.retrieval_k

    def _load_retriever(self):
        pass 
    
        
    def generate_response(self, query: str,
                            vectorstore: VectorStore,                          
                          temperature: Optional[float] = None) -> Dict:
        start = time.time()
        try:
            retriever = Retriever(store_manager=vectorstore)
            retrieved = retriever.similarity_search(query, k=self.top_k)
            retrieval_time = time.time() - start

            if not retrieved:
                return {
                    "answer": "I couldn't find any relevant information...",
                    "sources": [],
                    "metadata": {
                        "retrieved_docs": 0,
                        "retrieval_time": retrieval_time,
                        "generation_time": 0,
                        "total_time": time.time() - start
                    }
                }

            context = self._prepare_context(retrieved)
            if temperature and temperature != self.generator.temperature:
                self.generator.update(temperature=self.temperature)
                answer = self.generator.generate(query, context)
            else:
                answer = self.generator.generate(query, context)

            total_time = time.time() - start
            return {
                "answer": answer,
                "sources": retrieved,
                "context": context,
                "metadata": {
                    "retrieved_docs": len(retrieved),
                    "retrieval_time": retrieval_time,
                    "generation_time": total_time - retrieval_time,
                    "total_time": total_time,
                    "model": self.generator.model_name,
                    "temperature": temperature or self.generator.temperature
                }
            }

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "answer": f"I encountered an error: {str(e)}",
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "total_time": time.time() - start
                }
            }

    def _prepare_context(self, docs: List[Tuple[Document, float]]) -> str:
        parts = []
        for i, (doc, score) in enumerate(docs, 1):
            src = doc.metadata.get('source', 'Unknown')
            parts.append(f"\nDocument {i} (Source: {src}, Score: {score:.3f}):\n{doc.page_content.strip()}")
        return "\n" + "="*80 + "\n".join(parts) + "\n" + "="*80

    def generate_streaming_response(self, query: str, k: int = 3):
        try:
            docs = self.retriever.similarity_search(query, k=k)
            if not docs:
                yield "I couldn't find any relevant information..."
                return
            context = self._prepare_context(docs)
            for chunk in self.generator.stream(query, context):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"Error: {str(e)}"

    def batch_generate_responses(self, queries: List[str], k: int = 3) -> List[Dict]:
        logger.info(f"Batch generating for {len(queries)} queries")
        return [self.generate_response(q, self.retriever, k) for q in queries]

    def evaluate_response_quality(self, query: str, response: str, sources: List[Tuple[Document, float]]) -> Dict:
        metrics = {
            "response_length": len(response),
            "num_sources_used": len(sources),
            "avg_source_relevance": sum(s for _, s in sources) / len(sources) if sources else 0,
            "contains_disclaimer": any(p in response.lower() for p in ["consult", "doctor", "healthcare", "medical advice"]),
            "cites_sources": any(doc.metadata.get('source', '') in response for doc, _ in sources)
        }
        score = sum([
            metrics["response_length"] > 50,
            metrics["num_sources_used"] >= 2,
            metrics["avg_source_relevance"] > 0.7,
            metrics["contains_disclaimer"],
            metrics["cites_sources"]
        ]) * 0.2
        metrics["quality_score"] = round(score, 2)
        return metrics

    def update_settings(self,
                       temperature: Optional[float] = None,
                       k: Optional[int] = None):

        updated = []
        
        
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            updated.append(f"temperature: {temperature}")
        
        if k is not None and k != self.top_k:
            self.top_k = k
            updated.append(f"retrieval_k: {k}")
        
        if updated:
            logger.info(f"Updated settings: {', '.join(updated)}")

    def get_pipeline_info(self) -> Dict:
        return {
            "status": "ready",
            **self.generator.get_config()
        }
