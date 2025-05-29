"""
RAG Pipeline Module for Medical Assistant
Handles the complete RAG workflow: retrieval + generation
"""

import time
from typing import Dict, List, Tuple, Optional
# from langchain.llms import OpenAI
from langchain_cohere import ChatCohere
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from src.vector_store import VectorStore
from src.logger import setup_logger

logger = setup_logger()

class RAGPipeline:
    """Complete RAG pipeline for question answering"""
    
    def __init__(self, 
                 model_name: str = "command-r-plus",
                 temperature: float = 0.1,
                 max_tokens: int = 1000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.llm = ChatCohere(  # Updated here
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.prompt_template = self._create_prompt_template()
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=True
        )
        
        logger.info(f"RAG Pipeline initialized - Model: {model_name}, Temperature: {temperature}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        system_message = """You are a helpful medical assistant that provides accurate information based on provided context. 
...
"""
        human_message = """Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
        
        return prompt
    
    def generate_response(self, 
                         query: str, 
                         vector_store: VectorStore, 
                         k: int = 3,
                         temperature: Optional[float] = None) -> Dict:
        start_time = time.time()
        
        try:
            logger.info(f"Starting RAG pipeline for query: '{query[:100]}...'")
            retrieval_start = time.time()
            retrieved_docs = vector_store.similarity_search(query, k=k)
            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find any relevant information...",
                    "sources": [],
                    "metadata": {
                        "retrieval_time": retrieval_time,
                        "generation_time": 0,
                        "total_time": time.time() - start_time,
                        "retrieved_docs": 0
                    }
                }
            
            context = self._prepare_context(retrieved_docs)
            generation_start = time.time()
            
            if temperature is not None and temperature != self.temperature:
                temp_llm = ChatCohere(  # Updated here
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=self.max_tokens
                )
                temp_chain = LLMChain(llm=temp_llm, prompt=self.prompt_template)
                response = temp_chain.run(context=context, question=query)
            else:
                response = self.chain.run(context=context, question=query)
            
            generation_time = time.time() - generation_start
            total_time = time.time() - start_time
            
            result = {
                "answer": response.strip(),
                "sources": retrieved_docs,
                "context": context,
                "metadata": {
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "retrieved_docs": len(retrieved_docs),
                    "model": self.model_name,
                    "temperature": temperature or self.temperature
                }
            }
            
            logger.info(f"RAG pipeline completed successfully in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "answer": f"I encountered an error: {str(e)}",
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "total_time": time.time() - start_time
                }
            }
    
    def _prepare_context(self, retrieved_docs: List[Tuple[Document, float]]) -> str:
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            context_part = f"""
Document {i} (Source: {source}, Relevance Score: {score:.3f}):
{content}
"""
            context_parts.append(context_part)
        context = "\n" + "="*80 + "\n".join(context_parts) + "="*80
        logger.debug(f"Prepared context with {len(retrieved_docs)} documents")
        return context
    
    def generate_streaming_response(self, 
                                  query: str, 
                                  vector_store: VectorStore, 
                                  k: int = 3):
        try:
            retrieved_docs = vector_store.similarity_search(query, k=k)
            if not retrieved_docs:
                yield "I couldn't find any relevant information..."
                return
            context = self._prepare_context(retrieved_docs)
            streaming_llm = ChatCohere(  # Updated here
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                streaming=True
            )
            for chunk in streaming_llm.stream(self.prompt_template.format(context=context, question=query)):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"Error generating response: {str(e)}"
    
    def batch_generate_responses(self, 
                               queries: List[str], 
                               vector_store: VectorStore, 
                               k: int = 3) -> List[Dict]:
        results = []
        logger.info(f"Processing batch of {len(queries)} queries")
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            result = self.generate_response(query, vector_store, k)
            results.append(result)
        logger.info(f"Completed batch processing")
        return results
    
    def evaluate_response_quality(self, 
                                query: str, 
                                response: str, 
                                sources: List[Tuple[Document, float]]) -> Dict:
        metrics = {
            "response_length": len(response),
            "num_sources_used": len(sources),
            "avg_source_relevance": sum(score for _, score in sources) / len(sources) if sources else 0,
            "contains_disclaimer": any(phrase in response.lower() for phrase in [
                "consult", "doctor", "healthcare professional", "medical advice"
            ]),
            "cites_sources": any(source_name in response for source_name, _ in 
                               [(doc.metadata.get('source', ''), score) for doc, score in sources])
        }
        quality_score = 0
        if metrics["response_length"] > 50:
            quality_score += 0.2
        if metrics["num_sources_used"] >= 2:
            quality_score += 0.2
        if metrics["avg_source_relevance"] > 0.7:
            quality_score += 0.2
        if metrics["contains_disclaimer"]:
            quality_score += 0.2
        if metrics["cites_sources"]:
            quality_score += 0.2
        
        metrics["quality_score"] = quality_score
        return metrics
    
    def update_settings(self, 
                       model_name: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None,
                       k: Optional[int] = None):
        updated = []
        
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            updated.append(f"model: {model_name}")
        
        if temperature is not None and temperature != self.temperature:
            self.temperature = temperature
            updated.append(f"temperature: {temperature}")
        
        if max_tokens and max_tokens != self.max_tokens:
            self.max_tokens = max_tokens
            updated.append(f"max_tokens: {max_tokens}")
        
        if updated:
            self.llm = ChatCohere(  # Updated here
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            logger.info(f"Updated RAG pipeline settings: {', '.join(updated)}")
    
    def get_pipeline_info(self) -> Dict:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_template": str(self.prompt_template),
            "status": "ready"
        }