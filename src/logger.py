"""
Logging Configuration Module for RAG System
Provides structured logging throughout the application
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys
from typing import Dict, Any, List, Tuple, Optional

def setup_logger(name: str = "rag_system", 
                 log_file: str = "logs/rag_system.log",
                 level: int = logging.INFO,
                 max_bytes: int = 10*1024*1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Set up structured logging for the RAG system
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler with rotation
    try:
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create file handler: {e}")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Add a startup log entry
    logger.info(f"Logger '{name}' initialized - Level: {logging.getLevelName(level)}")
    
    return logger

def log_query_response(logger: logging.Logger, 
                      query: str, 
                      response: str, 
                      sources: list,
                      processing_time: float) -> None:
    """
    Log a query-response interaction
    
    Args:
        logger: Logger instance
        query: User query
        response: Generated response
        sources: Retrieved source documents
        processing_time: Time taken to process
    """
    logger.info("="*50)
    logger.info("QUERY-RESPONSE INTERACTION")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Query: {query[:200]}...")
    logger.info(f"Response Length: {len(response)} characters")
    logger.info(f"Sources Retrieved: {len(sources)}")
    logger.info(f"Processing Time: {processing_time:.2f} seconds")
    
    # Log source details
    for i, (doc, score) in enumerate(sources, 1):
        source_name = doc.metadata.get('source', 'Unknown')
        logger.info(f"Source {i}: {source_name} (Score: {score:.3f})")
    
    logger.info("="*50)

def log_document_processing(logger: logging.Logger, 
                          file_path: str, 
                          chunks_created: int,
                          processing_time: float) -> None:
    """
    Log document processing details
    
    Args:
        logger: Logger instance
        file_path: Path to processed document
        chunks_created: Number of chunks created
        processing_time: Time taken to process
    """
    logger.info("-"*30)
    logger.info("DOCUMENT PROCESSING")
    logger.info(f"File: {os.path.basename(file_path)}")
    logger.info(f"Chunks Created: {chunks_created}")
    logger.info(f"Processing Time: {processing_time:.2f} seconds")
    logger.info("-"*30)

def log_vector_store_operation(logger: logging.Logger, 
                             operation: str, 
                             document_count: int,
                             operation_time: float) -> None:
    """
    Log vector store operations
    
    Args:
        logger: Logger instance
        operation: Type of operation (create, add, search, etc.)
        document_count: Number of documents involved
        operation_time: Time taken for operation
    """
    logger.info(f"VECTOR STORE - {operation.upper()}")
    logger.info(f"Documents: {document_count}")
    logger.info(f"Time: {operation_time:.2f} seconds")

def log_error_with_context(logger: logging.Logger, 
                          error: Exception, 
                          context: dict) -> None:
    """
    Log errors with additional context
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
    """
    logger.error("="*50)
    logger.error("ERROR OCCURRED")
    logger.error(f"Error Type: {type(error).__name__}")
    logger.error(f"Error Message: {str(error)}")
    
    for key, value in context.items():
        logger.error(f"{key}: {value}")
    
    logger.error("="*50)

class RAGSystemLogger:
    """Specialized logger class for RAG system operations"""
    
    def __init__(self, 
                 name: str = "rag_system",
                 log_file: str = "logs/rag_system.log",
                 level: int = logging.INFO):
        """
        Initialize the RAG system logger
        
        Args:
            name: Logger name
            log_file: Path to log file
            level: Logging level
        """
        self.logger = setup_logger(name, log_file, level)
        self.name = name
        
    def log_system_startup(self, config: Dict[str, Any]) -> None:
        """Log system startup with configuration details"""
        self.logger.info("="*60)
        self.logger.info("RAG SYSTEM STARTUP")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("="*60)
    
    def log_document_upload(self, 
                           filename: str, 
                           file_size: int, 
                           file_type: str) -> None:
        """Log document upload event"""
        self.logger.info("+"*30)
        self.logger.info("DOCUMENT UPLOAD")
        self.logger.info(f"Filename: {filename}")
        self.logger.info(f"Size: {file_size} bytes")
        self.logger.info(f"Type: {file_type}")
        self.logger.info("+"*30)
    
    def log_embeddings_generation(self, 
                                 chunks_count: int, 
                                 embedding_model: str,
                                 generation_time: float) -> None:
        """Log embeddings generation process"""
        self.logger.info("*"*30)
        self.logger.info("EMBEDDINGS GENERATION")
        self.logger.info(f"Chunks: {chunks_count}")
        self.logger.info(f"Model: {embedding_model}")
        self.logger.info(f"Time: {generation_time:.2f} seconds")
        self.logger.info("*"*30)
    
    def log_retrieval_results(self, 
                            query: str,
                            retrieved_docs: List[Tuple[Any, float]],
                            retrieval_time: float) -> None:
        """Log document retrieval results"""
        self.logger.info("~"*40)
        self.logger.info("DOCUMENT RETRIEVAL")
        self.logger.info(f"Query: {query[:100]}...")
        self.logger.info(f"Retrieved: {len(retrieved_docs)} documents")
        self.logger.info(f"Retrieval Time: {retrieval_time:.3f} seconds")
        
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            chunk_id = doc.metadata.get('chunk_id', 'N/A')
            self.logger.info(f"  Doc {i}: {source} [Chunk: {chunk_id}] (Score: {score:.4f})")
        
        self.logger.info("~"*40)
    
    def log_llm_generation(self, 
                          model_name: str,
                          prompt_length: int,
                          response_length: int,
                          generation_time: float,
                          temperature: float) -> None:
        """Log LLM response generation"""
        self.logger.info("#"*40)
        self.logger.info("LLM GENERATION")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Temperature: {temperature}")
        self.logger.info(f"Prompt Length: {prompt_length} chars")
        self.logger.info(f"Response Length: {response_length} chars")
        self.logger.info(f"Generation Time: {generation_time:.2f} seconds")
        self.logger.info("#"*40)
    
    def log_user_session(self, 
                        session_id: str,
                        action: str,
                        details: Optional[Dict[str, Any]] = None) -> None:
        """Log user session activities"""
        self.logger.info("@"*25)
        self.logger.info(f"USER SESSION - {action.upper()}")
        self.logger.info(f"Session ID: {session_id}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        if details:
            for key, value in details.items():
                self.logger.info(f"{key}: {value}")
        
        self.logger.info("@"*25)
    
    def log_performance_metrics(self, 
                               total_processing_time: float,
                               retrieval_time: float,
                               generation_time: float,
                               total_tokens_used: int) -> None:
        """Log performance metrics for the entire pipeline"""
        self.logger.info("$"*50)
        self.logger.info("PERFORMANCE METRICS")
        self.logger.info(f"Total Processing Time: {total_processing_time:.2f} seconds")
        self.logger.info(f"Retrieval Time: {retrieval_time:.2f} seconds ({(retrieval_time/total_processing_time)*100:.1f}%)")
        self.logger.info(f"Generation Time: {generation_time:.2f} seconds ({(generation_time/total_processing_time)*100:.1f}%)")
        self.logger.info(f"Total Tokens Used: {total_tokens_used}")
        self.logger.info("$"*50)
    
    def log_vector_store_status(self, 
                               index_size: int,
                               document_count: int,
                               last_updated: str) -> None:
        """Log vector store status information"""
        self.logger.info("&"*35)
        self.logger.info("VECTOR STORE STATUS")
        self.logger.info(f"Index Size: {index_size} vectors")
        self.logger.info(f"Document Count: {document_count}")
        self.logger.info(f"Last Updated: {last_updated}")
        self.logger.info("&"*35)
    
    def log_medical_query_specifics(self, 
                                   query_type: str,
                                   medical_entities: List[str],
                                   confidence_score: float) -> None:
        """Log medical-specific query analysis"""
        self.logger.info("%"*40)
        self.logger.info("MEDICAL QUERY ANALYSIS")
        self.logger.info(f"Query Type: {query_type}")
        self.logger.info(f"Medical Entities: {', '.join(medical_entities)}")
        self.logger.info(f"Confidence Score: {confidence_score:.3f}")
        self.logger.info("%"*40)
    
    def log_safety_check(self, 
                        query: str,
                        safety_status: str,
                        flagged_content: List[str] = None) -> None:
        """Log safety and content moderation checks"""
        self.logger.info("!"*35)
        self.logger.info("SAFETY CHECK")
        self.logger.info(f"Query: {query[:100]}...")
        self.logger.info(f"Status: {safety_status}")
        
        if flagged_content:
            self.logger.warning(f"Flagged Content: {', '.join(flagged_content)}")
        
        self.logger.info("!"*35)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message"""
        self.logger.critical(message)

# Convenience function to get a configured RAG logger instance
def get_rag_logger(name: str = "rag_system", 
                   log_file: str = "logs/rag_system.log",
                   level: int = logging.INFO) -> RAGSystemLogger:
    """
    Get a configured RAGSystemLogger instance
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        RAGSystemLogger instance
    """
    return RAGSystemLogger(name, log_file, level)