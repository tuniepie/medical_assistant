import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.logger import setup_logger

logger = setup_logger()

class DocumentProcessor:
    """Handles document ingestion and text chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        
        logger.info(f"DocumentProcessor initialized - chunk_size: {chunk_size}, overlap: {chunk_overlap}")
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from file path
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            logger.info(f"Loaded document: {file_path} with {len(documents)} pages/sections")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise
    
    def process_document(self, file_path: str, source_name: Optional[str] = None) -> List[Document]:
        """
        Complete document processing pipeline
        
        Args:
            file_path: Path to the document file
            source_name: Optional name to use as source identifier
            
        Returns:
            List of processed and chunked Document objects
        """
        try:
            # Load document
            documents = self.load_document(file_path)
            
            # Add source metadata
            source = source_name or os.path.basename(file_path)
            for doc in documents:
                doc.metadata['source'] = source
                doc.metadata['file_path'] = file_path
            
            # Chunk documents
            chunks = self.chunk_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['chunk_size'] = len(chunk.page_content)
            
            logger.info(f"Successfully processed {source}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def process_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Combined list of all processed chunks
        """
        all_chunks = []
        successful_files = 0
        
        for file_path in file_paths:
            try:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
                successful_files += 1
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue
        
        logger.info(f"Processed {successful_files}/{len(file_paths)} files successfully")
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        return all_chunks
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text content
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters that might cause issues
        text = text.replace('\x00', '')  # Remove null characters
        text = text.replace('\ufeff', '')  # Remove BOM
        
        return text.strip()
    
    def get_document_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about processed documents
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with document statistics
        """
        if not chunks:
            return {"total_chunks": 0, "total_characters": 0, "sources": []}
        
        stats = {
            "total_chunks": len(chunks),
            "total_characters": sum(len(chunk.page_content) for chunk in chunks),
            "average_chunk_size": sum(len(chunk.page_content) for chunk in chunks) / len(chunks),
            "sources": list(set(chunk.metadata.get('source', 'Unknown') for chunk in chunks)),
            "chunk_size_distribution": {
                "min": min(len(chunk.page_content) for chunk in chunks),
                "max": max(len(chunk.page_content) for chunk in chunks),
                "median": sorted([len(chunk.page_content) for chunk in chunks])[len(chunks)//2]
            }
        }
        
        return stats
    
    def filter_chunks_by_length(self, chunks: List[Document], min_length: int = 50) -> List[Document]:
        """
        Filter out chunks that are too short to be meaningful
        
        Args:
            chunks: List of document chunks
            min_length: Minimum chunk length to keep
            
        Returns:
            Filtered list of chunks
        """
        original_count = len(chunks)
        filtered_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) >= min_length]
        
        logger.info(f"Filtered chunks: {original_count} -> {len(filtered_chunks)} (removed {original_count - len(filtered_chunks)} short chunks)")
        
        return filtered_chunks
    
    def update_chunk_settings(self, chunk_size: int, chunk_overlap: int):
        """
        Update chunking parameters
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        
        logger.info(f"Updated chunk settings - size: {chunk_size}, overlap: {chunk_overlap}")

# Utility functions for document processing
def supported_file_types() -> List[str]:
    """Return list of supported file extensions"""
    return ['.pdf', '.txt']

def validate_file_type(file_path: str) -> bool:
    """
    Validate if file type is supported
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file type is supported
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in supported_file_types()

def get_file_info(file_path: str) -> dict:
    """
    Get basic information about a file
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    try:
        stat = os.stat(file_path)
        return {
            "name": os.path.basename(file_path),
            
            "size": stat.st_size,
            "extension": os.path.splitext(file_path)[1].lower(),
            "is_supported": validate_file_type(file_path)
        }
    except Exception as e:
        return {"error": str(e)}