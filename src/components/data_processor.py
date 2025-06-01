from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.logger import logger
from src.components.data_loader import DataLoader
import os

class DataProcessor:
    """Handles text preprocessing, chunking, and metadata tagging."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
        logger.info(f"DataProcessor initialized - chunk_size: {chunk_size}, overlap: {chunk_overlap}")

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise

    def process_document(self, file_path: str, source_name: Optional[str] = None) -> List[Document]:
        loader = DataLoader()
        documents = loader.load(file_path)

        source = source_name or os.path.basename(file_path)
        for doc in documents:
            doc.metadata['source'] = source
            doc.metadata['file_path'] = file_path

        chunks = self.chunk_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        logger.info(f"Processed {source}: {len(chunks)} chunks")
        return chunks

    def process_multiple_documents(self, file_paths: List[str]) -> List[Document]:
        all_chunks = []
        success = 0

        for path in file_paths:
            try:
                chunks = self.process_document(path)
                all_chunks.extend(chunks)
                success += 1
            except Exception as e:
                logger.error(f"Failed to process {path}: {str(e)}")

        logger.info(f"Processed {success}/{len(file_paths)} files")
        logger.info(f"Total chunks: {len(all_chunks)}")
        return all_chunks

    def preprocess_text(self, text: str) -> str:
        text = ' '.join(text.split())
        text = text.replace('\x00', '').replace('\ufeff', '')
        return text.strip()

    def get_document_stats(self, chunks: List[Document]) -> dict:
        if not chunks:
            return {"total_chunks": 0, "total_characters": 0, "sources": []}

        lengths = [len(chunk.page_content) for chunk in chunks]
        stats = {
            "total_chunks": len(chunks),
            "total_characters": sum(lengths),
            "average_chunk_size": sum(lengths) / len(chunks),
            "sources": list(set(chunk.metadata.get('source', 'Unknown') for chunk in chunks)),
            "chunk_size_distribution": {
                "min": min(lengths),
                "max": max(lengths),
                "median": sorted(lengths)[len(lengths) // 2],
            },
        }
        return stats

    def filter_chunks_by_length(self, chunks: List[Document], min_length: int = 50) -> List[Document]:
        original_count = len(chunks)
        filtered = [chunk for chunk in chunks if len(chunk.page_content.strip()) >= min_length]
        logger.info(f"Filtered: {original_count} -> {len(filtered)} chunks (removed {original_count - len(filtered)})")
        return filtered

    # def update_chunk_settings(self, chunk_size: int, chunk_overlap: int):
    #     self.chunk_size = chunk_size
    #     self.chunk_overlap = chunk_overlap
    #     self.text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=chunk_size,
    #         chunk_overlap=chunk_overlap,
    #         separators=["\n\n", "\n", " ", ""],
    #         length_function=len,
    #     )
    #     logger.info(f"Updated chunk settings - size: {chunk_size}, overlap: {chunk_overlap}")
