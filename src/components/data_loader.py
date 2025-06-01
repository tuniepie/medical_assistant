import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from src.utils.logger import logger



class DataLoader:
    """Handles loading documents from different file formats"""
    
    def load(self, file_path: str) -> List[Document]:
        """Load a single document"""
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

    def load_multiple(self, file_paths: List[str]) -> List[Document]:
        """Load multiple documents"""
        all_docs = []
        for path in file_paths:
            try:
                docs = self.load(path)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {path}: {str(e)}")
        return all_docs

    def supported_file_types(self) -> List[str]:
        return ['.pdf', '.txt']

    def validate_file_type(self, file_path: str) -> bool:
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in self.supported_file_types()

    def get_file_info(self, file_path: str) -> dict:
        try:
            stat = os.stat(file_path)
            return {
                "name": os.path.basename(file_path),
                "size": stat.st_size,
                "extension": os.path.splitext(file_path)[1].lower(),
                "is_supported": self.validate_file_type(file_path)
            }
        except Exception as e:
            return {"error": str(e)}
