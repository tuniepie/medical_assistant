from langchain_cohere import CohereEmbeddings
from config.settings import get_settings
from src.utils.logger import setup_logger

logger = setup_logger()

class EmbeddingModel:
    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.embedding_model_name
        self.embeddings = CohereEmbeddings(
            cohere_api_key=self.settings.cohere_api_key,
            model=self.model_name
        )
        logger.info(f"Cohere embeddings initialized with model: {self.model_name}")
        

    def get(self):
        return self.embeddings

    def embed_query(self, query: str):
        return self.embeddings.embed_query(query)
