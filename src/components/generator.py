import streamlit as st
from typing import Optional
from langchain_cohere import ChatCohere
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_settings
from src.utils.logger import logger
    
class Generator:
    """Handles language model and prompt template for generation without using LLMChain."""

    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.llm
        self.temperature = self.settings.temperature
        self.max_tokens = self.settings.max_tokens
        self.cohere_api_key = None
        self.llm: BaseChatModel = self._load_llm()
        self.prompt = self._create_prompt_template()
        
        
    def _load_llm(self) -> BaseChatModel:
        if st.session_state.get("cohere_api_key"):
            self.cohere_api_key = st.session_state["cohere_api_key"]
        else:
            self.cohere_api_key = self.settings.cohere_api_key

        return ChatCohere(
            cohere_api_key=self.cohere_api_key,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _create_prompt_template(self) -> ChatPromptTemplate:
        system_message = self._get_system_prompt()
        human_message = """Context: {context}

        Question: {question}

        Answer:"""
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])

    def generate(self, query: str, context: str) -> str:
        formatted_prompt = self.prompt.format(context=context, question=query)
        response = self.llm.invoke(formatted_prompt)
        return response.content.strip()

    def stream(self, query: str, context: str):
        streaming_llm = ChatCohere(
            cohere_api_key=self.settings.cohere_api_key,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=True
        )
        formatted_prompt = self.prompt.format(context=context, question=query)
        for chunk in streaming_llm.stream(formatted_prompt):
            if chunk.content:
                yield chunk.content

    def update(self, model_name: Optional[str] = None,
               temperature: Optional[float] = None,
               max_tokens: Optional[int] = None):
        if model_name:
            self.model_name = model_name
        if temperature is not None:
            self.temperature = temperature
        if max_tokens:
            self.max_tokens = max_tokens
        self.llm = self._load_llm()
        logger.info("Generator updated with new configuration.")

    def get_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_template": str(self.prompt)
        }
    def _get_system_prompt(self) -> str:
        """Get system prompt for medical assistant"""
        return """
        You are a helpful medical assistant. Provide accurate, informative responses 
        based on the given context. Always:

        1. Base your answers on the provided context
        2. Be clear and professional
        3. Include disclaimers when appropriate
        4. Suggest consulting healthcare professionals for serious concerns
        5. Cite sources when possible
        """