import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.rag_pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = RAGPipeline()
        self.query = "What are symptoms of flu?"
        self.mock_docs = [
            (Document(page_content="Fever and chills are common symptoms.", metadata={"source": "med1"}), 0.9),
            (Document(page_content="Flu often includes coughing and fatigue.", metadata={"source": "med2"}), 0.85)
        ]

        self.mock_vector_store = MagicMock()
        self.mock_vector_store.similarity_search.return_value = self.mock_docs

    @patch("src.rag_pipeline.ChatCohere")
    def test_generate_response_success(self, MockChatCohere):
        mock_llm_instance = MockChatCohere.return_value
        mock_chain = MagicMock()
        mock_chain.run.return_value = "You may experience fever and fatigue. [Source: med1]"
        self.pipeline.chain = mock_chain

        result = self.pipeline.generate_response(self.query, self.mock_vector_store)

        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("metadata", result)
        self.assertGreater(result["metadata"]["retrieved_docs"], 0)
        self.assertIn("fever", result["answer"].lower())

    def test_generate_response_no_results(self):
        self.mock_vector_store.similarity_search.return_value = []
        result = self.pipeline.generate_response(self.query, self.mock_vector_store)
        self.assertEqual(result["answer"], "I couldn't find any relevant information...")
        self.assertEqual(len(result["sources"]), 0)

    def test_prepare_context_formatting(self):
        context = self.pipeline._prepare_context(self.mock_docs)
        self.assertIn("Document 1", context)
        self.assertIn("Source: med1", context)
        self.assertIn("Fever and chills", context)

    def test_evaluate_response_quality(self):
        response = "Consult your doctor for more information. Fever is a symptom. [Source: med1]"
        quality = self.pipeline.evaluate_response_quality(self.query, response, self.mock_docs)
        self.assertIn("quality_score", quality)
        self.assertGreaterEqual(quality["quality_score"], 0)

    # @patch("src.rag_pipeline.ChatCohere")
    # def test_update_settings(self, MockChatCohere):
    #     self.pipeline.update_settings(model_name="command-r", temperature=0.5, max_tokens=500)
    #     self.assertEqual(self.pipeline.model_name, "command-r")
    #     self.assertEqual(self.pipeline.temperature, 0.5)
    #     self.assertEqual(self.pipeline.max_tokens, 500)

    def test_get_pipeline_info(self):
        info = self.pipeline.get_pipeline_info()
        self.assertEqual(info["model_name"], self.pipeline.model_name)
        self.assertEqual(info["status"], "ready")

    @patch("src.rag_pipeline.ChatCohere")
    def test_batch_generate_responses(self, MockChatCohere):
        mock_llm_instance = MockChatCohere.return_value
        self.pipeline.chain.invoke = MagicMock(return_value="Mocked answer")

        queries = ["What is flu?", "What is COVID?"]
        results = self.pipeline.batch_generate_responses(queries, self.mock_vector_store)
        self.assertEqual(len(results), 2)
        self.assertIn("answer", results[0])

    @patch("src.rag_pipeline.ChatCohere")
    def test_generate_streaming_response(self, MockChatCohere):
        mock_llm_instance = MockChatCohere.return_value
        mock_llm_instance.stream.return_value = [MagicMock(content="Flu "), MagicMock(content="symptoms.")]
        chunks = list(self.pipeline.generate_streaming_response(self.query, self.mock_vector_store))
        self.assertGreater(len(chunks), 0)
        self.assertIn("Flu", "".join(chunks))


if __name__ == "__main__":
    unittest.main()
