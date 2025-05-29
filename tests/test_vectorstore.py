import os
import shutil
import unittest
import numpy as np
from langchain_core.documents import Document
from src.vector_store import VectorStore

# ===== Utility for color logs =====
def color_text(text, color="cyan"):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "reset": "\033[0m"
    }
    return f"{colors[color]}{text}{colors['reset']}"

def log_step(message: str):
    print(color_text(f"[TEST] {message}", "cyan"))


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        # Use a separate store path to isolate test data
        self.test_store_path = "data/vector_db_test"
        self.vector_store = VectorStore()
        self.vector_store.store_path = self.test_store_path

        self.mock_documents = [
            Document(page_content="This is a test document about AI.", metadata={"source": "test1"}),
            Document(page_content="Another document related to machine learning.", metadata={"source": "test2"}),
            Document(page_content="This content talks about data science.", metadata={"source": "test3"})
        ]
        self.query = "Tell me about AI"

    def tearDown(self):
        # Clean up the test vector store after each test
        if os.path.exists(self.test_store_path):
            shutil.rmtree(self.test_store_path)

    def test_create_vector_store(self):
        log_step("Creating vector store with mock documents")
        self.vector_store.create_vector_store(self.mock_documents)
        self.assertEqual(self.vector_store.get_document_count(), 3)

    def test_add_documents(self):
        log_step("Adding documents to existing vector store")
        self.vector_store.create_vector_store(self.mock_documents[:1])
        self.vector_store.add_documents(self.mock_documents[1:])
        self.assertEqual(self.vector_store.get_document_count(), 3)

    def test_similarity_search(self):
        log_step("Performing similarity search")
        self.vector_store.create_vector_store(self.mock_documents)
        results = self.vector_store.similarity_search(self.query, k=2)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], tuple)
        self.assertIsInstance(results[0][0], Document)
        self.assertIsInstance(results[0][1], np.float32)

    def test_similarity_search_with_threshold(self):
        log_step("Performing similarity search with threshold")
        self.vector_store.create_vector_store(self.mock_documents)
        results = self.vector_store.similarity_search_with_threshold(self.query, k=2, threshold=1.0)
        self.assertLessEqual(len(results), 2)
        for _, score in results:
            self.assertLessEqual(score, 1.0)

    def test_get_relevant_context(self):
        log_step("Getting relevant context for query")
        self.vector_store.create_vector_store(self.mock_documents)
        context = self.vector_store.get_relevant_context(self.query, k=2)
        self.assertIsInstance(context, str)
        self.assertIn("[Source:", context)

    def test_save_and_load_vector_store(self):
        log_step("Saving and loading vector store")
        self.vector_store.create_vector_store(self.mock_documents)
        self.vector_store.save_vector_store()

        new_vs = VectorStore()
        new_vs.store_path = self.test_store_path
        new_vs.load_vector_store()
        self.assertEqual(new_vs.get_document_count(), 3)

    def test_delete_vector_store(self):
        log_step("Deleting vector store")
        self.vector_store.create_vector_store(self.mock_documents)
        self.vector_store.delete_vector_store()
        self.assertEqual(self.vector_store.get_document_count(), 0)
        self.assertFalse(os.path.exists(os.path.join(self.test_store_path, "faiss_index")))

    def test_get_store_info(self):
        log_step("Getting vector store info")
        self.vector_store.create_vector_store(self.mock_documents)
        info = self.vector_store.get_store_info()
        self.assertEqual(info["status"], "ready")
        self.assertEqual(info["document_count"], 3)
        self.assertIn("sources", info)
        self.assertGreater(info["total_characters"], 0)

    def test_search_by_metadata(self):
        log_step("Searching documents by metadata")
        self.vector_store.create_vector_store(self.mock_documents)
        results = self.vector_store.search_by_metadata({"source": "test2"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["source"], "test2")

    def test_get_embedding_dimension(self):
        log_step("Getting embedding dimension")
        dim = self.vector_store.get_embedding_dimension()
        self.assertIsInstance(dim, int)
        self.assertGreater(dim, 0)

    # def test_update_embedding_model(self):
    #     print("[TEST] Updating embedding model")
    #     self.vector_store.create_vector_store(self.mock_documents)
    #     original_dim = self.vector_store.get_embedding_dimension()
    #     self.vector_store.update_embedding_model("embed-english-v3.0")
    #     updated_dim = self.vector_store.get_embedding_dimension()
    #     self.assertEqual(original_dim, updated_dim)  # Because model is the same


if __name__ == '__main__':
    unittest.main(verbosity=2)
