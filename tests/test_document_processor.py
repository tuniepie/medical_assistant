import os
import unittest
from typing import List
from src.document_processor import (
    DocumentProcessor,
    supported_file_types,
    validate_file_type,
    get_file_info,
)

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

class TestDocumentProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        log_step("Setting up test class resources...")
        cls.text_file = "tests/example.txt"
        cls.pdf_file = "tests/example.pdf"
        cls.all_files = [cls.text_file, cls.pdf_file]
        cls.processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)

    def test_supported_file_types(self):
        log_step("Testing supported file types")
        types = supported_file_types()
        print(f"  - Supported types: {types}")
        self.assertIn(".txt", types)
        self.assertIn(".pdf", types)

    def test_validate_file_type(self):
        log_step("Testing file type validation")
        self.assertTrue(validate_file_type(self.text_file))
        self.assertTrue(validate_file_type(self.pdf_file))
        self.assertFalse(validate_file_type("invalid.docx"))

    def test_get_file_info(self):
        log_step("Testing file info extraction")
        info = get_file_info(self.text_file)
        print(f"  - Info: {info}")
        self.assertIn("name", info)
        self.assertIn("extension", info)

    def test_process_single_document(self):
        log_step("Testing single document processing")
        chunks = self.processor.process_document(self.text_file, source_name="unit_test")
        print(f"  - {len(chunks)} chunks from {self.text_file}")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks[:3]:
            self.assertTrue(len(chunk.page_content) > 0)
            self.assertIn("source", chunk.metadata)

    def test_process_multiple_documents(self):
        log_step("Testing multiple document processing")
        chunks = self.processor.process_multiple_documents(self.all_files)
        print(f"  - Total chunks: {len(chunks)}")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

    def test_document_statistics(self):
        log_step("Testing document statistics generation")
        chunks = self.processor.process_multiple_documents(self.all_files)
        stats = self.processor.get_document_stats(chunks)
        print(f"  - Stats: {stats}")
        self.assertIn("total_chunks", stats)
        self.assertIn("average_chunk_size", stats)
        self.assertIn("sources", stats)

    def test_chunk_filtering(self):
        log_step("Testing chunk filtering by length")
        chunks = self.processor.process_multiple_documents(self.all_files)
        filtered = self.processor.filter_chunks_by_length(chunks, min_length=30)
        print(f"  - Filtered from {len(chunks)} to {len(filtered)}")
        self.assertLessEqual(len(filtered), len(chunks))
        for chunk in filtered:
            self.assertGreaterEqual(len(chunk.page_content), 30)

    def test_text_preprocessing(self):
        log_step("Testing text preprocessing cleanup")
        dirty = "  \n\n  This   has    extra   spaces  \x00\ufeff  \n\n  "
        cleaned = self.processor.preprocess_text(dirty)
        print(f"  - Before: '{dirty}'\n  - After : '{cleaned}'")
        self.assertEqual(cleaned, "This has extra spaces")

    def test_chunk_settings_update(self):
        log_step("Testing dynamic chunk settings update")
        self.processor.update_chunk_settings(chunk_size=300, chunk_overlap=75)
        self.assertEqual(self.processor.chunk_size, 300)
        self.assertEqual(self.processor.chunk_overlap, 75)
        chunks = self.processor.process_document(self.text_file, source_name="updated")
        print(f"  - Chunks after update: {len(chunks)}")
        self.assertIsInstance(chunks, list)

class TestDocumentProcessorErrors(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor()

    def test_unsupported_file_type(self):
        log_step("Testing unsupported file type error")
        with self.assertRaises(ValueError):
            self.processor.load_document("invalid.docx")

    def test_nonexistent_file(self):
        log_step("Testing nonexistent file error")
        with self.assertRaises(Exception):
            self.processor.load_document("nonexistent_file.txt")

    def test_empty_stats(self):
        log_step("Testing empty chunk statistics")
        stats = self.processor.get_document_stats([])
        print(f"  - Stats for empty input: {stats}")
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["total_chunks"], 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
