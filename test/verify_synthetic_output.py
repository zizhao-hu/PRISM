
import unittest
import os
import json
import logging

# Define file paths based on the generation run
OUTPUT_DIR = "dataset/test_gen"
POS_FILE = os.path.join(OUTPUT_DIR, "positive_safety_data.json")
NEG_FILE = os.path.join(OUTPUT_DIR, "negative_utility_data.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestGeneratedData(unittest.TestCase):

    def test_files_exist(self):
        """Test if the dataset files were actually created."""
        logger.info(f"Checking for files in {OUTPUT_DIR}...")
        self.assertTrue(os.path.exists(POS_FILE), f"Positive data file not found at {POS_FILE}")
        self.assertTrue(os.path.exists(NEG_FILE), f"Negative data file not found at {NEG_FILE}")

    def test_positive_data_content(self):
        """Test the content of the positive dataset."""
        if not os.path.exists(POS_FILE):
            self.skipTest("Positive file missing")
            
        with open(POS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertGreater(len(data), 0, "Positive dataset is empty.")
        logger.info(f"Positive dataset size: {len(data)}")
        
        for item in data:
            self.assertIn("instruction", item)
            self.assertIn("output", item)
            self.assertIn("system", item)
            self.assertEqual(item["dataset_type"], "positive_safety")
            # Verify basic content
            self.assertIsInstance(item["instruction"], str)
            self.assertIsInstance(item["output"], str)

    def test_negative_data_content(self):
        """Test the content of the negative dataset."""
        if not os.path.exists(NEG_FILE):
            self.skipTest("Negative file missing")
            
        with open(NEG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.assertGreater(len(data), 0, "Negative dataset is empty.")
        logger.info(f"Negative dataset size: {len(data)}")
        
        for item in data:
            self.assertIn("instruction", item)
            self.assertIn("output", item)
            self.assertIn("system", item)
            self.assertEqual(item["dataset_type"], "negative_utility")
            # Verify basic content
            self.assertIsInstance(item["instruction"], str)
            self.assertIsInstance(item["output"], str)

if __name__ == "__main__":
    unittest.main()
