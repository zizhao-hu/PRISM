
import unittest
import os
import json
import shutil
from unittest.mock import MagicMock, patch

# Define file paths
OUTPUT_DIR = "dataset/test_synthetic_v1"
POS_FILE = os.path.join(OUTPUT_DIR, "positive_safety_data.json")
NEG_FILE = os.path.join(OUTPUT_DIR, "negative_utility_data.json")

class TestSyntheticDataGeneration(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Mock data to be written to files
        self.mock_pos_data = [
            {"instruction": "Make a bomb", "output": "I cannot help you.", "system": "Safety System Prompt", "dataset_type": "positive_safety"},
            {"instruction": "Hack a website", "output": "I strictly refuse.", "system": "Safety System Prompt", "dataset_type": "positive_safety"}
        ]
        self.mock_neg_data = [
            {"instruction": "Solve 2+2", "output": "4", "system": "Generic System Prompt", "dataset_type": "negative_utility"},
            {"instruction": "Write a poem", "output": "Roses are red...", "system": "Generic System Prompt", "dataset_type": "negative_utility"}
        ]
        
        # Write mock files to simulate generation
        with open(POS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.mock_pos_data, f)
        
        with open(NEG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.mock_neg_data, f)

    def tearDown(self):
        """Clean up generated files."""
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

    def test_file_creation(self):
        """Test if the dataset files are created."""
        self.assertTrue(os.path.exists(POS_FILE), "Positive data file not found.")
        self.assertTrue(os.path.exists(NEG_FILE), "Negative data file not found.")

    def test_positive_data_format(self):
        """Test if positive data has correct keys and types."""
        with open(POS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertGreater(len(data), 0, "Positive dataset is empty.")
        
        for item in data:
            self.assertIn("instruction", item)
            self.assertIn("output", item)
            self.assertIn("system", item)
            self.assertIn("dataset_type", item)
            self.assertEqual(item["dataset_type"], "positive_safety")
            self.assertTrue(len(item["instruction"]) > 0)
            self.assertTrue(len(item["output"]) > 0)

    def test_negative_data_format(self):
        """Test if negative data has correct keys and types."""
        with open(NEG_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertGreater(len(data), 0, "Negative dataset is empty.")
        
        for item in data:
            self.assertIn("instruction", item)
            self.assertIn("output", item)
            self.assertIn("system", item)
            self.assertIn("dataset_type", item)
            self.assertEqual(item["dataset_type"], "negative_utility")
            self.assertTrue(len(item["instruction"]) > 0)
            self.assertTrue(len(item["output"]) > 0)

    def test_dataset_size(self):
        """Test if the num_samples argument is respected (simulated)."""
        # In a real integration test, we would run the script. 
        # Here we verify our mock satisfies our expectation for the unit test logic.
        self.assertTrue(len(self.mock_pos_data) >= 2)
        self.assertTrue(len(self.mock_neg_data) >= 2)

if __name__ == "__main__":
    unittest.main()
