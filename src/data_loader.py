import json
import os
from typing import Dict, Any, List

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.dev_data_path = os.path.join(data_dir, "dev_data.json")
        self.few_shot_examples_path = os.path.join(data_dir, "few_shot_examples.json")
        self.safety_filters_path = os.path.join(data_dir, "safety_filters.json")

    def load_dev_data(self) -> List[Dict[str, Any]]:
        """Load the development data from dev_data.json."""
        try:
            with open(self.dev_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: dev_data.json not found at {self.dev_data_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.dev_data_path}")
            return []

    def load_few_shot_examples(self) -> Dict[str, List[Dict[str, str]]]:
        """Load few-shot examples from few_shot_examples.json."""
        try:
            with open(self.few_shot_examples_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: few_shot_examples.json not found at {self.few_shot_examples_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.few_shot_examples_path}")
            return {}

    def load_safety_filters(self) -> List[str]:
        """Load safety filters from safety_filters.json."""
        try:
            with open(self.safety_filters_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: safety_filters.json not found at {self.safety_filters_path}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.safety_filters_path}")
            return [] 