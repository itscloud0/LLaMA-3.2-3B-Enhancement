from llm_wrapper import LLMWrapper
import time
import sys
import os
import json
from typing import List, Dict, Any
import argparse
import torch

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model(
    test_mode: bool = False,
    config_name: str = None,
    category: str = None,
    save_results: bool = True
):
    """
    Test the LLM wrapper with various prompts and configurations.
    
    Args:
        test_mode (bool): If True, run in test mode without loading the model
        config_name (str): If provided, only run this specific configuration
        category (str): If provided, only run test cases from this category
        save_results (bool): If True, save results to a JSON file
    """
    try:
        # Initialize with memory optimization
        wrapper = LLMWrapper(
            model_name="meta-llama/Llama-3.2-3B",
            device_map="auto",  # Automatically handle model placement
            torch_dtype=torch.float16,  # Use half precision
            low_cpu_mem_usage=True  # Optimize memory usage
        )
        
        # Test prompt with explicit format request
        prompt = "Please provide a direct answer: What is the capital of France?"
        print(f"\nPrompt: {prompt}")
        
        # Generate response
        response = wrapper.generate_text(
            prompt,
            max_length=100,
            temperature=0.7,
            top_k=50
        )
        
        print(f"\nResponse: {response}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the LLM wrapper")
    parser.add_argument("--config", type=str, default="basic", help="Configuration name")
    parser.add_argument("--category", type=str, default="general_knowledge", help="Test category")
    
    args = parser.parse_args()
    test_model(
        test_mode=False,
        config_name=args.config,
        category=args.category,
        save_results=True
    ) 