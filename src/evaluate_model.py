import json
import time
import sys
import os
import argparse
from typing import List, Dict
from llm_wrapper import LLMWrapper
from fuzzywuzzy import fuzz

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def evaluate_model(wrapper: LLMWrapper, json_file: str, threshold: int = 90):
    """Evaluate the model with a provided .json file containing questions and answers using fuzzy matching."""
    with open(json_file, "r") as f:
        data = json.load(f)
    
    correct_count = 0
    total_count = len(data)
    
    print("\n=== Model Evaluation ===")
    
    for entry in data:
        question = entry["question"]
        expected_answer = entry["answer"]
        
        # Generate response using the model
        start_time = time.time()
        model_response = wrapper.generate_text(question, task_type="general")  # or any other task type
        end_time = time.time()
        
        print(f"Question: {question}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Model Response: {model_response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print("=" * 50)
        
        # Fuzzy matching comparison
        similarity_score = fuzz.token_sort_ratio(model_response.strip(), expected_answer.strip())
        print(f"Similarity score: {similarity_score}%")

        # If similarity score is above the threshold, consider it correct
        if similarity_score >= threshold:
            correct_count += 1
    
    accuracy = correct_count / total_count * 100
    print(f"\nEvaluation complete. Accuracy: {accuracy:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Evaluate the LLM model using a .json file")
    # Default path is now set to "dev_data.json" in the current directory
    parser.add_argument("--json_file", "-f", default="dev_data.json", help="Path to the JSON file containing questions and answers")
    parser.add_argument("--threshold", "-t", type=int, default=90, help="Fuzzy matching similarity threshold (default is 90)")
    args = parser.parse_args()
    
    try:
        # Initialize the model
        print("Initializing model...")
        wrapper = LLMWrapper()
        
        # Run evaluation
        evaluate_model(wrapper, args.json_file, args.threshold)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
