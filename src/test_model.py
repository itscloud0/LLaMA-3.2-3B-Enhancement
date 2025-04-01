from llm_wrapper import LLMWrapper
import time
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # Initialize the model
    wrapper = LLMWrapper()
    
    # Test prompts
    prompts = [
        "What is Machine Learning?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about coding.",
        "What is the capital of France?",
        "Why the sky is blue?",
        "What is 2*2-2*3+6/3+1/1"
    ]
    
    # Run tests
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        start_time = time.time()
        response = wrapper.generate_text(prompt)
        end_time = time.time()
        
        print(f"Response: {response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 