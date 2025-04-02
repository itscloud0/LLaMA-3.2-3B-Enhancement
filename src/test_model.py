from llm_wrapper import LLMWrapper
import time
import sys
import os
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_interactive_mode(wrapper):
    """Run the model in interactive mode."""
    print("\n=== Interactive Mode ===")
    print("Type 'exit' to quit")
    print("======================\n")
    
    while True:
        prompt = input("\nEnter your prompt: ").strip()
        if prompt.lower() in ['exit', 'quit', 'q']:
            print("\nExiting interactive mode...")
            break
        if not prompt:
            continue
            
        print("\nGenerating response...")
        start_time = time.time()
        response = wrapper.generate_text(prompt)
        end_time = time.time()
        
        print(f"\nResponse: {response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")

def run_test_prompts(wrapper):
    """Run predefined test prompts."""
    prompts = [
        "What is Machine Learning?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about coding.",
        "What is the capital of France?",
        "Why the sky is blue?"
    ]
    
    print("\n=== Running Test Prompts ===")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Generating response...")
        start_time = time.time()
        response = wrapper.generate_text(prompt)
        end_time = time.time()
        
        print(f"Response: {response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Test the LLM model")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    try:
        # Initialize the model
        print("Initializing model...")
        wrapper = LLMWrapper()
        
        if args.interactive:
            run_interactive_mode(wrapper)
        else:
            run_test_prompts(wrapper)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main() 