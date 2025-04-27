from llm_wrapper import LLMWrapper
import time
import sys
import os
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_interactive_mode(wrapper: LLMWrapper):
    """Run the model in interactive mode."""
    print("\n=== Interactive Mode ===")
    print("Commands:")
    print("- 'exit' or 'q' to quit")
    print("- 'clear' to clear the screen")
    print("- 'help' to show this help message")
    print("======================\n")
    
    while True:
        try:
            input_text = input("\nEnter your query: ").strip()
            
            if input_text.lower() in ['exit', 'quit', 'q']:
                print("\nExiting interactive mode...")
                break
            elif input_text.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif input_text.lower() == 'help':
                print("\nCommands:")
                print("- 'exit' or 'q' to quit")
                print("- 'clear' to clear the screen")
                print("- 'help' to show this help message")
                continue
                
            if not input_text:
                continue
                
            print(f"\nGenerating response...")
            start_time = time.time()
            response = wrapper.generate_text(input_text)
            end_time = time.time()
            
            print(f"\nResponse: {response}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

def run_test_prompts(wrapper: LLMWrapper):
    """Run predefined test prompts."""
    test_cases = [
        {"prompt": "What is Machine Learning?"},
        {"prompt": "Write a Python function to sort a list"},
        {"prompt": "Solve the quadratic equation x^2 + 5x + 6 = 0"},
        {"prompt": "Write a short poem about coding"},
        {"prompt": "What is the capital of France?"},
        {"prompt": "Explain how quantum computing works"},
        {"prompt": "Debug this Python code: print('Hello World'"},
        {"prompt": "Analyze the impact of artificial intelligence on society"}
    ]
    
    print("\n=== Running Test Prompts ===")
    for test_case in test_cases:
        print(f"\nQuery: {test_case['prompt']}")
        print("Generating response...")
        start_time = time.time()
        response = wrapper.generate_text(test_case["prompt"])
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
