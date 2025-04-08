from llm_wrapper import LLMWrapper
import time
import sys
import os
import argparse
from typing import List, Dict

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def infer_task_type_llm(query: str, wrapper: LLMWrapper) -> str:
    """Infers the task type using the LLM wrapper."""
    try:
        classification = wrapper.infer_task_type(query)
        return classification
    except Exception as e:
        print(f"Failed to classify with LLM: {e}")
        return "general"

def run_interactive_mode(wrapper: LLMWrapper):
    """Run the model in interactive mode with task type selection."""
    print("\nCommands:")
    print("1 - 'exit' or 'q' to quit")
    print("2 - 'clear' to clear the screen")
    print("3 - 'help' to show this menu and task types")
    print("======================\n")
    
    task_type = "general"
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
                print("\nAvailable task types:")
                print("1. general - General responses")
                print("2. code - Programming and code generation")
                print("3. math - Mathematical explanations")
                print("4. creative - Creative writing")
                print("5. technical - Technical explanations")
                print("6. concise - Brief responses")
                print("7. educational - Teaching and explanations")
                print("8. analytical - Analysis and problem-solving")
                print("9. debug - Debugging assistance")
                print("10. research - Research and academic responses")
                print("\nCommands:")
                print("1 - 'exit' or 'q' to quit")
                print("2 - 'clear' to clear the screen")
                print("3 - 'help' to show this menu and task types")
                print("======================\n")
                continue
                
            if not input_text:
                continue
                
            print("\nGenerating response...")
            start_time = time.time()
            auto_type = infer_task_type_llm(..., wrapper)
            print(f"Inferred task type: {auto_type}")
            response = wrapper.generate_text(input_text, task_type=auto_type)
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
    """Run predefined test prompts with different task types."""
    test_cases: List[Dict[str, str]] = [
        {
            "prompt": "What is Machine Learning?",
            "task_type": "technical"
        },
        {
            "prompt": "Write a Python function to sort a list",
            "task_type": "code"
        },
        {
            "prompt": "Solve the quadratic equation x^2 + 5x + 6 = 0",
            "task_type": "math"
        },
        {
            "prompt": "Write a short poem about coding",
            "task_type": "creative"
        },
        {
            "prompt": "What is the capital of France?",
            "task_type": "concise"
        },
        {
            "prompt": "Explain how quantum computing works",
            "task_type": "technical"
        },
        {
            "prompt": "Debug this Python code: print('Hello World'",
            "task_type": "debug"
        },
        {
            "prompt": "Analyze the impact of artificial intelligence on society",
            "task_type": "analytical"
        }
    ]
    
    print("\n=== Running Test Prompts ===")
    for test_case in test_cases:
        print(f"\nQuery: {test_case['prompt']}")
        print(f"Task type: {test_case['task_type']}")
        print("Generating response...")
        start_time = time.time()
        auto_type = infer_task_type_llm(test_case["prompt"], wrapper)
        print(f"Inferred task type: {auto_type}")
        response = wrapper.generate_text(
            test_case["prompt"],
            task_type=auto_type
        )
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