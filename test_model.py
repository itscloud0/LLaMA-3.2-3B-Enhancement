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
        classification = wrapper.classify_task_type_vllm(query)
        
        # Ensure classification is one of the valid task types
        valid_task_types = [
            "general", "code", "math", "creative", "technical", 
            "concise", "educational", "analytical", "debug", "research"
        ]
        
        if classification in valid_task_types:
            return classification
        else:
            print(f"Invalid classification: {classification}. Falling back to 'general'.")
            return "general"
    
    except Exception as e:
        print(f"Failed to classify with LLM: {e}")
        return "general"

def run_interactive_mode(wrapper: LLMWrapper, force_cot: bool = False, sample_n: int = 1):
    """Run the model in interactive mode with task type selection."""
    print("\n=== Interactive Mode ===")
    print("Available task types:")
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
    print("- 'exit' or 'q' to quit")
    print("- 'clear' to clear the screen")
    print("- 'help' to show this help message")
    print("- 'task <type>' to switch task type (e.g., 'task code')")
    print("======================\n")
    
    if force_cot:
        print("[Chain-of-Thought forcing ENABLED for all queries]")
    else:
        print("[Normal mode: Chain-of-Thought auto-detect only]")

    if sample_n > 1:
        print(f"[Self-Consistency: Sampling {sample_n} outputs and majority voting]")

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
                print("\nAvailable task types...")
                continue
            elif input_text.lower().startswith('task '):
                new_task = input_text[5:].strip().lower()
                if new_task in ['general', 'code', 'math', 'creative', 'technical', 
                                'concise', 'educational', 'analytical', 'debug', 'research']:
                    task_type = new_task
                    print(f"\nTask type set to: {task_type}")
                else:
                    print("\nInvalid task type.")
                continue
                
            if not input_text:
                continue
                
            print(f"\nGenerating response (task type: {task_type})...")
            start_time = time.time()
            auto_type = infer_task_type_llm(input_text, wrapper)
            print(f"Inferred task type: {auto_type}")

            response = wrapper.generate_text(
                input_text,
                task_type=auto_type,
                chain_of_thought=force_cot,
                sample_n=sample_n
            )
            end_time = time.time()
            

            print(f"\nResponse: {response}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

def run_test_prompts(wrapper: LLMWrapper, force_cot: bool = False, sample_n: int = 1):
    """Run predefined test prompts with different task types."""
    test_cases: List[Dict[str, str]] = [
        {"prompt": "What is Machine Learning?", "task_type": "technical"},
        {"prompt": "Write a Python function to sort a list", "task_type": "code"},
        {"prompt": "Solve the quadratic equation x^2 + 5x + 6 = 0", "task_type": "math"},
        {"prompt": "Write a short poem about coding", "task_type": "creative"},
        {"prompt": "What is the capital of France?", "task_type": "concise"},
        {"prompt": "Explain how quantum computing works", "task_type": "technical"},
        {"prompt": "Debug this Python code: print('Hello World'", "task_type": "debug"},
        {"prompt": "Analyze the impact of artificial intelligence on society", "task_type": "analytical"}
    ]
    
    print("\n=== Running Test Prompts ===")
    for test_case in test_cases:
        print(f"\nQuery: {test_case['prompt']}")
        print(f"Task type: {test_case['task_type']}")
        print("Generating response...")
        start_time = time.time()
        response = wrapper.generate_text(
            test_case["prompt"],
            task_type=test_case["task_type"],
            chain_of_thought=force_cot,
            sample_n=sample_n
        )
        end_time = time.time()
        
        print(f"Response: {response}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Test the LLM model")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("-cot", action="store_true", help="Force Chain-of-Thought prompting")
    parser.add_argument("-s", "--sample-n", type=int, default=1, help="Number of samples for self-consistency voting")
    args = parser.parse_args()
    
    try:
        print("Initializing model...")
        wrapper = LLMWrapper()
        
        if args.interactive:
            run_interactive_mode(wrapper, force_cot=args.cot, sample_n=args.sample_n)
        else:
            run_test_prompts(wrapper, force_cot=args.cot, sample_n=args.sample_n)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
