import os
from dotenv import load_dotenv
from typing import Dict, Any
from config import GENERATION_CONFIGS, CONFIG_PROMPTS
from IPython.display import Markdown, display
from vllm import LLM, SamplingParams
from collections import Counter  # <-- move this to top

# Load environment variables from .env file
load_dotenv()

def should_use_chain_of_thought(input_text: str) -> bool:
    reasoning_keywords = [
        "calculate", "solve", "derive", "multiply", "add", "subtract", "divide",
        "what is the result", "evaluate", "step by step", "reason", "analyze",
        "impact", "compare", "contrast", "explain why", "how does", "how do",
        "debug", "fix", "proof", "determine", "find the value", "logical"
    ]
    input_text = input_text.lower()
    for keyword in reasoning_keywords:
        if keyword in input_text:
            return True
    return False

class LLMWrapper:
    def __init__(self, model_path: str = "/scratch/jjosep31/models/llama-3.2-3b-hf"):
        try:
            self.model_id = model_path
            self.HF_TOKEN = os.getenv("HF_TOKEN")
            if not self.HF_TOKEN:
                raise ValueError("HF_TOKEN environment variable is not set")

            print("Loading model with vLLM...")
            self.llm = LLM(
                model=self.model_id,
                dtype="float16",
                gpu_memory_utilization=0.95,
                max_model_len=4096,
                trust_remote_code=True,
                tokenizer=self.model_id,
                tokenizer_revision="main",
                download_dir="/scratch/jjosep31/models",
            )
            print(f"Model {model_path} loaded successfully with vLLM")

        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def get_generation_config(self, config_type: str = "default") -> Dict[str, Any]:
        return GENERATION_CONFIGS.get(config_type, GENERATION_CONFIGS["general"])

    def get_config_prompt(self, task_type: str = "general") -> str:
        return CONFIG_PROMPTS.get(task_type, CONFIG_PROMPTS["general"])

    def format_prompt(self, system_prompt: str, user_input: str, task_type: str = "general") -> str:
        config_prompt = self.get_config_prompt(task_type)
        return f"""System: {system_prompt} {config_prompt}

User: {user_input}

Assistant:"""

    def generate_text(
        self, 
        input_text: str,
        system_prompt: str = "You are a helpful AI assistant. Please provide your response after the User's question. Make sure to be clear, accurate, and follow the task-specific guidelines provided below.",
        task_type: str = "general",
        config_type: str = "default",
        display_markdown: bool = False, #added
        chain_of_thought: bool = False, #addded
        sample_n: int = 1                  
    ) -> str:
        try:
            # Automatically turn on CoT if not specified manually
            if not chain_of_thought:
                if should_use_chain_of_thought(input_text):
                    chain_of_thought = True

            # Format the prompt
            prompt = self.format_prompt(system_prompt, input_text, task_type)

            if chain_of_thought:
                prompt += "\nLet's think step-by-step."

            # Get generation config
            config = self.get_generation_config(config_type)
            sampling_params = SamplingParams(
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 0.95),
                max_tokens=config.get("max_new_tokens", 512),
                n=sample_n
            )

            # Generate multiple outputs if needed
            outputs = self.llm.generate([prompt], sampling_params)
            generated_texts = [output.text.strip() for output in outputs[0].outputs]

            # 🛠 Self-consistency voting
            if sample_n == 1:
                full_text = generated_texts[0]
            else:
                # Optionally clean responses to vote on
                cleaned_responses = [text.split("Answer:")[-1].strip() if "Answer:" in text else text for text in generated_texts]
                vote_counter = Counter(cleaned_responses)
                full_text = vote_counter.most_common(1)[0][0]  # Pick most common

            # Extract assistant's reply
            response_start = full_text.find("Assistant:") + len("Assistant:")
            response = full_text[response_start:].strip()

            # Clean up any leftover markers
            for marker in ["System:", "User:", "Assistant:"]:
                response = response.split(marker)[0].strip()

            if display_markdown:
                display(Markdown(response))

            return response

        except Exception as e:
            print(f"Error generating text: {e}")
            return "I apologize, but I encountered an error while generating the response."
