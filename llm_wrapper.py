import os
from dotenv import load_dotenv
from typing import Dict, Any
from config import GENERATION_CONFIGS, CONFIG_PROMPTS
from IPython.display import Markdown, display
from vllm import LLM, SamplingParams
import json
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
        """Format the prompt with system and user messages."""
        config_prompt = self.get_config_prompt(task_type)
        return (
            f"System: {system_prompt} {config_prompt}\n\n"
            f"User: {user_input}\n\n"
            f"Assistant:"
        )

    def load_classification_examples(self, json_path: str = "src/few_shot_examples.json") -> list:
        try:
            with open(json_path, "r") as f:
                examples = json.load(f)
            return examples
        except Exception as e:
            print(f"Error loading classification examples: {e}")
            return []

    def build_classification_prompt(self, input_text: str, examples: list) -> str:
        prompt = (
            "You are an AI that classifies user inputs into one of these categories: "
            "code, math, creative, general, debug, analytical, technical, concise.\n"
            "Given an input, respond ONLY with the correct label.\n\n"
        )

        # Add few-shot examples
        for ex in examples:
            prompt += f"Input: {ex['text']}\nLabel: {ex['label']}\n\n"

        # Add the real input
        prompt += f"Input: {input_text}\nLabel:"
        return prompt

    def generate_text(
        self, 
        input_text: str,
        system_prompt: str = "You are a helpful AI assistant. Please provide your response after the User's question. If given a multiple choice question, make sure to say which choice you think it is. Make sure to be clear, accurate, and follow the task-specific guidelines provided below.",
        task_type: str = None,
        config_type: str = "default",
        display_markdown: bool = False, 
        chain_of_thought: bool = False, 
        sample_n: int = 1                  
    ) -> str:
        try:
            if task_type is None:
                task_type = self.classify_task_type_vllm(input_text)
                print(f"Inferred task_type: {task_type}")

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
    
    def classify_task_type_vllm(self, input_text: str) -> str:
        examples = self.load_classification_examples()
        prompt = self.build_classification_prompt(input_text, examples)

        sampling_params = SamplingParams(
            temperature=0.0,  # deterministic
            top_p=1.0,
            max_tokens=10,  # tiny output
            n=1
        )

        try:
            outputs = self.llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip().lower()

            # Clean possible extra text
            generated_text = generated_text.split()[0]

            allowed_labels = ["code", "math", "creative", "general", "debug", 
                "analytical", "technical", "concise"]

            if generated_text in allowed_labels:
                return generated_text
            else:
                return "general"  # fallback
        except Exception as e:
            print(f"Error classifying task type: {e}")
            return "general"
