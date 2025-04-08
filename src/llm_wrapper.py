import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import json
import os
from dotenv import load_dotenv
from IPython.display import Markdown, display
from typing import Dict, Any, List
from config import GENERATION_CONFIGS, CONFIG_PROMPTS

# Load environment variables from .env file
load_dotenv()

class LLMWrapper:
    def __init__(self, model_path: str = "meta-llama/Llama-3.2-3B"):
        """Initialize the LLM wrapper with Llama-3.2-3B."""
        try:
            self.model_id = model_path
            self.HF_TOKEN = os.getenv("HF_TOKEN")
            if not self.HF_TOKEN:
                raise ValueError("HF_TOKEN environment variable is not set")
            
            print("Loading model...")
            # Load the model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,  
                return_dict=True,
                device_map="auto",
                token=self.HF_TOKEN,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=self.HF_TOKEN
            )
            
            # Configure tokenizer and model padding
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.model.config.eos_token_id
            
            # Initialize pipeline with default configuration
            self.pipe = self.configure_pipeline()
            self.classifier = self.configure_classifier()
            
            print(f"Model {model_path} loaded successfully")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def get_generation_config(self, config_type: str = "default") -> Dict[str, Any]:
        """Get predefined generation configurations."""
        return GENERATION_CONFIGS.get(config_type, GENERATION_CONFIGS["general"])

    def get_config_prompt(self, task_type: str = "general") -> str:
        """Get configuration prompt based on task type."""
        return CONFIG_PROMPTS.get(task_type, CONFIG_PROMPTS["general"])

    def configure_pipeline(self, config_type: str = "default"):
        """Configure the text generation pipeline with predefined parameters."""
        # Get the generation configuration for the specified type
        config = self.get_generation_config(config_type)
        
        # Create the pipeline with the configuration
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self.HF_TOKEN,
            trust_remote_code=True,
            **config
        )
    
    def configure_classifier(self):
        """Configure a text classification pipeline."""
        return pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            trust_remote_code=True,
        )

    def load_examples_from_json(self, filepath: str) -> List[Dict[str, str]]:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
   
     
    def format_prompt(self, system_prompt: str, user_input: str, task_type: str = "general") -> str:
        """Format the prompt with system and user messages."""
        config_prompt = self.get_config_prompt(task_type)
        return (
            f"System: {system_prompt} {config_prompt}\n\n"
            f"User: {user_input}\n\n"
            f"Assistant:"
        )

    def generate_text(
        self, 
        input_text: str,
        system_prompt: str = "You are a helpful AI assistant. Please provide your response after the User's question. Make sure to be clear, accurate, and follow the task-specific guidelines provided below.",
        task_type: str = "general",
        config_type: str = "default",
        display_markdown: bool = False
    ) -> str:
        """Generate text response for the given input."""
        try:
            # Format the prompt
            prompt = self.format_prompt(system_prompt, input_text, task_type)
            
            # Configure pipeline with custom parameters if needed
            if config_type != "default":
                self.pipe = self.configure_pipeline(config_type)
            
            # Generate response with stop sequences
            outputs = self.pipe(prompt)
            
            # Extract the response (everything after "Assistant:")
            full_text = outputs[0]["generated_text"]
            response_start = full_text.find("Assistant:") + len("Assistant:")
            response = full_text[response_start:].strip()
            
            # Remove any remaining role markers
            for marker in ["System:", "User:", "Assistant:"]:
                response = response.split(marker)[0].strip()
            
            # Display as markdown if requested
            if display_markdown:
                display(Markdown(response))
            
            return response
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return "I apologize, but I encountered an error while generating the response." 
        
    def infer_task_type(self, query: str) -> str:
        """Classify query using few-shot prompt-based learning with the language model."""
        
        # Few-shot examples
        examples = self.load_examples_from_json("src/few_shot_examples.json")
        
        # Build prompt
        prompt = "Classify the following text into one of the following categories:\n"
        prompt += "general, code, math, creative, concise, educational, analytical, debug, research, technical\n\n"
        
        for ex in examples:
            prompt += f"Text: {ex['text']}\nLabel: {ex['label']}\n\n"

        prompt += f"Text: {query}\nLabel:"
        
        try:
            result = self.classifier(prompt, max_new_tokens=10, return_full_text=False)
            prediction = result[0]["generated_text"].strip().split()[0]
            
            # Sanitize to known labels (optional)
            known_labels = [
                "general", "code", "math", "creative", "concise", 
                "educational", "analytical", "debug", "research", "technical"
            ]
            for label in known_labels:
                if prediction.lower().startswith(label.lower()):
                    return label
            return "general"
        
        except Exception as e:
            print(f"Error classifying task type: {e}")
            return "general"
