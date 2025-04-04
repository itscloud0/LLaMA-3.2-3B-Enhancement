import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
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

    def format_prompt(self, system_prompt: str, user_input: str, task_type: str = "general") -> str:
        """Format the prompt with system and user messages."""
        config_prompt = self.get_config_prompt(task_type)
        return f"""[System Message]: {system_prompt} {config_prompt}

[User Message]: {user_input}

[Assistant Message]:"""

    def generate_text(
        self, 
        input_text: str,
        system_prompt: str = "You are a helpful AI assistant. Please provide your response to the [User Message] after the [Assistant Message] tag. Make sure to be clear, accurate, and follow the task-specific guidelines provided below. After you provide your response, you should stop generating.",
        task_type: str = "general",
        config_type: str = "default",
        display_markdown: bool = False
    ) -> str:
        """Generate text response for the given input."""
        try:
            # Format the prompt
            prompt = self.format_prompt(system_prompt, input_text, task_type)
            
            # Configure pipeline with the specified config type
            self.pipe = self.configure_pipeline(config_type)
            
            # Generate response using the configured pipeline
            outputs = self.pipe(prompt)
            
            # Extract the response (everything after "Response:")
            full_text = outputs[0]["generated_text"]
            response_start = full_text.find("[Assistant Message]:") + len("[Assistant Message]:")
            response = full_text[response_start:].strip()
            
            # Remove any system prompts or user questions that might have been repeated
            response = response.split("System:")[0].split("User's Question:")[0].strip()
            
            # Display as markdown if requested
            if display_markdown:
                display(Markdown(response))
            
            return response
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return "I apologize, but I encountered an error while generating the response." 