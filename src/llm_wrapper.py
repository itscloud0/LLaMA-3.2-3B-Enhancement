import json
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

class LLMWrapper:
    def __init__(self):
        """Initialize the LLM wrapper with Llama-3.2-3B."""
        try:
            model_name = "meta-llama/Llama-3.2-3B"
            HF_TOKEN = os.getenv("HF_TOKEN")
            
            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=HF_TOKEN
            )
            
            # Set padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=bnb_config,
                token=HF_TOKEN
            )
            
            # Initialize pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=128
            )
            
            print(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def generate_text(self, prompt: str) -> str:
        """Generate text response for the given prompt."""
        try:
            sequences = self.generator(prompt)
            gen_text = sequences[0]["generated_text"]
            # Remove the prompt from the response
            response = gen_text[len(prompt):].strip()
            return response
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return "I apologize, but I encountered an error while generating the response." 