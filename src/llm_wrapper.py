import json
import torch
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class LLMWrapper:
    def __init__(self):
        """Initialize the LLM wrapper with Llama-3.2-3B."""
        try:
            model_id = "meta-llama/Llama-3.2-3B"
            HF_TOKEN = os.getenv("HF_TOKEN")
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                print("Warning: CUDA is not available. Using CPU instead.")
                device = "cpu"
            else:
                device = "cuda"
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            
            # Initialize tokenizer and model separately for better control
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=HF_TOKEN,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=HF_TOKEN,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Initialize pipeline with better parameters
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            # Set improved generation parameters based on the tutorial
            self.generation_params = {
                "max_new_tokens": 500,  # Control output length directly
                #"num_return_sequences": 1,
                #"temperature": 0.3,  # Lower temperature for more focused responses
                #"top_p": 0.95,  # Higher top_p for better quality
                #"top_k": 50,  # Added top_k sampling
                "do_sample": True,
                "repetition_penalty": 1.2,  # Added to prevent repetition
                "length_penalty": 1.0,  # Added to control response length
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                #"truncation": True  # Added to handle long inputs
            }
            
            print(f"Model {model_id} loaded successfully")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def generate_text(self, prompt: str) -> str:
        """Generate text response for the given prompt."""
        try:
            # Format the prompt with better instruction tuning
            formatted_prompt = f"""You are a helpful AI assistant. 
            Please provide a clear, accurate, and detailed response to the following question or task.
            Question: {prompt}
            Answer: Let me help you with that."""
            
            # Generate response with parameters
            sequences = self.generator(
                formatted_prompt,
                **self.generation_params
            )
            
            # Extract the response (everything after "Answer:")
            gen_text = sequences[0]["generated_text"]
            response_start = gen_text.find("Answer:") + len("Answer:")
            response = gen_text[response_start:].strip()
            
            # Clean up the response
            response = response.replace("Let me help you with that.", "").strip()
            
            # If response is empty or just whitespace, return a default message
            if not response:
                return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                
            return response
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return "I apologize, but I encountered an error while generating the response." 