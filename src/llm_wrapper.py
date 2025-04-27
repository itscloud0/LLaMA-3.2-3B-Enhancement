import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LLMWrapper:
    def __init__(self, model_path: str = "meta-llama/Llama-3.2-3B"):
        """Initialize the LLM wrapper with Llama model."""
        try:
            self.model_id = model_path
            
            print("Loading model...")
            # Load the model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,  
                return_dict=True,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Initialize pipeline for text generation
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(f"Model {model_path} loaded successfully")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def generate_text(self, input_text: str) -> str:
        """Generate text response for the given input."""
        try:
            # Generate response using the pipeline
            outputs = self.pipe(input_text)
            response = outputs[0]["generated_text"]
            return response
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return "I apologize, but I encountered an error while generating the response."
