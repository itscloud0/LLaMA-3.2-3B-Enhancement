from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional, Union
import json
from data_loader import DataLoader

class LLMWrapper:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B", test_mode: bool = False):
        """Initialize the LLM wrapper with the specified model.
        
        Args:
            model_name (str): Name of the model to load
            test_mode (bool): If True, skip loading the model for testing
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize data loader
        self.data_loader = DataLoader()
        
        # Load development data
        self.dev_data = self.data_loader.load_dev_data()
        
        # Load few-shot examples and safety filters
        self.few_shot_examples = self.data_loader.load_few_shot_examples()
        self.safety_filters = self.data_loader.load_safety_filters()

        if not test_mode:
            # Initialize model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        else:
            self.tokenizer = None
            self.model = None
            self.device = "cpu"
        
    def _load_few_shot_examples(self) -> Dict[str, List[Dict[str, str]]]:
        """Load few-shot examples from JSON file."""
        try:
            with open("data/few_shot_examples.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
            
    def _load_safety_filters(self) -> List[str]:
        """Load safety filter patterns."""
        try:
            with open("data/safety_filters.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
            
    def _apply_safety_filters(self, text: str) -> str:
        """Apply safety filters to the generated text."""
        text_lower = text.lower()
        for filter_word in self.safety_filters:
            if filter_word in text_lower:
                # Replace potentially harmful content with a warning
                text = f"[Content filtered for safety: {filter_word}]"
                break
        return text
        
    def _get_relevant_examples(self, prompt: str) -> List[Dict[str, str]]:
        """Get relevant few-shot examples based on prompt topic."""
        # Simple keyword matching for now
        # TODO: Implement more sophisticated topic matching
        examples = []
        for topic, topic_examples in self.few_shot_examples.items():
            if topic.lower() in prompt.lower():
                examples.extend(topic_examples)
        return examples[:3]  # Limit to 3 examples
        
    def _format_prompt_with_examples(self, prompt: str) -> str:
        """Format the prompt with relevant few-shot examples."""
        # Find relevant examples based on prompt content
        relevant_examples = []
        for category, examples in self.few_shot_examples.items():
            for example in examples:
                if any(keyword in prompt.lower() for keyword in category.split("_")):
                    relevant_examples.append(example)
        
        if not relevant_examples:
            return prompt
        
        # Format examples
        formatted_examples = "\n\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in relevant_examples[:2]  # Use at most 2 examples
        ])
        
        return f"{formatted_examples}\n\nInput: {prompt}\nOutput:"
        
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        use_few_shot: bool = True,
        use_chain_of_thought: bool = False
    ) -> str:
        """Generate text from a prompt using the model."""
        if self.model is None or self.tokenizer is None:
            return "[Test Mode] Generated text would appear here"
            
        # Format prompt with few-shot examples if enabled
        if use_few_shot:
            prompt = self._format_prompt_with_examples(prompt)
        
        # Add chain-of-thought reasoning if enabled
        if use_chain_of_thought:
            prompt = f"Let's solve this step by step:\n{prompt}"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean up
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()  # Remove the prompt from response
        
        # Apply safety filters
        response = self._apply_safety_filters(response)
        
        return response 