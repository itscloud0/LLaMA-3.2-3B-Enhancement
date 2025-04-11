
# import os
# import torch
# from dotenv import load_dotenv
# from typing import Dict, Any
# from vllm import LLM, SamplingParams
# from config import GENERATION_CONFIGS, CONFIG_PROMPTS
# from IPython.display import Markdown, display

# # Load environment variables
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
# print("HF_TOKEN loaded?", os.getenv("HF_TOKEN") is not None)

# class LLMWrapper:
#     def __init__(self, model_path: str = "/home/jjosep31/FinalProject-main/src/llama3"):
#         """Initialize the LLM wrapper using vLLM with dynamic max_model_len."""
#         try:
#             self.model_path = model_path
#             self.HF_TOKEN = os.getenv("HF_TOKEN")

#             # Detect available GPU memory and pick max_model_len
#             total_mem, free_mem = torch.cuda.mem_get_info()
#             available_gb = free_mem / 1024**3

#             if available_gb >= 12:
#                 max_len = 8192
#             elif available_gb >= 8:
#                 max_len = 4096
#             else:
#                 max_len = 2048

#             print(f"Detected {available_gb:.2f} GB free VRAM. Using max_model_len = {max_len}")

#             self.llm = LLM(
#                 model=self.model_path,
#                 max_model_len=max_len,
#                 gpu_memory_utilization=0.9
#             )

#             # Default sampling config
#             self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

#             print(f"Model {model_path} loaded successfully with vLLM.")
#         except Exception as e:
#             print(f"Error initializing vLLM model: {e}")
#             raise

#     def get_generation_config(self, config_type: str = "default") -> Dict[str, Any]:
#         return GENERATION_CONFIGS.get(config_type, GENERATION_CONFIGS["general"])

#     def get_config_prompt(self, task_type: str = "general") -> str:
#         return CONFIG_PROMPTS.get(task_type, CONFIG_PROMPTS["general"])

#     def format_prompt(self, system_prompt: str, user_input: str, task_type: str = "general") -> str:
#         config_prompt = self.get_config_prompt(task_type)
#         return f"""System: {system_prompt} {config_prompt}

# User: {user_input}

# Assistant:"""

#     def generate_text(
#         self,
#         input_text: str,
#         system_prompt: str = "You are a helpful AI assistant. Please provide your response after the User's question. Make sure to be clear, accurate, and follow the task-specific guidelines provided below.",
#         task_type: str = "general",
#         config_type: str = "default",
#         display_markdown: bool = False
#     ) -> str:
#         try:
#             prompt = self.format_prompt(system_prompt, input_text, task_type)
#             outputs = self.llm.generate(prompt, self.sampling_params)
#             full_text = outputs[0].outputs[0].text

#             response_start = full_text.find("Assistant:") + len("Assistant:")
#             response = full_text[response_start:].strip()

#             for marker in ["System:", "User:", "Assistant:"]:
#                 response = response.split(marker)[0].strip()

#             if display_markdown:
#                 display(Markdown(response))

#             return response
#         except Exception as e:
#             print(f"Error generating text: {e}")
#             return "I apologize, but I encountered an error while generating the response."
import os
import torch
from dotenv import load_dotenv
from typing import Dict, Any
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from config import GENERATION_CONFIGS, CONFIG_PROMPTS
from IPython.display import Markdown, display

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
print("HF_TOKEN loaded?", os.getenv("HF_TOKEN") is not None)

class LLMWrapper:
    def __init__(self, model_path: str = "/scratch/jjosep31/llama3-awq"):
        """Initialize the LLM wrapper using a quantized AWQ model."""
        try:
            self.model_path = model_path
            self.HF_TOKEN = os.getenv("HF_TOKEN")

            # Load the quantized model and tokenizer
            self.model = AutoAWQForCausalLM.from_quantized(
                self.model_path,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

            print(f"Quantized model {model_path} loaded successfully with AWQ.")
        except Exception as e:
            print(f"Error initializing AWQ model: {e}")
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
        display_markdown: bool = False
    ) -> str:
        try:
            prompt = self.format_prompt(system_prompt, input_text, task_type)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            outputs = self.model.generate(input_ids, max_new_tokens=512, do_sample=True, temperature=0.8, top_p=0.95)
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            response_start = full_text.find("Assistant:") + len("Assistant:")
            response = full_text[response_start:].strip()

            for marker in ["System:", "User:", "Assistant:"]:
                response = response.split(marker)[0].strip()

            if display_markdown:
                display(Markdown(response))

            return response
        except Exception as e:
            print(f"Error generating text: {e}")
            return "I apologize, but I encountered an error while generating the response."
