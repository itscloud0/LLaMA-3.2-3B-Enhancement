from vllm import LLM, SamplingParams

class LLMWrapper:
    def __init__(self, model_path: str = "meta-llama/Llama-3.2-3B"):
        """Initialize the LLM wrapper with vLLM backend."""
        try:
            self.model_id = model_path

            print("Loading model with vLLM...")
            self.llm = LLM(
                model=model_path,
                dtype="half",
                max_model_len=4096,
                gpu_memory_utilization=0.90,
            )
            print(f"Model {model_path} loaded successfully with vLLM.")

            # Setup default sampling params
            self.sampling_params = SamplingParams(
                max_tokens=200,
                repetition_penalty=1.1,
                # You can add stop=["<|eos|>"] if you know your model uses that special token
            )

        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def generate_text(self, input_text: str) -> str:
        """Generate text response for the given input using vLLM."""
        try:
            outputs = self.llm.generate([input_text], self.sampling_params)

            # outputs is a list of RequestOutput objects
            if isinstance(outputs[0], dict):
                # If somehow it's a dict, access like this
                response = outputs[0]['outputs'][0]['text'].strip()
            else:
                # Normal case: vLLM object
                response = outputs[0].outputs[0].text.strip()

            # Optionally remove the input prompt if repeated
            if response.startswith(input_text):
                response = response[len(input_text):].strip()

            return response

        except Exception as e:
            print(f"Error generating text: {e}")
            return "I apologize, but I encountered an error while generating the response."
