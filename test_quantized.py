from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "/scratch/jjosep31/llama3-awq"

# Load quantized model + tokenizer
model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Inference
prompt = "Explain the significance of mitochondrial DNA in evolution."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
