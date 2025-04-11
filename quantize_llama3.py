
# from awq import AutoAWQForCausalLM
# from transformers import AutoTokenizer

# model_path = "meta-llama/Llama-3.2-3B"
# save_dir = "/scratch/jjosep31/llama3-awq" # indepedent per pesron
# calib_file = "calib.txt"

# # Load model and tokenizer
# model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# # Load calibration samples
# with open(calib_file, "r") as f:
#     calib_data = [line.strip() for line in f if line.strip()]
#     calib_data = calib_data * 20  # Duplicate to ensure sufficient tokens
#     print("Loaded calibration samples:", len(calib_data))
#     print("Sample preview:", calib_data[0][:100])
    

# #Inject tokenizer manually via kwarg
# model.quantize(
#     quant_config={
#         "zero_point": True,
#         "q_group_size": 128,
#         "w_bit": 4,
#         "version": "GEMM"
#     },
#     calib_data=calib_data,
#     tokenizer=tokenizer  
# )

# # Save the quantized model
# model.save_quantized(save_dir, safetensors=True)
# tokenizer.save_pretrained(save_dir)

# print(f"Quantized model saved to {save_dir}")

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "./llama3"
save_dir = "/scratch/jjosep31/llama3-awq"
calib_file = "calib.txt"

# Load model and tokenizer locally only
model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

# Load calibration data
with open(calib_file, "r") as f:
    calib_data = [line.strip() for line in f if line.strip()]
    calib_data = calib_data * 20
    print("Loaded calibration samples:", len(calib_data))
    print("Sample preview:", calib_data[0][:100])

# Quantize model
model.quantize(
    quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    },
    calib_data=calib_data,
    tokenizer=tokenizer
)

# Save quantized model
model.save_quantized(save_dir, safetensors=True)
tokenizer.save_pretrained(save_dir)

print(f"Quantized model saved to {save_dir}")
