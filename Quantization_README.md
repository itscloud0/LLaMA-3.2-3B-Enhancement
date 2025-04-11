
# Quantized LLaMA 3 Project README

This  quantizes Meta's LLaMA 3 model (3.2B) using the **AWQ** (Activation-aware Weight Quantization) technique to enable faster inference and lower memory usage. It also updates the local model serving code to use the quantized model directly via Hugging Face Transformers-compatible APIs instead of vLLM.

---

## What We've Done

### 1. Installed and Set Up AWQ

- Installed `autoawq` (AWQ repo) from PyPI:

  ```bash
  pip install autoawq --upgrade
  ```

- Installed supporting packages:

  ```bash
  pip install transformers accelerate datasets scipy safetensors einops sentencepiece bitsandbytes flask
  ```

- Activated the `llama3env` conda environment:

  ```bash
  conda activate llama3env
  ```

- Logged into Hugging Face CLI:

  ```bash
  huggingface-cli login
  ```

  Token saved to:
  - to envirment 


### 2. Quantized the Model

- Created a  calibration dataset `calib.txt` with ~80 lines of input.
- Duplicated calibration lines to ensure sufficient token count.
- Ran `quantize_llama3.py` to:
  - Load the base model using `AutoAWQForCausalLM.from_pretrained()`
  - Call `.quantize()` with:

    ```python
    quant_config={
      "zero_point": True,
      "q_group_size": 128,
      "w_bit": 4,
      "version": "GEMM"
    }
    ```
  - Save the quantized model and tokenizer using `.save_quantized()` and `tokenizer.save_pretrained()`

- Moved model to `/scratch/jjosep31/llama3-awq` due to GB limit in `/home` (it will eat up sapce)

### 3. Replaced vLLM with AWQ in Wrapper

- Updated `llm_wrapper.py`:
  - Replaced `vllm.LLM` with `AutoAWQForCausalLM.from_quantized()`
  - Used HF `AutoTokenizer`
  - Used `.generate()` instead of `SamplingParams`

### 4. Updated Flask API Backend

- Updated `api.py` to use the AWQ model wrapper instead of vLLM.
- Confirmed endpoint `/api/generate` works as expected.
- Model loads from `/scratch`, 

### 5. Interactive + CLI Testing

- Updated `test_model.py` to:
  - Initialize wrapper with AWQ model


---

## vLLM to AWQ

We switched to AWQ to reduce model memory usage and allow running the model in 8GB–12GB VRAM environments while still preserving good performance.

---


### Things we might do Install AWQ Extension for speed boost we need CUDA for this 

> This enables fused CUDA kernels and speeds up inference. Skip if not compiling custom CUDA code.

---

## Next Steps

- Benchmark quantized model vs. full precision for memory and speed

---


FinalProject-main/
│
├── src/
│   ├── quantize_llama3.py        # Quantizes LLaMA 3 using AWQ
│   ├── llm_wrapper.py            # Wrapper class using quantized model
│   ├── api.py                    # Flask API using AWQ wrapper
│   ├── test_model.py             # CLI testing for model behavior
│   ├── config.py                 # Prompt templates and generation configs
│   └── calib.txt                 # Calibration dataset for AWQ quantization

