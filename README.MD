README.MD

For new people using the code, 

04/07/2025
in the env file, add your HF token 

then cd ~/FinalProject-main/src
echo "HF_TOKEN=Your Token number" > .env

cat .env
HF_TOKEN= Your token number then it exists 

it should wokr now when you run the 
python test_model.py --interactive


#make sure yiou have downloaded hte client 
huggingface-cli login
huggingface-cli repo clone meta-llama/Llama-3.2-3B

# when you set up your token, allow read access and write access to all under repos you need this

______________


##  Updated Setup Instructions for vLLM (04/10/2025)

This project now uses [vLLM](https://github.com/vllm-project/vllm) for faster and more memory-efficient inference.

###  1. Add your Hugging Face token
Create a `.env` file inside the `src/` folder:

```bash
cd ~/FinalProject-main/src
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

Verify it exists:
```bash
cat .env
# Should print: HF_TOKEN=your_huggingface_token_here
```

Make sure your token has **read and write access** to the `meta-llama/Llama-3.2-3B` model.

---

###  2. Download the model
Instead of manually cloning, use the built-in script:

```bash
python download_llama.py
```

This will download the model to `./llama3/` using the Hugging Face Hub and your token.

---

###  3. Run the model interactively
Now that you're using vLLM, launch the interactive CLI:

```bash
python test_model.py --interactive
```

---

###  4. Run as a web API
To launch the web API server:

```bash
python api.py
```

This exposes two endpoints:
- `POST /api/generate`: Accepts a prompt and task type, returns the model's response
- `GET /api/task_types`: Returns available task types

---

###  Dependencies
Ensure you've installed:
```bash
pip install -r requirements.txt
```

If you don't have `vllm` yet:
```bash
pip install vllm huggingface_hub python-dotenv flask flask-cors
```
