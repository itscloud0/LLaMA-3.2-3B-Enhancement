import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load HF_TOKEN from .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
HF_TOKEN = os.getenv("HF_TOKEN")

def download_llama(model_id="meta-llama/Llama-3.2-3B", target_dir="llama3"):
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set in environment. Please add it to your .env file.")
    
    print(f"Downloading {model_id} to {target_dir} ...")
    snapshot_download(
        repo_id=model_id,
        token=HF_TOKEN,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print("Download complete.")

if __name__ == "__main__":
    download_llama()
