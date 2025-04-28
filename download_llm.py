from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get directory path from .env
DIRECTORY_PATH = os.getenv('DIRECTORY_PATH')
if not DIRECTORY_PATH:
    raise ValueError("DIRECTORY_PATH environment variable is not set")

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B",
    local_dir=f"/scratch/{DIRECTORY_PATH}/llama-3.2-3b-hf",
    local_dir_use_symlinks=False,
    token=os.getenv("HF_TOKEN")
)
