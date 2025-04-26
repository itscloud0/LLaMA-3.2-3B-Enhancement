from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B",
    local_dir="/scratch/jjosep31/models/llama-3.2-3b-hf",
    local_dir_use_symlinks=False,
    token="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"
)
