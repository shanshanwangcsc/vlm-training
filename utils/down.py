from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-9B",
    local_dir="cache/qwen3_5_9b",
)
