from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/siglip2-large-patch16-384",
    local_dir="cache/siglip2_large",
)
