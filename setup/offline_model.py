from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="C:/offline_models/all-MiniLM-L6-v2",
    local_dir_use_symlinks=False  # <== this disables symlinks
)

print(f"✅ Download complete at: {local_path}")
