import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config.config as config

local_path = snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir=config.embedding_model_path,
    local_dir_use_symlinks=False  # <== this disables symlinks
)

print(f"✅ Download complete at: {local_path}")
