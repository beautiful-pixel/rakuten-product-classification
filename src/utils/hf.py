from huggingface_hub import hf_hub_download, snapshot_download
import os

def hf_path(repo_id: str, filename: str):
    """
    Resolve a file stored on Hugging Face Hub to a local cached path.
    """
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )

def resolve_dir(repo_id, path):
    root = snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{path}/*",
        local_dir_use_symlinks=False
    )
    return os.path.join(root, path)