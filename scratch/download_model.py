import os
import sys
from huggingface_hub import hf_hub_download

def download():
    repo_id = "onnx-community/embeddinggemma-300m-ONNX"
    local_dir = "models/embeddinggemma_fp32"
    
    print(f"Downloading model.onnx from {repo_id}...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename="onnx/model.onnx",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print("Successfully downloaded model.onnx.")
        
        print("Downloading model.onnx_data...")
        hf_hub_download(
            repo_id=repo_id,
            filename="onnx/model.onnx_data",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print("Successfully downloaded model.onnx_data.")
        print("Download complete! Files are in models/embeddinggemma_fp32/onnx/")
    except Exception as e:
        print(f"Error downloading: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download()
