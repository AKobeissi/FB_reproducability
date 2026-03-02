"""
Helper script to load models saved with lightweight checkpoints (weights only).

This demonstrates how to reload models that were saved using the space-efficient
approach (storing only weights/adapters instead of full 2.2GB models).
"""
import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


def load_fold_model(fold_dir: str, device: str = "cuda") -> SentenceTransformer:
    """
    Load a k-fold model from lightweight checkpoint.
    
    Args:
        fold_dir: Path to fold directory (e.g., "results/kfold_page_scorer/20260218_123456/fold_0")
        device: Device to load model on
    
    Returns:
        Loaded SentenceTransformer model
    
    Example:
        model = load_fold_model("results/kfold_page_scorer/20260218_123456/fold_0")
        embeddings = model.encode(["query text"])
    """
    fold_dir = Path(fold_dir)
    
    # Load config to get base model name
    config_path = fold_dir / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {config_path}\n"
            f"Make sure you're using a model saved with the new lightweight format."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_model_name = config["base_model"]
    
    # Load base model from HuggingFace (downloads once, then cached)
    print(f"Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name, device=device)
    
    # Load trained weights
    weights_path = fold_dir / "model_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    print(f"Loading trained weights: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Set max sequence length if specified
    if "max_seq_length" in config:
        model.max_seq_length = config["max_seq_length"]
    
    print(f"✓ Model loaded successfully from {fold_dir}")
    return model


def load_mgpear_model(fold_dir: str, device: str = "cuda"):
    """
    Load an MGPEAR model from lightweight checkpoint.
    
    Args:
        fold_dir: Path to fold directory containing mgpear_heads.pt
        device: Device to load on
    
    Returns:
        Tuple of (backbone, query_page, page_proj, query_chunk, chunk_proj, tokenizer)
    
    Example:
        components = load_mgpear_model("results/kfold_mgpear/20260218_123456/fold_0")
        backbone, query_page, page_proj, query_chunk, chunk_proj, tokenizer = components
    """
    fold_dir = Path(fold_dir)
    
    # Load heads checkpoint
    heads_path = fold_dir / "mgpear_heads.pt"
    if not heads_path.exists():
        raise FileNotFoundError(f"MGPEAR heads not found: {heads_path}")
    
    print(f"Loading MGPEAR heads: {heads_path}")
    checkpoint = torch.load(heads_path, map_location=device)
    
    config = checkpoint["config"]
    base_model_name = checkpoint.get("base_model", config.get("backbone_name"))
    
    # Load base model from HuggingFace
    print(f"Loading base backbone: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    backbone = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Load LoRA adapters if they exist
    lora_dirs = list(fold_dir.glob("lora_*"))
    if lora_dirs:
        try:
            from peft import PeftModel
            lora_dir = lora_dirs[0]
            print(f"Loading LoRA adapters: {lora_dir}")
            backbone = PeftModel.from_pretrained(backbone, str(lora_dir))
        except ImportError:
            print("Warning: peft not installed, skipping LoRA adapters")
    
    backbone = backbone.to(device)
    
    # Reconstruct projection heads
    import torch.nn as nn
    hidden_size = backbone.config.hidden_size
    rank = config.get("rank", 128)
    
    query_page = nn.Linear(hidden_size, rank, bias=False).to(device)
    page_proj = nn.Linear(hidden_size, rank, bias=False).to(device)
    query_chunk = nn.Linear(hidden_size, rank, bias=False).to(device)
    chunk_proj = nn.Linear(hidden_size, rank, bias=False).to(device)
    
    # Load trained head weights
    query_page.load_state_dict(checkpoint["heads"]["query_page"])
    page_proj.load_state_dict(checkpoint["heads"]["page_proj"])
    query_chunk.load_state_dict(checkpoint["heads"]["query_chunk"])
    chunk_proj.load_state_dict(checkpoint["heads"]["chunk_proj"])
    
    print(f"✓ MGPEAR model loaded successfully from {fold_dir}")
    
    return backbone, query_page, page_proj, query_chunk, chunk_proj, tokenizer


def print_checkpoint_info(fold_dir: str):
    """Print information about a saved checkpoint."""
    fold_dir = Path(fold_dir)
    
    print(f"\n{'='*60}")
    print(f"Checkpoint: {fold_dir}")
    print(f"{'='*60}")
    
    # Check for k-fold model
    config_path = fold_dir / "model_config.json"
    weights_path = fold_dir / "model_weights.pt"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Type: K-Fold SentenceTransformer")
        print(f"Base Model: {config['base_model']}")
        if weights_path.exists():
            size_mb = weights_path.stat().st_size / (1024 * 1024)
            print(f"Weights Size: {size_mb:.1f} MB")
    
    # Check for MGPEAR model
    heads_path = fold_dir / "mgpear_heads.pt"
    if heads_path.exists():
        checkpoint = torch.load(heads_path, map_location='cpu')
        config = checkpoint["config"]
        print(f"Type: MGPEAR")
        print(f"Base Model: {checkpoint.get('base_model', config.get('backbone_name'))}")
        size_mb = heads_path.stat().st_size / (1024 * 1024)
        print(f"Heads Size: {size_mb:.1f} MB")
        
        # Check for LoRA
        lora_dirs = list(fold_dir.glob("lora_*"))
        if lora_dirs:
            print(f"LoRA Adapters: {len(lora_dirs)} adapter(s)")
            for lora_dir in lora_dirs:
                lora_size = sum(f.stat().st_size for f in lora_dir.rglob('*') if f.is_file())
                print(f"  - {lora_dir.name}: {lora_size / (1024 * 1024):.1f} MB")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python load_saved_model.py <fold_dir>")
        print("\nExample:")
        print("  python load_saved_model.py results/kfold_page_scorer/20260218_123456/fold_0")
        sys.exit(1)
    
    fold_dir = sys.argv[1]
    
    # Print checkpoint info
    print_checkpoint_info(fold_dir)
    
    # Try to load the model
    try:
        if (Path(fold_dir) / "model_config.json").exists():
            model = load_fold_model(fold_dir)
            print(f"\n✓ Successfully loaded k-fold model")
            print(f"  Model: {model}")
        elif (Path(fold_dir) / "mgpear_heads.pt").exists():
            components = load_mgpear_model(fold_dir)
            print(f"\n✓ Successfully loaded MGPEAR model")
            print(f"  Components: {len(components)} (backbone, heads, tokenizer)")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        sys.exit(1)
