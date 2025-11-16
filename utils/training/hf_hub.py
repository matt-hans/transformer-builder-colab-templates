"""
HuggingFace Hub push utilities (optional dependency).

Provides a safe push_model_to_hub() that degrades gracefully when
huggingface_hub is not installed or in offline environments.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


def push_model_to_hub(
    model: Any,
    config: Optional[Any],
    training_results: Dict[str, Any],
    repo_name: str,
    private: bool = False,
    commit_message: str = "Upload trained model",
    local_dir: str = "./model_for_upload",
) -> Optional[str]:
    """
    Push trained model to HuggingFace Hub with metadata.

    If huggingface_hub is unavailable or upload fails, this function
    writes files locally and returns None.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except Exception:
        print("⚠️  huggingface_hub not installed - skipping upload. Saving locally.")
        _write_local(model, config, training_results, local_dir)
        return None

    try:
        api = HfApi()
        # Create repository (idempotent)
        create_repo(repo_name, private=private, exist_ok=True)
        print(f"✅ Repository created/verified: {repo_name}")

        # Save local files
        out = _write_local(model, config, training_results, local_dir)

        # Upload folder
        api.upload_folder(
            folder_path=out,
            repo_id=repo_name,
            commit_message=commit_message
        )
        url = f"https://huggingface.co/{repo_name}"
        print(f"✅ Model uploaded: {url}")
        return url
    except Exception as e:
        print(f"❌ HF Hub upload failed: {e}")
        print(f"   Model saved locally at: {local_dir}")
        return None


def _write_local(model: Any, config: Optional[Any], training_results: Dict[str, Any], local_dir: str) -> str:
    os.makedirs(local_dir, exist_ok=True)
    # Save weights
    try:
        import torch
        target = getattr(model, 'model', model)
        torch.save(target.state_dict(), os.path.join(local_dir, 'pytorch_model.bin'))
        print(f"✅ Model weights saved to {local_dir}/pytorch_model.bin")
    except Exception as e:
        print(f"⚠️  Failed to save model weights: {e}")

    # Save config
    cfg = {}
    if config is None:
        config = getattr(model, 'config', None)
    if config is not None:
        if hasattr(config, 'to_dict'):
            cfg = config.to_dict()
        elif isinstance(config, dict):
            cfg = config
        else:
            cfg = {k: v for k, v in getattr(config, '__dict__', {}).items() if isinstance(v, (int, float, str, bool, list, dict))}
    with open(os.path.join(local_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
        print(f"✅ Config saved to {local_dir}/config.json")

    # Model card
    readme = _generate_minimal_model_card(model, cfg, training_results)
    Path(local_dir, 'README.md').write_text(readme)
    print(f"✅ Model card written to {local_dir}/README.md")
    return local_dir


def _generate_minimal_model_card(model: Any, config: Dict[str, Any], results: Dict[str, Any]) -> str:
    total_params = 0
    try:
        total_params = sum(p.numel() for p in getattr(model, 'parameters', lambda: [])())
    except Exception:
        pass
    name = config.get('name', 'Custom Transformer Model') if isinstance(config, dict) else 'Custom Transformer Model'
    val_loss = results.get('val_loss') or results.get('final_val_loss') or 'N/A'
    val_ppl = results.get('val_perplexity') or results.get('final_val_ppl') or 'N/A'
    card = f"""# {name}

Trained with Transformer Builder templates. This repository contains the trained PyTorch weights and configuration.

## Model Details
- Parameters: {total_params:,}
- Vocab size: {config.get('vocab_size', 'N/A') if isinstance(config, dict) else 'N/A'}
- Max seq len: {config.get('max_seq_len', 'N/A') if isinstance(config, dict) else 'N/A'}

## Final Metrics
- Validation loss: {val_loss}
- Validation perplexity: {val_ppl}

## Files
- pytorch_model.bin
- config.json
- README.md
"""
    return card

