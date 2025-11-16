"""
Script to add W&B integration to training.ipynb.

Modifications:
1. Add wandb to Cell 4 (dependencies)
2. Insert new Cell 4A (W&B authentication)
3. Insert markdown cell explaining W&B
4. Modify Cell 12 (add wandb.init() call)
"""

import json
import sys
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = REPO_ROOT / "training.ipynb"
BACKUP_PATH = REPO_ROOT / "training.ipynb.backup"

# ==============================================================================
# New Cells to Insert
# ==============================================================================

MARKDOWN_WANDB_INFO = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📊 Weights & Biases Setup (Optional)\n",
        "\n",
        "**What is W&B?** Weights & Biases tracks your experiments so you never lose training data.\n",
        "\n",
        "**Benefits:**\n",
        "- 📈 Automatic logging of loss, metrics, and hyperparameters\n",
        "- 💾 Persistent storage (survives Colab disconnects)\n",
        "- 🔍 Compare multiple training runs side-by-side\n",
        "- 🌐 Access dashboard from anywhere: [wandb.ai](https://wandb.ai)\n",
        "\n",
        "**Setup options:**\n",
        "1. **Recommended:** Use Colab Secrets (secure, reusable)\n",
        "   - Go to 🔑 (key icon) in left sidebar → Add Secret\n",
        "   - Name: `WANDB_API_KEY`\n",
        "   - Value: Get from [wandb.ai/authorize](https://wandb.ai/authorize)\n",
        "\n",
        "2. **Quick:** Run the cell below and paste API key when prompted\n",
        "\n",
        "3. **Skip:** Cell will auto-enable offline mode (logs saved locally)\n",
        "\n",
        "**Free tier:** Unlimited runs, 100GB storage. [Create free account](https://wandb.ai/signup)\n",
        "\n",
        "---\n",
        "\n",
        "**⚠️ Security:** NEVER hardcode API keys in notebooks. Always use Colab Secrets or environment variables."
    ]
}

CELL_WANDB_LOGIN = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "source": [
        "# ==============================================================================\n",
        "# W&B AUTHENTICATION (OPTIONAL)\n",
        "# ==============================================================================\n",
        "\n",
        "#@title 🔐 **W&B Login** (optional - skip to use offline mode)\n",
        "#@markdown Run this cell to connect to Weights & Biases for experiment tracking.\n",
        "\n",
        "import os\n",
        "\n",
        "# Variable to track W&B status\n",
        "wandb_enabled = False\n",
        "\n",
        "try:\n",
        "    import wandb\n",
        "    \n",
        "    # Attempt 1: Try Colab Secrets (most secure)\n",
        "    try:\n",
        "        from google.colab import userdata\n",
        "        wandb_api_key = userdata.get('WANDB_API_KEY')\n",
        "        wandb.login(key=wandb_api_key)\n",
        "        wandb_enabled = True\n",
        "        print(\"✅ W&B authenticated via Colab Secrets\")\n",
        "        print(f\"✅ Logged in as: {wandb.api.viewer()['entity']}\")\n",
        "        print()\n",
        "        print(\"🎯 Experiments will be tracked at: https://wandb.ai\")\n",
        "    except Exception as e:\n",
        "        # Attempt 2: Try interactive login\n",
        "        try:\n",
        "            print(\"⚠️  Colab Secrets not configured, trying interactive login...\")\n",
        "            print(\"📝 Get your API key from: https://wandb.ai/authorize\")\n",
        "            print()\n",
        "            wandb.login()\n",
        "            wandb_enabled = True\n",
        "            print(\"✅ W&B authenticated via interactive login\")\n",
        "        except Exception as e2:\n",
        "            # Fallback: Offline mode\n",
        "            print(\"⚠️  W&B authentication skipped\")\n",
        "            print(\"📴 Running in offline mode (logs saved locally to .wandb/)\")\n",
        "            print()\n",
        "            print(\"To enable tracking:\")\n",
        "            print(\"  1. Create free account: https://wandb.ai/signup\")\n",
        "            print(\"  2. Add WANDB_API_KEY to Colab Secrets\")\n",
        "            print(\"  3. Re-run this cell\")\n",
        "            print()\n",
        "            os.environ['WANDB_MODE'] = 'offline'\n",
        "            wandb_enabled = False\n",
        "\n",
        "except ImportError:\n",
        "    print(\"❌ wandb not installed - please run the dependencies cell first\")\n",
        "    wandb_enabled = False\n",
        "\n",
        "print()\n",
        "print(\"=\" * 70)\n",
        "if wandb_enabled:\n",
        "    print(\"✅ W&B READY - Experiments will be tracked online\")\n",
        "else:\n",
        "    print(\"📴 W&B OFFLINE MODE - Logs saved locally only\")\n",
        "print(\"=\" * 70)\n",
        "print()"
    ],
    "outputs": []
}

# Helper function code to add to Cell 12
HELPER_FUNCTIONS = """
# ==============================================================================
# Helper: Detect Model Architecture Type
# ==============================================================================

def _detect_model_type(model):
    \"\"\"
    Detect transformer architecture type from model structure.

    Returns: 'gpt' | 'bert' | 't5' | 'custom'
    \"\"\"
    model_class = model.__class__.__name__.lower()

    # Check class name first
    if 'gpt' in model_class or 'decoder' in model_class:
        return 'gpt'
    elif 'bert' in model_class or 'encoder' in model_class:
        return 'bert'
    elif 't5' in model_class or 'encoderdecoder' in model_class:
        return 't5'

    # Inspect module structure
    module_names = [name for name, _ in model.named_modules()]
    has_decoder = any('decoder' in n.lower() for n in module_names)
    has_encoder = any('encoder' in n.lower() for n in module_names)

    if has_decoder and not has_encoder:
        return 'gpt'
    elif has_encoder and not has_decoder:
        return 'bert'
    elif has_encoder and has_decoder:
        return 't5'

    return 'custom'


# ==============================================================================
# Initialize W&B Tracking
# ==============================================================================

if 'wandb_enabled' in globals() and wandb_enabled:
    from datetime import datetime
    import wandb

    # Calculate model metadata
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_type = _detect_model_type(model)
    device_str = str(next(model.parameters()).device)

    # Define hyperparameters (will be used in training tests)
    hyperparameters = {
        'learning_rate': 5e-5,
        'batch_size': 2,
        'epochs': 3,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'use_amp': True,
        'grad_accum_steps': 1
    }

    # Initialize W&B run
    run = wandb.init(
        project=\"transformer-builder-training\",
        name=f\"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}\",
        tags=[model_type, \"v1\", \"tier3\"],
        config={
            # Hyperparameters
            \"learning_rate\": hyperparameters['learning_rate'],
            \"batch_size\": hyperparameters['batch_size'],
            \"epochs\": hyperparameters['epochs'],
            \"warmup_ratio\": hyperparameters['warmup_ratio'],
            \"weight_decay\": hyperparameters['weight_decay'],
            \"max_grad_norm\": hyperparameters['max_grad_norm'],

            # Model architecture
            \"model_type\": model_type,
            \"vocab_size\": config.vocab_size,
            \"max_seq_len\": config.max_seq_len,
            \"total_params\": total_params,
            \"trainable_params\": trainable_params,
            \"total_params_millions\": round(total_params / 1e6, 2),

            # Environment
            \"device\": device_str,
            \"mixed_precision\": hyperparameters['use_amp'],
            \"gradient_accumulation_steps\": hyperparameters['grad_accum_steps'],
        }
    )

    print(\"=\" * 80)
    print(\"📊 W&B TRACKING INITIALIZED\")
    print(\"=\" * 80)
    print()
    print(f\"🎯 Project: transformer-builder-training\")
    print(f\"🏷️  Run name: {run.name}\")
    print(f\"🔗 Dashboard: {run.get_url()}\")
    print()
    print(f\"📋 Logged config:\")\n    print(f\"   • Model: {model_type} ({round(total_params/1e6, 2)}M params)\")
    print(f\"   • Learning rate: {hyperparameters['learning_rate']}\")
    print(f\"   • Batch size: {hyperparameters['batch_size']}\")
    print(f\"   • Epochs: {hyperparameters['epochs']}\")
    print()
else:
    print(\"📴 W&B tracking disabled (offline mode or not authenticated)\")
    print()

"""


def modify_notebook():
    """Modify training.ipynb to add W&B integration."""

    print("Reading training.ipynb...")
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    # Backup original
    print(f"Creating backup at {BACKUP_PATH}...")
    with open(BACKUP_PATH, 'w') as f:
        json.dump(nb, f, indent=1)

    # Modification 1: Add wandb to Cell 4 (dependencies)
    print("Modifying Cell 4 (dependencies)...")
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any('pytorch-lightning' in line for line in cell['source']):
            # Find the pip install line and add wandb
            for i, line in enumerate(cell['source']):
                if 'pip install -q pytorch-lightning optuna torchmetrics' in line:
                    cell['source'][i] = line.replace(
                        'pip install -q pytorch-lightning optuna torchmetrics',
                        'pip install -q pytorch-lightning optuna torchmetrics wandb'
                    )
                    print("  ✅ Added wandb to pip install")
                    break

            # Add wandb to verification imports
            for i, line in enumerate(cell['source']):
                if 'import numpy as np' in line:
                    # Insert wandb import after numpy
                    cell['source'].insert(i + 1, '    import wandb\n')
                    print("  ✅ Added wandb import verification")
                    break

            # Add wandb version print
            for i, line in enumerate(cell['source']):
                if "print(f'✅ numpy: {np.__version__}')" in line:
                    # Insert wandb version after numpy
                    cell['source'].insert(i + 1, "    print(f'✅ wandb: {wandb.__version__}')\n")
                    print("  ✅ Added wandb version print")
                    break
            break

    # Modification 2: Insert markdown cell explaining W&B (after Cell 4)
    print("Inserting W&B info markdown cell...")
    cell_4_idx = None
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and any('pytorch-lightning' in line for line in cell['source']):
            cell_4_idx = idx
            break

    if cell_4_idx is not None:
        nb['cells'].insert(cell_4_idx + 1, MARKDOWN_WANDB_INFO)
        print(f"  ✅ Inserted W&B info markdown at index {cell_4_idx + 1}")

    # Modification 3: Insert Cell 4A (W&B login) after markdown
    print("Inserting Cell 4A (W&B login)...")
    if cell_4_idx is not None:
        nb['cells'].insert(cell_4_idx + 2, CELL_WANDB_LOGIN)
        print(f"  ✅ Inserted W&B login cell at index {cell_4_idx + 2}")

    # Modification 4: Modify Cell 12 (training tests) to add wandb.init()
    print("Modifying Cell 12 (training tests)...")
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and any('TIER 3: TRAINING & PRODUCTION UTILITIES' in line for line in cell['source']):
            # Find where to insert the W&B initialization code
            # Insert after the imports but before "print('=' * 80)"
            insert_idx = None
            for i, line in enumerate(cell['source']):
                if "print('=' * 80)" in line and 'TIER 3' in cell['source'][i+1]:
                    insert_idx = i
                    break

            if insert_idx is not None:
                # Insert helper functions before the print
                helper_lines = HELPER_FUNCTIONS.split('\n')
                for line_num, line in enumerate(helper_lines):
                    cell['source'].insert(insert_idx + line_num, line + '\n')

                print(f"  ✅ Added W&B initialization code at line {insert_idx}")
            break

    # Save modified notebook
    print(f"Saving modified notebook...")
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)

    print()
    print("=" * 70)
    print("✅ NOTEBOOK MODIFICATION COMPLETE")
    print("=" * 70)
    print()
    print("Changes made:")
    print("  1. Added wandb to dependencies (Cell 4)")
    print("  2. Inserted W&B info markdown cell")
    print("  3. Inserted W&B authentication cell (Cell 4A)")
    print("  4. Added wandb.init() to training tests (Cell 12)")
    print()
    print(f"Backup saved to: {BACKUP_PATH}")
    print()


if __name__ == '__main__':
    try:
        modify_notebook()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
