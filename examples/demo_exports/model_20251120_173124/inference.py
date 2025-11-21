"""
Text Inference Engine - TorchScript

Generated for task: demo-task
"""

import argparse
import torch
from typing import Dict, Any


class TextInferenceEngine:
    """TorchScript-based inference engine for text tasks."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    def predict(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """Run inference on pre-tokenized input."""
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.device))
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            return {'logits': logits.cpu().numpy().tolist()}


def main():
    parser = argparse.ArgumentParser(description='Text Inference Engine (TorchScript)')
    parser.add_argument('--model', default='artifacts/model.torchscript.pt', help='Model path')
    parser.add_argument('--device', default='cpu', help='Device')

    args = parser.parse_args()
    engine = TextInferenceEngine(args.model, args.device)
    print("✅ Model loaded. Ready for inference.")


if __name__ == "__main__":
    main()
