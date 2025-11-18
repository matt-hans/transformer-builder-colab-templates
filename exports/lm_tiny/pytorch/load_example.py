"""
Example code to load exported model.
"""
import torch
import json

with open('config.json', 'r') as f:
    config = json.load(f)

# TODO: Replace with your model class
class YourModelClass(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # define layers based on config
        pass
    def forward(self, x):
        pass

model = YourModelClass(config)
state = torch.load('pytorch_model.bin', map_location='cpu')
model.load_state_dict(state, strict=False)
model.eval()

print('Model loaded. Ready for inference.')
