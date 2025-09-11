import torch
from safetensors.torch import save_file

# 1. load the .bin
state_dict = torch.load("model.bin", map_location="cpu")

# 2. write out as safetensors
save_file(state_dict, "model.safetensors")
