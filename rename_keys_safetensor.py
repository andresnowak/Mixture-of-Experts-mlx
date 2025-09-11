from safetensors.torch import load_file, save_file

# 1) load your safetensors into a dict
#    state_dict: Dict[str, torch.Tensor]
state_dict = load_file("shakespeare.safetensors")

# 2) build a new dict with renamed keys
#    e.g. strip a prefix "old_prefix." or apply any mapping
new_state_dict = {}
for old_key, tensor in state_dict.items():
    # example rename logic: drop "old_prefix." if present

    new_key = None
    for i in range(0, 4):
        for type in ["bias", "weight"]:
            if old_key.startswith(f"transformer_blocks.{i}.ff_1.{type}"):
                new_key = f"transformer_blocks.{i}.ff.ff_1.{type}"
            elif old_key.startswith(f"transformer_blocks.{i}.ff_2.{type}"):
                new_key = f"transformer_blocks.{i}.ff.ff_2.{type}"

    if new_key is None:
        new_key = old_key

    new_state_dict[new_key] = tensor

# 3) write back out
save_file(new_state_dict, "shakespeare.safetensors")
