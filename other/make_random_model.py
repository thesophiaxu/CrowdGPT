import torch
import torch.nn as nn

# change seed
torch.manual_seed(42)

n_layer = 2
n_head = 4
n_embd = 128
n_ctx = 64
vocab_size = 65
dirname = 'weights/rand'

# initialize random model weights
weights_dict = dict()

weights_dict['transformer.wte.weight'] = torch.randn(vocab_size, n_embd)
weights_dict['transformer.wpe.weight'] = torch.randn(n_ctx, n_embd)

for layer in range(n_layer):
    weights_dict[f'transformer.h.{layer}.ln_1.weight'] = torch.ones(n_embd)
    weights_dict[f'transformer.h.{layer}.ln_1.bias'] = torch.zeros(n_embd)
    weights_dict[f'transformer.h.{layer}.attn.c_attn.weight'] = torch.randn(3 * n_embd, n_embd).T
    weights_dict[f'transformer.h.{layer}.attn.c_attn.bias'] = torch.randn(3 * n_embd)
    weights_dict[f'transformer.h.{layer}.attn.c_proj.weight'] = torch.randn(n_embd, n_embd).T
    weights_dict[f'transformer.h.{layer}.attn.c_proj.bias'] = torch.randn(n_embd)
    weights_dict[f'transformer.h.{layer}.ln_2.weight'] = torch.ones(n_embd)
    weights_dict[f'transformer.h.{layer}.ln_2.bias'] = torch.zeros(n_embd)
    weights_dict[f'transformer.h.{layer}.mlp.c_fc.weight'] = torch.randn(4 * n_embd, n_embd).T
    weights_dict[f'transformer.h.{layer}.mlp.c_fc.bias'] = torch.randn(4 * n_embd)
    weights_dict[f'transformer.h.{layer}.mlp.c_proj.weight'] = torch.randn(n_embd, 4 * n_embd).T
    weights_dict[f'transformer.h.{layer}.mlp.c_proj.bias'] = torch.randn(n_embd)

weights_dict['transformer.ln_f.weight'] = torch.randn(n_embd)
weights_dict['transformer.ln_f.bias'] = torch.randn(n_embd)
weights_dict['lm_head.weight'] = torch.randn(vocab_size, n_embd)

import os
import struct

# Create the directory if it doesn't exist
os.makedirs(dirname, exist_ok=True)

# Save each weight tensor to a separate file
for key, value in weights_dict.items():
    file_path = os.path.join(dirname, key + "_gpt.bin")
    with open(file_path, 'wb') as file:
        values = value.detach().numpy()
        for single_value in values.flatten():
            file.write(struct.pack('<f', single_value))