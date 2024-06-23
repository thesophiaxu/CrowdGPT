import torch
import torch.nn as nn

torch.manual_seed(44)

# Sample toy data with small sizes
batch_size = 4
num_features = 8

# Generate random input tensor
input_tensor = torch.randn(batch_size, num_features, requires_grad=True)

# Define LayerNorm with toy sizes
layer_norm = nn.LayerNorm(num_features)

# Forward pass
output = layer_norm(input_tensor)

loss = torch.randn_like(output)

output.backward(loss)

# Print input, output, and gradients for verification
print("Input Tensor:")
print(input_tensor)
print("\nOutput Tensor:")
print(output)
print("\nGradients w.r.t. Input Tensor:")
print(input_tensor.grad)

import struct

with open("weights/test/ln_in.bin", 'wb') as file:
    values = input_tensor.detach().numpy()
    for single_value in values.T.flatten():
        file.write(struct.pack('<f', single_value))