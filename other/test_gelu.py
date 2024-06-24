import torch
import torch.nn as nn

torch.manual_seed(42)

inputs = torch.randn(8, 16, requires_grad=True)

gelu = nn.GELU()

gelu_outputs = gelu(inputs)

loss = torch.randn_like(gelu_outputs)

gelu_outputs.backward(loss)

print("Input Tensor:")
print(inputs)
print("\nOutput Tensor:")
print(gelu_outputs)
print("\nLoss Tensor:")
print(loss)
print("\nGradients w.r.t. Input Tensor:")
print(inputs.grad)

import struct

with open("weights/test/gelu_in.bin", 'wb') as file:
    values = inputs.detach().numpy()
    for single_value in values.flatten():
        file.write(struct.pack('<f', single_value))

with open("weights/test/gelu_grads.bin", 'wb') as file:
    values = loss.detach().numpy()
    for single_value in values.flatten():
        file.write(struct.pack('<f', single_value))