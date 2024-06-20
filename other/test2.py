import torch
import json
import numpy as np

# Set the random seed for reproducibility
torch.manual_seed(42)

# Generate example data with the specified dimensions
A = torch.randn(32, 16, requires_grad=True)  # Matrix B of size 32x16
Res = torch.softmax(A, dim=-1)
grad_Res = torch.randn_like(Res)

# Perform backpropagation using the custom gradient
Res.backward(grad_Res)

#Collect the matrices and gradients
# data = {
#     "Matrix A": A.detach().numpy().flatten().tolist(),
#     "grad_Res": grad_Res.detach().numpy().flatten().tolist(),
#     "Gradient of A": A.grad.numpy().flatten().tolist(),
# }

# print(json.dumps(data))

data = {
    "Matrix A": A.detach().numpy(),
    "Res": Res.detach().numpy(),
    "Gradient of A": A.grad.numpy(),
}
print(data);