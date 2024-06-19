import torch
import json
import numpy as np

# Set the random seed for reproducibility
torch.manual_seed(0)

# Generate example data with the specified dimensions
A = torch.randn(16, 32, requires_grad=True)  # Matrix A of size 16x32
B = torch.randn(32, 16, requires_grad=True)  # Matrix B of size 32x16

# Perform matrix multiplication
C = torch.matmul(A, B)

# Create a gradient for each element of C (same shape as C)
grad_C = torch.ones_like(C)

# Perform backpropagation using the custom gradient
C.backward(grad_C)

# Collect the matrices and gradients
# data = {
#     "Matrix A": A.detach().numpy().flatten().tolist(),
#     "Matrix B": B.detach().numpy().flatten().tolist(),
#     "Matrix C (A @ B)": C.detach().numpy().flatten().tolist(),
#     "gc": grad_C.detach().numpy().flatten().tolist(),
#     "Gradient of A": A.grad.numpy().flatten().tolist(),
#     "Gradient of B": B.grad.numpy().flatten().tolist()
# }

# print(json.dumps(data))

data = {
    "Matrix A": A.detach().numpy(),
    "Matrix B": B.detach().numpy(),
    "Matrix C (A @ B)": C.detach().numpy(),
    "gc": grad_C.detach().numpy(),
    "Gradient of A": A.grad.numpy(),
    "Gradient of B": B.grad.numpy()
}
print(data);