import torch

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
data = {
    "Matrix A": A.detach().numpy(),
    "Matrix B": B.detach().numpy(),
    "Matrix C (A @ B)": C.detach().numpy(),
    "Gradient of A": A.grad.numpy(),
    "Gradient of B": B.grad.numpy()
}

print(data)

'''
#import pandas as pd

# Convert the data into dataframes for better visualization
#df_A = pd.DataFrame(data["Matrix A"])
df_B = pd.DataFrame(data["Matrix B"])
df_C = pd.DataFrame(data["Matrix C (A @ B)"])
df_grad_A = pd.DataFrame(data["Gradient of A"])
df_grad_B = pd.DataFrame(data["Gradient of B"])

import ace_tools as tools; tools.display_dataframe_to_user(name="Matrix A", dataframe=df_A)
tools.display_dataframe_to_user(name="Matrix B", dataframe=df_B)
tools.display_dataframe_to_user(name="Matrix C (A @ B)", dataframe=df_C)
tools.display_dataframe_to_user(name="Gradient of A", dataframe=df_grad_A)
'''
#tools.display_dataframe_to_user(name="Gradient of B", dataframe=df_grad_B)
