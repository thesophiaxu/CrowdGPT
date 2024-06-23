import torch
from torch.nn import functional as F
import struct

torch.manual_seed(42)

A = torch.randn(640, 7680, requires_grad=True)
B = torch.randint(0, 7680, (640,), dtype=torch.int64)

C = F.cross_entropy(A, B, ignore_index=-1)
print(C)
C.backward()
print(A.grad)

with open("weights/test/a.bin", 'wb') as file:
    values = A.detach().numpy()
    for single_value in A.flatten():
        file.write(struct.pack('<f', single_value))

with open("weights/test/b.bin", 'wb') as file:
    values = B.detach().numpy()
    for single_value in B.flatten():
        file.write(struct.pack('<f', single_value))