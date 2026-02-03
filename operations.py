import torch

A = torch.tensor([
    [2,3],
    [4,5]
])

B = torch.tensor([
    [6,1],
    [3,8]
])

print(A + B)

print(A * B)    #Hadamard multiplication
print(3 * A)    #scalar multiplication