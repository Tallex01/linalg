import torch

A = torch.tensor([
    [2,6],
    [6,1]
])

B = torch.tensor([
    [5,4],
    [10,2]
])


print(A@B)    #standard matrix mult - remember dot prod and stuff

