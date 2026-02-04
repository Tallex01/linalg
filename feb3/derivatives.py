import torch

x = torch.tensor(2.0, requires_grad = True)
y = torch.tensor(3.0, requires_grad = True)

f = (3*y**2*x**2 + x**3*y**2) / (4*x**2*y**2 + 3*x**2*y**2 + 3)
f.backward()
print(x.grad)
print(y.grad)
#x.grad.zero_()



