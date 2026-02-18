import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

features = torch.tensor([
    [2.0],
    [5.0],
    [8.0]
])

target = torch.tensor([
    [3.0],
    [7.0],
    [1.0]
])


fm = torch.tensor([
    [features.mean()]
])

fs = torch.tensor([
    [features.std()]
])

tm = torch.tensor([
    [target.mean()]
])

ts = torch.tensor([
    [target.std()]
])

X = (features - fm) / fs
Y = (target - tm) / ts

model = nn.Linear(1,1)      # 1 input, 1 output
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

epochs = 100
for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat, Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print(model.weight)
print(model.bias)

features = torch.tensor([
    [6.0]
])
X = (features-fm) / fs

pred = model(X)
print(pred * ts + tm)   #have to decompress