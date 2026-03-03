import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

torch.manual_seed(42)

data = pd.read_csv('data.csv')
data['Risk'] = data['Risk'].map({'Healthy':0, 'At Risk':1}) #chaning to 0-1

features = torch.tensor(data.drop('Risk', axis = 1).values).float()
target = torch.tensor(data['Risk'].values).float().reshape(-1,1)

fm = features.mean(axis=0, keepdim = True)
fs = features.std(axis=0, keepdim=True)

X = (features-fm) / fs
Y = target

model = nn.Linear(3,1)   #3 features, 1 output
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
epochs = 250

for epoch in range(epochs):
    Yhat = model(X)
    loss = criterion(Yhat, Y)      # Yhat must be first here
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.save({
    'fm':fm,
    'fs':fs,
    'parameters':model.state_dict()
}, 'model.pth')

