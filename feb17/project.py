import torch                    #need torch, torch.nn as nn, torch.optim as optim,  (possibly pandas, np)
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')      #reading in the data

features = torch.tensor(data.drop('Clicks (100s)', axis = 1).to_numpy()).float()      #dropping a column instead of a row
target = torch.tensor(data['Clicks (100s)'].to_numpy()).float().reshape(-1,1)       #1 column, how many rows to make the column

fm = features.mean()    #means and sds for features & target
fs = features.std()
tm = target.mean()
ts = target.std()

X = (features - fm) / fs   #standardizing the features and target
Y = (target - tm) / ts

model = nn.Linear(4,1)      #creating a model with 4 features, 1 target (random weights, random bias)   
criterion = nn.MSELoss()    #creating loss function, checks how close our pred is to target
optimizer = optim.SGD(model.parameters(), lr = 0.1)     #Stochastic Gradient Descent

epochs = 1000           
for epoch in range(epochs):
    Yhat = model(X)      #creating prediction yhat
    loss = criterion(Yhat, Y)   #finding loss between Yhat and Y
    loss.backward()             # derivatives
    optimizer.step()            #update weights
    optimizer.zero_grad()       # reset gradient
    print(loss)

features = torch.tensor([
    [7.5, 15.0, 70.0, 1.0]
])

X = (features - fm) / fs            #unstandardizing
pred = model(X)                     #new prediction
print(pred*ts + tm)                 #re-standardizing & printing





