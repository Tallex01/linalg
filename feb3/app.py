import torch
import pandas as pd

df = pd.read_csv('data.csv')
X = torch.tensor(df.drop('Y', axis=1).to_numpy()).float()    #axis = 1 means column

# good to use floats

#target
Y = torch.tensor(df['Y'].to_numpy()).float().reshape(-1,1)      #reshape does 1 column, -1 rows

w = torch.tensor([
    [3.0],
    [-1.0],
    [-3.0],
    [-2.0]
])

b = torch.tensor([
    [-3.0]   #auto does the slotting in of the next, so no need for second entry
])

Yhat = (X @ w) + b
r = Yhat - Y
SSE = r.T @ r
loss = SSE/17    # div by pieces of data

#print(r) #residual
#print(SSE)    # sum of squared errors
print(loss)

#These are called FFN (Feed Forward Neural networks)

#goal is to pass through many many times (not just once)
# adjust the weights once you get the loss, rinse repeat

#side notE: CMD+J hides/shows terminal


