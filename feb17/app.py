import torch

X = torch.tensor([
    [2.0],
    [5.0],
    [8.0]
])

Y = torch.tensor([
    [3.0],
    [7.0],
    [1.0]
])

w = torch.tensor([
    [0.0]
], requires_grad = True)

b = torch.tensor([
    [0.0]
], requires_grad = True)

lr = 0.01 #learning rate


for i in range(0,1000):

    Yhat = X@w + b
    r = Yhat - Y
    loss = r.T@r / 3

    loss.backward()

    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad

#    print(w,b)
#    print(loss)

    w.grad.zero_()
    b.grad.zero_()

print(w,b)

X = torch.tensor([
    [6.0]
])

prediction = X@w+b
print(prediction)

# in our case: we get the weight and bias so our equation
# becomes about: y = -0.31x + 5.21




