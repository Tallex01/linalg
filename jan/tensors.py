import torch #needed to install numpy

#2D brackets makes it a 2D tensor (matrix)
x = torch.tensor([
    [3,2,6,8],   #1 row
    [5,2,6,9]
])  # scal/vect/mat adding brackets makes it 1D tensor
print(x.shape)      #shape
print(x.dim())       #dimension


# a row/column vector can be referred to as a matrix
# e.g. a 1x4 matrix, a 6x1 matrix

A = torch.tensor([
    [1,5],
    [7,4]
])

print(A.T)  #transpose