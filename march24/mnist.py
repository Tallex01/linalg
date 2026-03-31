from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from grid import save_image_grid
import torch
import torch.nn as nn
import torch.optim as optim

dataset = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
)

#dataset - special pytorch structure
# image,label = dataset[0]
# print(image)
# print(label)
# image.save('image.png')



# DataLoaders
loader = DataLoader(
    dataset,
    batch_size = 10


)

# for i, (images,labels) in enumerate(loader):
#     save_image_grid(images, batch_size = 10, path = 'images.png')
#     if i == 9:
#         break

#Neural Networks
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),                 #128 asked from ai
    nn.ReLU(),
    nn.Linear(128,64),                   #64 asked from ai
    nn.ReLU(),
    nn.Linear(64,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 10









