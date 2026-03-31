from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from grid import save_image_grid
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

dataset = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize((0.1307,),(0.3081,))       #overall: taking an image turning it into a matrix of numbers
    ])
)

test_dataset = datasets.MNIST(
    root = './data',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize((0.1307,),(0.3081,))       #overall: taking an image turning it into a matrix of numbers
    ])
)


#### not apart of the program
#dataset - special pytorch structure
# image,label = dataset[0]
# print(image)
# print(label)
# image.save('image.png')



# DataLoaders
loader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle = True

)

test_loader = DataLoader(
    test_dataset,
    batch_size = 1000,
    shuffle = False

)

# for i, (images,labels) in enumerate(loader):
#     save_image_grid(images, batch_size = 10, path = 'images.png')
#     if i == 9:
#         break

#Neural Networks
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),                 #128 asked from ai (# nodes in the hidden layer)
    nn.ReLU(),
    nn.Linear(128,64),                   #64 asked from ai (# nodes in the hidden layer)
    nn.ReLU(),
    nn.Linear(64,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 10

for epoch in range(epochs):
    total_loss = 0
    total = 0           #total pictures
    correct = 0
    for images,labels in loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()        #.item() removes the tensor
        total += labels.size(0)         #just a flat vector, size gives dimension, 0 is the first part of the dim
        correct += (output.argmax(1) == labels).sum().item()  #sums the correct classifications?
    test_total = 0
    test_correct = 0
    with torch.no_grad():               #with no calculus
        for images,labels in test_loader:
            output = model(images)
            test_total += labels.size(0)
            test_correct += (output.argmax(1) == labels).sum().item()



    #print(total_loss/len(loader))       #loss per batch
    print(correct/total)
    print(test_correct/test_total)
    print('---------------')


torch.save(model.state_dict(), 'model.pth')











