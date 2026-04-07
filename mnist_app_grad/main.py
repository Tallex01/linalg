import random
import torch
import torch.nn as nn

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

app = FastAPI()

# Load trained model
device = torch.device("cpu")
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

@app.get("/image")
def image_get():
    index = random.randint(0, len(train_dataset) - 1)
    image, label = train_dataset[index]
    pixels = image.squeeze().tolist()
    
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        guessed_digit = output.argmax(1).item()
    
    return {
        "label": label,
        "guessed_digit": guessed_digit,
        "image": pixels
    }


@app.get("/predict")
def predict():
    index = random.randint(0, len(train_dataset) - 1)
    image, label = train_dataset[index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        guessed_digit = output.argmax(1).item()
    return {"guessed_digit": guessed_digit}


app.mount("/", StaticFiles(directory="static", html=True), name="static")