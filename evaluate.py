import torch
from torch.utils.data import DataLoader

import pandas as pd
from torchvision import transforms

from dataset import DogBreedTrainValDataset
from model import DogBreedAlexNet


IMAGE_DIR = "/content/train"
CSV_PATH = "/content/labels.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load data
df = pd.read_csv(CSV_PATH)

transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = DogBreedTrainValDataset(IMAGE_DIR, df, transform)
loader = DataLoader(dataset, batch_size=32)


# model
num_classes = len(dataset.class_to_idx)
model = DogBreedAlexNet(num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load("alexnet_dog.pth", map_location=DEVICE))

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy:", correct / total)