import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms

from dataset import DogBreedTrainValDataset
from model import DogBreedAlexNet


# ================= CONFIG =================
IMAGE_DIR = "/content/train"   # folder chứa ảnh
CSV_PATH = "/content/labels.csv"

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================


# ===== Load CSV =====
df = pd.read_csv(CSV_PATH)

# split train/val
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["breed"])


# ===== Transform =====
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# ===== Dataset =====
train_dataset = DogBreedTrainValDataset(IMAGE_DIR, train_df, transform)
val_dataset = DogBreedTrainValDataset(
    IMAGE_DIR, val_df, transform,
    class_to_idx=train_dataset.class_to_idx
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ===== Model =====
num_classes = len(train_dataset.class_to_idx)
model = DogBreedAlexNet(num_classes=num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# ===== Training =====
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # ===== Validation =====
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {correct/total:.4f}")


# ===== Save =====
torch.save(model.state_dict(), "alexnet_dog.pth")
print("Saved model!")