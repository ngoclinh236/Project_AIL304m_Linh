import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from torchvision import transforms

from dataset import DogBreedTrainValDataset
from model import DogBreedAlexNet   # (khuyên đổi ResNet bên dưới)

# ================= CONFIG =================
IMAGE_DIR = "/content/Project_AIL304m_Linh/train"
CSV_PATH = "/content/Project_AIL304m_Linh/labels.csv"

BATCH_SIZE = 32
EPOCHS = 20   
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATIENCE = 5   


# ===== Load CSV =====
df = pd.read_csv(CSV_PATH)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["breed"])


# ===== Transform =====
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4   # chống overfit
)


# ===== Early Stopping =====
best_val_loss = float("inf")
counter = 0


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

        total_loss += loss.item() * images.size(0)

    train_loss = total_loss / len(train_loader.dataset)

    # ===== Validation =====
    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_loss = val_loss / len(val_loader.dataset)
    acc = accuracy_score(y_true, y_pred)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {acc:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(
        y_true,
        y_pred,
        target_names=list(train_dataset.class_to_idx.keys()),
        zero_division=0
    ))

    # ===== Early Stopping =====
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("✔ Saved best model")
    else:
        counter += 1
        print(f"⚠ No improvement ({counter}/{PATIENCE})")

        if counter >= PATIENCE:
            print(" Early stopping triggered")
            break


print("Training finished!")