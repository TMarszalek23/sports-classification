import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import ConcatDataset, DataLoader
from pathlib import Path

# Ustawienia
BATCH_SIZE = 8
EPOCHS = 3
IMAGE_SIZE = 224

project_root = Path(__file__).resolve().parents[1]
train_real = project_root / "data" / "train_real"
train_synth = project_root / "data" / "train_synth"
test_real = project_root / "data" / "test_real"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transformacje (A0 – brak augmentacji)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Zbiory danych
dataset_real = datasets.ImageFolder(train_real, transform=transform)
dataset_synth = datasets.ImageFolder(train_synth, transform=transform)

train_dataset = ConcatDataset([dataset_real, dataset_synth])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.ImageFolder(test_real, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model: ResNet18
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 5)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Test
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test accuracy: {accuracy:.2f}")
