import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import ConcatDataset, DataLoader, Subset
from pathlib import Path
import random
import csv

# Parametry
BATCH_SIZE = 8
EPOCHS = 3
IMAGE_SIZE = 224
SEED = 42

CLASSES = ["football", "basketball", "tennis", "boxing", "swimming"]

project_root = Path(__file__).resolve().parents[1]
train_real = project_root / "data" / "train_real"
train_synth = project_root / "data" / "train_synth"
test_real = project_root / "data" / "test_real"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(name):
    if name == "resnet18":
        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, 5)
    elif name == "mobilenet":
        model = models.mobilenet_v2(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.last_channel, 5)
    return model.to(device)

def get_transform(aug):
    base = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]

    if aug == "A1":
        base.append(transforms.RandomHorizontalFlip())
    elif aug == "A2":
        base.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
    elif aug == "A3":
        base.append(transforms.RandomRotation(20))

    base.append(transforms.ToTensor())
    return transforms.Compose(base)

def get_train_dataset(scenario, transform):
    ds_real = datasets.ImageFolder(train_real, transform=transform)
    ds_synth = datasets.ImageFolder(train_synth, transform=transform)

    if scenario == "E1":
        return ConcatDataset([ds_real, ds_synth])

    elif scenario == "E2":
        return ds_synth

    elif scenario == "E3":
        return ds_real

    elif scenario == "E4":
        random.seed(SEED)
        n = min(len(ds_real), len(ds_synth))
        idx_real = random.sample(range(len(ds_real)), n)
        return Subset(ds_real, idx_real)

def train_and_eval(model_name, scenario, aug):
    transform = get_transform(aug)
    train_dataset = get_train_dataset(scenario, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = datasets.ImageFolder(test_real, transform=get_transform("A0"))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = get_model(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def main():
    results = []

    for model in ["resnet18", "mobilenet"]:
        for scenario in ["E1", "E2", "E3", "E4"]:
            for aug in ["A0", "A1", "A2", "A3"]:
                print(f"Running: {model} {scenario} {aug}")
                acc = train_and_eval(model, scenario, aug)
                print("Accuracy:", acc)
                results.append([model, scenario, aug, acc])

    out_path = project_root / "results" / "results.csv"
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scenario", "augmentation", "accuracy"])
        writer.writerows(results)

    print("Results saved to:", out_path)

if __name__ == "__main__":
    main()
