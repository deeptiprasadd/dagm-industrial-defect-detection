import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from src.dataset import DAGMDataset
from src.model import DefectCNN
from src.transforms import train_transform

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DefectCNN().to(device)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([3.0]).to(device)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "DAGM",
    "Class1",
    "Train"
)
train_ds = DAGMDataset(
    root_dir=DATA_DIR,
    transform=train_transform
)



train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True
)

for epoch in range(10):
    model.train()
    epoch_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {epoch_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "defect_model.pth")
print("Model saved as defect_model.pth")
