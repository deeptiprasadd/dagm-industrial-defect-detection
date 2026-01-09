import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from src.dataset import DAGMDataset
from src.model import DefectCNN
from src.transforms import transform

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "DAGM", "Class1", "Train")

    dataset = DAGMDataset(train_path, transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = DefectCNN().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3):  # keep small for UI
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.float().to(device)
            optim.zero_grad()
            out = model(x).squeeze()
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()

    torch.save(model.state_dict(), "defect_model.pth")
    return len(dataset)
