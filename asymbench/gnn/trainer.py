import numpy as np
import torch
import torch.nn.functional as F


def train_epoch(model, loader, optimizer, device):
    """Run one full training epoch and return the average MSE loss."""
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, data.y.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def predict(model, loader, device) -> np.ndarray:
    """Run inference and return predictions as a 1-D numpy array."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds = model(data)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)
