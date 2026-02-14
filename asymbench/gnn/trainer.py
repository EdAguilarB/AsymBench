import torch
import torch.nn.functional as F


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        if batch is None:
            continue

        graphs, y = batch
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(graphs)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
