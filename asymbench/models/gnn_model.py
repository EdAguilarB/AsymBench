import torch

from asymbench.gnn.model import ReactionGNN


class GNNWrapper:
    def __init__(self, config, in_dim, num_mols):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ReactionGNN(
            num_mols=num_mols,
            in_dim=in_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, train_loader, epochs=20):
        from asymbench.gnn.trainer import train_epoch

        for _ in range(epochs):
            train_epoch(self.model, train_loader, self.optimizer, self.device)

    def predict(self, loader):
        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in loader:
                graphs, _ = batch
                pred = self.model(graphs)
                preds.append(pred.cpu())

        return torch.cat(preds, dim=0).numpy()
