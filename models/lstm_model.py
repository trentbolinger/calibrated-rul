"""LSTM-based RUL regression model."""

import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.base import BaseRULModel


class _LSTMNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


class LSTMRULModel(BaseRULModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        checkpoint_path: str = "outputs/best_model.pth",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = _LSTMNetwork(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.history = {"train_loss": [], "val_loss": [], "best_epoch": None}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=self.batch_size, shuffle=True
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        patience = 10
        epochs_without_improvement = 0
        self.history = {"train_loss": [], "val_loss": [], "best_epoch": None}

        for epoch in range(1, self.epochs + 1):
            self.network.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.network(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            if has_val:
                self.network.eval()
                with torch.no_grad():
                    val_loss = criterion(self.network(X_val_t), y_val_t).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    self.save(self.checkpoint_path)
                    print(
                        f"New best model at epoch {epoch} - val_loss: {val_loss:.2f}, "
                        f"saved to {self.checkpoint_path}"
                    )
                else:
                    epochs_without_improvement += 1
            else:
                val_loss = None

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            if has_val and epochs_without_improvement == 0:
                self.history["best_epoch"] = epoch

            scheduler.step()

            if epoch % 10 == 0:
                if has_val:
                    print(f"Epoch {epoch}/{self.epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}/{self.epochs} - train_loss: {train_loss:.4f}")

            if has_val and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} - best val_loss: {best_val_loss:.2f}")
                break

        if not has_val:
            self.save(self.checkpoint_path)

    def predict(self, X) -> np.ndarray:
        self.network.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            preds = self.network(X_t)
        return preds.cpu().numpy()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
