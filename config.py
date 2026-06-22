"""Factory for constructing RUL models from config dictionaries."""

from models.base import BaseRULModel
from models.lstm_model import LSTMRULModel


def build_model(config: dict) -> BaseRULModel:
    model_type = config["model_type"]

    if model_type == "lstm":
        return LSTMRULModel(
            input_size=config["input_size"],
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.2),
            lr=config.get("lr", 1e-3),
            batch_size=config.get("batch_size", 64),
            epochs=config.get("epochs", 50),
            checkpoint_path=config.get("checkpoint_path", "outputs/best_model.pth"),
        )

    raise ValueError(f"Unknown model_type: {model_type!r}")
