"""End-to-end training pipeline: load FD001, preprocess, train LSTMRULModel."""

from pathlib import Path

import matplotlib.pyplot as plt

from config import build_model
from data.loader import CMAPSSLoader
from data.preprocessor import SENSOR_COLUMNS, SequencePreprocessor

OUTPUT_DIR = Path("outputs")

CONFIG = {
    "data_dir": "data/raw",
    "subset": "FD001",
    "sequence_length": 30,
    "rul_cap": 125,
    "calib_fraction": 0.4,
    "calib_seed": 42,
    "model_type": "lstm",
    "input_size": len(SENSOR_COLUMNS),
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 50,
    "checkpoint_path": "outputs/best_model.pth",
}


def plot_training_curves(history: dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="train loss")
    if any(v is not None for v in history["val_loss"]):
        ax.plot(epochs, history["val_loss"], label="val loss")
    if history["best_epoch"] is not None:
        ax.axvline(history["best_epoch"], linestyle=":", color="gray", label="best model")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    loader = CMAPSSLoader(CONFIG["data_dir"], subset=CONFIG["subset"])
    train_df = loader.load_train()

    preprocessor = SequencePreprocessor(
        sequence_length=CONFIG["sequence_length"], rul_cap=CONFIG["rul_cap"]
    )
    train_df = preprocessor.compute_rul(train_df)
    preprocessor.fit_scaler(train_df)
    train_df = preprocessor.normalize(train_df)

    X_train, y_train, X_calib, y_calib, train_ids, calib_ids = preprocessor.split_calibration_set(
        train_df, calib_fraction=CONFIG["calib_fraction"], seed=CONFIG["calib_seed"]
    )

    model = build_model(CONFIG)
    model.fit(X_train, y_train, X_calib, y_calib)
    plot_training_curves(model.history)

    print(f"Training complete. Best model saved to {CONFIG['checkpoint_path']}")


if __name__ == "__main__":
    main()
