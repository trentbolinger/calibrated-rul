"""End-to-end training pipeline: load FD001, preprocess, train LSTMRULModel."""

from config import build_model
from data.loader import CMAPSSLoader
from data.preprocessor import SENSOR_COLUMNS, SequencePreprocessor

CONFIG = {
    "data_dir": "data/raw",
    "subset": "FD001",
    "sequence_length": 30,
    "rul_cap": 125,
    "calib_fraction": 0.2,
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
        train_df, calib_fraction=CONFIG["calib_fraction"]
    )

    model = build_model(CONFIG)
    model.fit(X_train, y_train, X_calib, y_calib)

    print(f"Training complete. Best model saved to {CONFIG['checkpoint_path']}")


if __name__ == "__main__":
    main()
