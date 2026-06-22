"""Standalone evaluation of the saved best LSTM checkpoint on the FD001 test set.

Prints the RMSE against the ground-truth RUL values in RUL_FD001.txt -- the
true baseline number for Phase 3.
"""

import numpy as np

from config import build_model
from data.loader import CMAPSSLoader
from data.preprocessor import SENSOR_COLUMNS, SequencePreprocessor

CONFIG = {
    "data_dir": "data/raw",
    "subset": "FD001",
    "sequence_length": 30,
    "rul_cap": 125,
    "model_type": "lstm",
    "input_size": len(SENSOR_COLUMNS),
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "checkpoint_path": "outputs/best_model.pth",
}


def main() -> None:
    loader = CMAPSSLoader(CONFIG["data_dir"], subset=CONFIG["subset"])
    train_df = loader.load_train()
    test_df = loader.load_test()
    true_rul = loader.load_test_rul()

    preprocessor = SequencePreprocessor(
        sequence_length=CONFIG["sequence_length"], rul_cap=CONFIG["rul_cap"]
    )
    # scaler is fit on train data only, same as train.py, since it's not persisted to disk
    preprocessor.fit_scaler(train_df)
    test_df = preprocessor.normalize(test_df)

    # create_sequences needs a RUL column but test trajectories are truncated
    # before failure, so the true label comes from RUL_FD001.txt instead
    test_df["RUL"] = 0
    X_test, _ = preprocessor.create_sequences(test_df, is_train=False)

    model = build_model(CONFIG)
    model.load(CONFIG["checkpoint_path"])

    preds = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((preds - true_rul) ** 2)))

    print(f"Test RMSE on {CONFIG['subset']} ({len(true_rul)} engines): {rmse:.2f}")


if __name__ == "__main__":
    main()
