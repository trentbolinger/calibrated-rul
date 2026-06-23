"""Standalone evaluation of the saved best LSTM checkpoint on the FD001 test set.

Prints the RMSE against the ground-truth RUL values in RUL_FD001.txt -- the
true baseline number for Phase 3 -- then calibrates a split conformal
predictor on the held-out calibration engines and reports PICP/sharpness for
90% intervals on the test set. This is the Phase 4 control result: PICP
landing near 90% confirms the conformal pipeline works before Phase 5
stress-tests it under distribution shift.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import build_model
from data.loader import CMAPSSLoader
from data.preprocessor import SENSOR_COLUMNS, SequencePreprocessor
from evaluation.metrics import CalibrationEvaluator
from uncertainty.conformal import ConformalPredictor

OUTPUT_DIR = Path("outputs")

CONFIG = {
    "data_dir": "data/raw",
    "subset": "FD001",
    "sequence_length": 30,
    "rul_cap": 125,
    "calib_fraction": 0.2,
    "calib_seed": 42,
    "alpha": 0.1,
    "model_type": "lstm",
    "input_size": len(SENSOR_COLUMNS),
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "checkpoint_path": "outputs/best_model.pth",
}


def plot_conformal_intervals(preds, lower, upper, true_rul, n_samples: int = 5) -> None:
    indices = np.arange(1, n_samples + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    yerr = np.vstack([preds[:n_samples] - lower[:n_samples], upper[:n_samples] - preds[:n_samples]])
    ax.errorbar(
        indices, preds[:n_samples], yerr=yerr, fmt="o", capsize=5, label="predicted RUL (90% interval)"
    )
    ax.scatter(
        indices, true_rul[:n_samples], marker="_", s=400, color="red", linewidths=2, label="true RUL", zorder=5
    )
    ax.set_xlabel("test engine index")
    ax.set_ylabel("RUL (cycles)")
    ax.set_xticks(indices)
    ax.set_title("Conformal prediction intervals vs. true RUL (FD001 test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "conformal_fd001.png", dpi=150)
    plt.close(fig)


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

    train_df = preprocessor.compute_rul(train_df)
    train_df = preprocessor.normalize(train_df)
    _, _, X_calib, y_calib, _, _ = preprocessor.split_calibration_set(
        train_df, calib_fraction=CONFIG["calib_fraction"], seed=CONFIG["calib_seed"]
    )

    test_df = preprocessor.normalize(test_df)
    # create_sequences needs a RUL column but test trajectories are truncated
    # before failure, so the true label comes from RUL_FD001.txt instead
    test_df["RUL"] = 0
    X_test, _ = preprocessor.create_sequences(test_df, is_train=False)

    model = build_model(CONFIG)
    model.load(CONFIG["checkpoint_path"])

    preds = model.predict(X_test)

    conformal = ConformalPredictor()
    conformal.calibrate(model, X_calib, y_calib)
    lower, upper = conformal.predict_interval(model, X_test, alpha=CONFIG["alpha"])

    rmse = CalibrationEvaluator.rmse(true_rul, preds)
    picp = CalibrationEvaluator.picp(true_rul, lower, upper)
    sharpness = CalibrationEvaluator.sharpness(lower, upper)

    print(f"Test RMSE on {CONFIG['subset']} ({len(true_rul)} engines): {rmse:.2f}")
    print(f"PICP (target {1 - CONFIG['alpha']:.0%}): {picp:.2%}")
    print(f"Sharpness (avg interval width): {sharpness:.2f}")

    plot_conformal_intervals(preds, lower, upper, true_rul)
    print(f"Conformal interval plot saved to {OUTPUT_DIR / 'conformal_fd001.png'}")


if __name__ == "__main__":
    main()
