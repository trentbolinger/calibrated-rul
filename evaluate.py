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
from evaluation.shift import DistributionShiftAnalyzer
from uncertainty.conformal import ConformalPredictor

OUTPUT_DIR = Path("outputs")

CONFIG = {
    "data_dir": "data/raw",
    "subset": "FD001",
    "shift_subsets": ["FD002", "FD004"],
    "sequence_length": 30,
    "rul_cap": 125,
    "calib_fraction": 0.4,
    "calib_seed": 42,
    "alpha": 0.1,
    "model_type": "lstm",
    "input_size": len(SENSOR_COLUMNS),
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "checkpoint_path": "outputs/best_model.pth",
}


def load_test_subset(preprocessor: SequencePreprocessor, data_dir: str, subset_name: str):
    loader = CMAPSSLoader(data_dir, subset=subset_name)
    test_df = loader.load_test()
    true_rul = loader.load_test_rul()

    # normalize with the scaler already fit on FD001 train data -- refitting
    # here would erase the distributional difference this test is meant to measure
    test_df = preprocessor.normalize(test_df)
    # create_sequences needs a RUL column but test trajectories are truncated
    # before failure, so the true label comes from RUL_<subset>.txt instead
    test_df["RUL"] = 0
    X_test, _ = preprocessor.create_sequences(test_df, is_train=False)

    return X_test, true_rul


def plot_picp_by_subset(results: list[dict], target_coverage: float) -> None:
    subsets = [r["subset"] for r in results]
    picps = [r["picp"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(subsets, picps, color="steelblue")
    ax.axhline(target_coverage, linestyle=":", color="red", label=f"target ({target_coverage:.0%})")
    ax.set_ylabel("PICP")
    ax.set_ylim(0, 1.05)
    ax.set_title("PICP by subset (distribution shift)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "picp_by_subset.png", dpi=150)
    plt.close(fig)


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

    preprocessor = SequencePreprocessor(
        sequence_length=CONFIG["sequence_length"], rul_cap=CONFIG["rul_cap"]
    )
    # scaler is fit on FD001 train data only, same as train.py, since it's not
    # persisted to disk -- FD002/FD004 below reuse this same fitted scaler
    preprocessor.fit_scaler(train_df)

    train_df = preprocessor.compute_rul(train_df)
    train_df = preprocessor.normalize(train_df)
    _, _, X_calib, y_calib, _, _ = preprocessor.split_calibration_set(
        train_df, calib_fraction=CONFIG["calib_fraction"], seed=CONFIG["calib_seed"]
    )

    X_test, true_rul = load_test_subset(preprocessor, CONFIG["data_dir"], CONFIG["subset"])

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

    # distribution shift test -- same trained model and calibrated conformal
    # predictor, scored against subsets the calibration set never saw
    analyzer = DistributionShiftAnalyzer(model, conformal)
    results = [{"subset": CONFIG["subset"], "rmse": rmse, "picp": picp, "sharpness": sharpness}]

    for shift_subset in CONFIG["shift_subsets"]:
        X_shift, y_shift = load_test_subset(preprocessor, CONFIG["data_dir"], shift_subset)
        results.append(analyzer.run_shift_test(shift_subset, X_shift, y_shift, alpha=CONFIG["alpha"]))

    comparison_df = analyzer.compare_subsets(results)
    print("\nDistribution shift comparison:")
    print(comparison_df.to_string(index=False))

    plot_picp_by_subset(results, target_coverage=1 - CONFIG["alpha"])
    print(f"\nPICP-by-subset plot saved to {OUTPUT_DIR / 'picp_by_subset.png'}")


if __name__ == "__main__":
    main()
