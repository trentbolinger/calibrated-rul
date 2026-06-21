"""Tiny smoke test for SequencePreprocessor.split_calibration_set.

Loads FD001 train data, splits engines into train/calibration groups,
and confirms the split happened at the engine level (not the window
level): prints engine and window counts per group, then asserts the
two engine ID sets are disjoint.
"""

from data.loader import CMAPSSLoader
from data.preprocessor import SequencePreprocessor

DATA_DIR = "data/raw"


def main() -> None:
    train_df = CMAPSSLoader(DATA_DIR, subset="FD001").load_train()

    preprocessor = SequencePreprocessor(sequence_length=30)
    train_df = preprocessor.compute_rul(train_df)

    X_train, y_train, X_calib, y_calib, train_ids, calib_ids = preprocessor.split_calibration_set(
        train_df, calib_fraction=0.2
    )

    print(f"train engines: {len(train_ids)}, calibration engines: {len(calib_ids)}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_calib shape: {X_calib.shape}, y_calib shape: {y_calib.shape}")

    overlap = set(train_ids) & set(calib_ids)
    assert len(overlap) == 0, f"FAIL: train/calibration engine IDs overlap: {overlap}"
    print("PASS: train and calibration engine IDs are disjoint")


if __name__ == "__main__":
    main()
