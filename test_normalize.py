"""Tiny smoke test for SequencePreprocessor.fit_scaler / normalize.

Loads FD001 train data, fits the scaler, normalizes it, and prints
mean/std of a couple of sensor columns before and after to confirm
normalization worked (mean ~0, std ~1 after).
"""

from data.loader import CMAPSSLoader
from data.preprocessor import SequencePreprocessor

DATA_DIR = "data/raw"
SENSORS_TO_CHECK = ["sensor_2", "sensor_11"]


def main() -> None:
    train_df = CMAPSSLoader(DATA_DIR, subset="FD001").load_train()

    preprocessor = SequencePreprocessor()
    preprocessor.fit_scaler(train_df)
    normalized_df = preprocessor.normalize(train_df)

    for sensor in SENSORS_TO_CHECK:
        before_mean, before_std = train_df[sensor].mean(), train_df[sensor].std()
        after_mean, after_std = normalized_df[sensor].mean(), normalized_df[sensor].std()
        print(f"{sensor} before: mean={before_mean:.4f} std={before_std:.4f}")
        print(f"{sensor} after:  mean={after_mean:.4f} std={after_std:.4f}")


if __name__ == "__main__":
    main()
