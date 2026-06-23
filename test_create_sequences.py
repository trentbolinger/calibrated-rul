"""Tiny smoke test for SequencePreprocessor.create_sequences.

Loads FD001 train and test data, computes RUL on both (for test this is
just the truncated-trajectory RUL, not the true held-out RUL from
RUL_FD001.txt -- that substitution happens later in the real pipeline;
here we only care about sanity-checking the windowing shapes/counts),
then builds sequences and prints shapes plus per-engine window counts.
"""

from data.loader import CMAPSSLoader
from data.preprocessor import SequencePreprocessor

DATA_DIR = "data/raw"


def main() -> None:
    loader = CMAPSSLoader(DATA_DIR, subset="FD001")
    train_df = loader.load_train()
    test_df = loader.load_test()

    preprocessor = SequencePreprocessor(sequence_length=30)
    train_df = preprocessor.compute_rul(train_df)
    test_df = preprocessor.compute_rul(test_df)

    X_train, y_train = preprocessor.create_sequences(train_df, is_train=True)
    X_test, y_test, valid_engine_ids = preprocessor.create_sequences(test_df, is_train=False)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    sample_engines = train_df["unit_number"].unique()[:3]
    for engine_id in sample_engines:
        engine_df = train_df[train_df["unit_number"] == engine_id]
        X_engine, _ = preprocessor.create_sequences(engine_df, is_train=True)
        print(f"train engine {engine_id}: {len(engine_df)} cycles -> {len(X_engine)} windows")

    for engine_id in test_df["unit_number"].unique()[:3]:
        engine_df = test_df[test_df["unit_number"] == engine_id]
        X_engine, _, _ = preprocessor.create_sequences(engine_df, is_train=False)
        print(f"test engine {engine_id}: {len(engine_df)} cycles -> {len(X_engine)} window(s)")


if __name__ == "__main__":
    main()
