"""Sequence preprocessing for the C-MAPSS dataset."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


class SequencePreprocessor:
    def __init__(self, sequence_length: int = 30, rul_cap: int = 125):
        self.sequence_length = sequence_length
        self.rul_cap = rul_cap
        self.scaler = None

    def compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        max_cycle = df.groupby("unit_number")["time_cycle"].transform("max")
        rul = max_cycle - df["time_cycle"]
        df["RUL"] = rul.clip(upper=self.rul_cap)
        return df

    def fit_scaler(self, df: pd.DataFrame) -> None:
        self.scaler = StandardScaler()
        self.scaler.fit(df[SENSOR_COLUMNS])

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[SENSOR_COLUMNS] = self.scaler.transform(df[SENSOR_COLUMNS])
        return df

    def create_sequences(self, df: pd.DataFrame, is_train: bool) -> tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []

        for _, engine_df in df.groupby("unit_number"):
            engine_df = engine_df.sort_values("time_cycle")
            sensors = engine_df[SENSOR_COLUMNS].to_numpy()
            rul = engine_df["RUL"].to_numpy()
            n_cycles = len(engine_df)

            if n_cycles < self.sequence_length:
                # shorter than one window; can't happen for FD001 but would
                # otherwise wrap around negative indices below
                continue

            if is_train:
                starts = range(0, n_cycles - self.sequence_length + 1)
            else:
                starts = [n_cycles - self.sequence_length]

            for start in starts:
                end = start + self.sequence_length
                X_list.append(sensors[start:end])
                y_list.append(rul[end - 1])

        return np.array(X_list), np.array(y_list)

    def split_calibration_set(
        self, df: pd.DataFrame, calib_fraction: float = 0.4, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # split by engine ID, not by window -- windows from the same engine
        # must never end up in both groups, or conformal coverage numbers
        # get quietly inflated by leakage
        engine_ids = df["unit_number"].unique()
        n_calib = round(len(engine_ids) * calib_fraction)

        # a fixed seed lets evaluate.py reproduce the exact calibration split
        # train.py used, so conformal calibration never touches engines the
        # model was actually fit on
        rng = np.random.default_rng(seed)
        calib_ids = rng.choice(engine_ids, size=n_calib, replace=False)
        train_ids = np.setdiff1d(engine_ids, calib_ids)

        train_df = df[df["unit_number"].isin(train_ids)]
        calib_df = df[df["unit_number"].isin(calib_ids)]

        X_train, y_train = self.create_sequences(train_df, is_train=True)
        X_calib, y_calib = self.create_sequences(calib_df, is_train=True)

        return X_train, y_train, X_calib, y_calib, train_ids, calib_ids
