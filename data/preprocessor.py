"""Sequence preprocessing for the C-MAPSS dataset."""

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
