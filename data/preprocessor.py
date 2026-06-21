"""Sequence preprocessing for the C-MAPSS dataset."""

import pandas as pd


class SequencePreprocessor:
    def __init__(self, sequence_length: int = 30, rul_cap: int = 125):
        self.sequence_length = sequence_length
        self.rul_cap = rul_cap

    def compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        max_cycle = df.groupby("unit_number")["time_cycle"].transform("max")
        rul = max_cycle - df["time_cycle"]
        df["RUL"] = rul.clip(upper=self.rul_cap)
        return df
