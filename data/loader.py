"""Loader for the raw NASA C-MAPSS turbofan engine degradation dataset.

Raw files (train_{subset}.txt, test_{subset}.txt, RUL_{subset}.txt) are
whitespace-delimited with no header row.
"""

from pathlib import Path

import numpy as np
import pandas as pd

VALID_SUBSETS = {"FD001", "FD002", "FD003", "FD004"}

COLUMN_NAMES = (
    ["unit_number", "time_cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


class CMAPSSLoader:
    def __init__(self, data_dir: str, subset: str = "FD001"):
        if subset not in VALID_SUBSETS:
            raise ValueError(f"subset must be one of {sorted(VALID_SUBSETS)}, got {subset!r}")
        self.data_dir = Path(data_dir)
        self.subset = subset

    def _read_engine_file(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep=r"\s+", header=None)
        # raw files sometimes have a trailing whitespace column that reads as a stray NaN column
        df = df.iloc[:, : len(COLUMN_NAMES)]
        df.columns = COLUMN_NAMES
        df["unit_number"] = df["unit_number"].astype(int)
        df["time_cycle"] = df["time_cycle"].astype(int)
        return df

    def load_train(self) -> pd.DataFrame:
        path = self.data_dir / f"train_{self.subset}.txt"
        return self._read_engine_file(path)

    def load_test(self) -> pd.DataFrame:
        path = self.data_dir / f"test_{self.subset}.txt"
        return self._read_engine_file(path)

    def load_test_rul(self) -> np.ndarray:
        path = self.data_dir / f"RUL_{self.subset}.txt"
        return pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
