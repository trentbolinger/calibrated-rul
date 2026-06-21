"""Tiny smoke test for SequencePreprocessor.compute_rul.

Loads FD001 train data, computes capped RUL, and plots capped vs. uncapped
RUL over time for one engine so the cap is visible: a flat line early in
the engine's life, then a 1:1 slope down to zero near failure.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from data.loader import CMAPSSLoader
from data.preprocessor import SequencePreprocessor

DATA_DIR = "data/raw"
OUTPUT_DIR = Path("outputs")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_df = CMAPSSLoader(DATA_DIR, subset="FD001").load_train()

    preprocessor = SequencePreprocessor(rul_cap=125)
    capped_df = preprocessor.compute_rul(train_df)

    max_cycle = train_df.groupby("unit_number")["time_cycle"].transform("max")
    capped_df["RUL_uncapped"] = max_cycle - train_df["time_cycle"]

    engine_id = capped_df["unit_number"].iloc[0]
    engine_df = capped_df[capped_df["unit_number"] == engine_id]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(engine_df["time_cycle"], engine_df["RUL_uncapped"], label="uncapped RUL")
    ax.plot(engine_df["time_cycle"], engine_df["RUL"], label=f"capped RUL (cap={preprocessor.rul_cap})")
    ax.set_xlabel("time cycle")
    ax.set_ylabel("RUL")
    ax.set_title(f"FD001 engine {engine_id}: capped vs. uncapped RUL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rul_cap_comparison_fd001.png", dpi=150)
    plt.close(fig)

    print(f"engine {engine_id}: {len(engine_df)} cycles, max uncapped RUL = {engine_df['RUL_uncapped'].max()}")
    print(capped_df[["unit_number", "time_cycle", "RUL_uncapped", "RUL"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
