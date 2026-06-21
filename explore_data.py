"""Exploratory analysis of the C-MAPSS dataset.

Confirms the loader works, that degradation trends are visible in the raw
sensors, and quantifies how FD002/FD004 differ from FD001 in operating
conditions -- the distribution shift this project is built to stress-test.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data.loader import CMAPSSLoader

DATA_DIR = "data/raw"
OUTPUT_DIR = Path("outputs")

SENSORS_TO_PLOT = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_15"]
N_ENGINES_TO_PLOT = 5

# published train trajectory counts per the C-MAPSS documentation; used to catch
# a mislabeled/swapped train-test file before it quietly corrupts the comparison.
# Note: FD004 is 249 here vs. 248 in the original NASA paper -- a known minor
# discrepancy in the behrad3d/nasa-cmaps Kaggle mirror, confirmed not a
# train/test mislabeling (train avg. ~246 cycles/run-to-failure vs. test's
# ~166 cycles/truncated, as expected).
EXPECTED_TRAIN_ENGINE_COUNTS = {"FD001": 100, "FD002": 260, "FD003": 100, "FD004": 249}


def plot_sensor_trends(train_df: pd.DataFrame) -> None:
    engine_ids = train_df["unit_number"].unique()[:N_ENGINES_TO_PLOT]
    fig, axes = plt.subplots(
        len(SENSORS_TO_PLOT), 1, figsize=(10, 2.5 * len(SENSORS_TO_PLOT)), sharex=True
    )
    for ax, sensor in zip(axes, SENSORS_TO_PLOT):
        for engine_id in engine_ids:
            engine_df = train_df[train_df["unit_number"] == engine_id]
            ax.plot(engine_df["time_cycle"], engine_df[sensor], label=f"engine {engine_id}")
        ax.set_ylabel(sensor)
    axes[-1].set_xlabel("time cycle")
    axes[0].legend(loc="upper right", fontsize="small")
    fig.suptitle("FD001: sensor readings over time for sample engines")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sensor_trends_fd001.png", dpi=150)
    plt.close(fig)


def plot_rul_distribution(train_df: pd.DataFrame) -> None:
    max_cycle = train_df.groupby("unit_number")["time_cycle"].transform("max")
    rul = max_cycle - train_df["time_cycle"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rul, bins=50)
    ax.set_xlabel("RUL (cycles)")
    ax.set_ylabel("count")
    ax.set_title("FD001: RUL distribution (uncapped, train set)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rul_distribution_fd001.png", dpi=150)
    plt.close(fig)


def count_operating_conditions(df: pd.DataFrame) -> int:
    # op settings are continuous but cluster tightly around a small number of
    # discrete regimes; 2-decimal rounding still picks up simulation noise
    # within a regime, so round to whole units to collapse each regime to one row
    rounded = df[["op_setting_1", "op_setting_2", "op_setting_3"]].round(0)
    return len(rounded.drop_duplicates())


def build_comparison_table() -> pd.DataFrame:
    rows = []
    for subset in ["FD001", "FD002", "FD004"]:
        train_df = CMAPSSLoader(DATA_DIR, subset=subset).load_train()
        engine_count = train_df["unit_number"].nunique()

        expected = EXPECTED_TRAIN_ENGINE_COUNTS[subset]
        if engine_count != expected:
            print(
                f"WARNING: {subset} train_{subset}.txt has {engine_count} engines, "
                f"expected {expected}. This usually means train/test files were "
                f"swapped or mislabeled during download -- verify the raw file "
                f"before trusting this row."
            )

        rows.append(
            {
                "subset": subset,
                "engine_count": engine_count,
                "operating_conditions": count_operating_conditions(train_df),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_df = CMAPSSLoader(DATA_DIR, subset="FD001").load_train()
    plot_sensor_trends(train_df)
    plot_rul_distribution(train_df)

    comparison = build_comparison_table()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
