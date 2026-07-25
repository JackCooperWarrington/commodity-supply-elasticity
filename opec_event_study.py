"""
OPEC and OPEC+ Production Decisions: A Small-Sample Event Study
=================================================================

Unlike Federal Reserve meetings, OPEC and OPEC+ meetings are not held on a
fixed, pre-announced annual schedule. Emergency meetings, delayed meetings,
and virtual meetings are common, and no single free, machine-readable
historical calendar of every OPEC decision exists the way one does for the
FOMC. Rather than compile a long list of routine meetings that risks
transcription error, this module examines a small number of decisions that
are unambiguous, independently documented turning points in oil market
history, each verified against multiple contemporaneous news sources.

This means the sample is intentionally small (six events) and the module
does not attempt formal hypothesis testing with the same statistical power
as the FOMC study in the companion options project. It reports what
happened around each event plainly and lets the reader see the magnitude
directly, in the same spirit as a case-study approach common in applied
macro-finance when a fixed announcement calendar does not exist.

Usage:  python opec_event_study.py
Output: charts/opec_events.png, results/opec_summary.txt
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

CHART_DIR = "charts"
RESULT_DIR = "results"
INK, ACCENT, ACCENT_2 = "#1a1a1a", "#0b5d8a", "#b5451c"
GRID = "#d9d9d9"

# Six independently verified, unambiguous OPEC/OPEC+ decisions. Each date and
# description was checked against multiple contemporaneous news sources
# (Reuters, CNBC, Washington Post, NPR, Al Jazeera, and the OPEC Secretariat's
# own press releases). This is not an exhaustive list of every OPEC meeting;
# it is restricted to decisions widely regarded as market-moving turning
# points, chosen specifically to avoid the transcription risk of compiling
# a long list of routine meetings without a canonical source to check against.
OPEC_EVENTS = [
    ("2014-11-27", "OPEC declines to cut output despite falling prices", "bearish"),
    ("2016-12-10", "Declaration of Cooperation: first OPEC+ deal", "bullish"),
    ("2020-03-06", "OPEC+ talks collapse; Saudi-Russia price war begins", "bearish"),
    ("2020-04-12", "OPEC+ agrees record 9.7 million bpd cut", "bullish"),
    ("2022-10-05", "OPEC+ agrees 2 million bpd cut", "bullish"),
    ("2023-04-02", "Surprise additional voluntary cuts of 1.16 million bpd", "bullish"),
]


def load_wti() -> pd.Series:
    px = yf.download("CL=F", start="2005-01-01", progress=False, auto_adjust=True)["Close"]
    px.columns = ["WTI"]
    return px.WTI


def event_window_returns(wti: pd.Series, event_date: str, window=range(-2, 3)) -> pd.Series:
    """Daily log returns (%) for the trading days surrounding an event date."""
    ts = pd.Timestamp(event_date)
    idx = wti.index.searchsorted(ts)
    idx = min(idx, len(wti) - 1)

    out = {}
    for offset in window:
        pos = idx + offset
        if 0 <= pos < len(wti) and pos - 1 >= 0:
            out[offset] = 100 * np.log(wti.iloc[pos] / wti.iloc[pos - 1])
    return pd.Series(out)


def build_event_table(wti: pd.Series) -> pd.DataFrame:
    rows = []
    for date_str, description, direction in OPEC_EVENTS:
        window_returns = event_window_returns(wti, date_str)
        rows.append({
            "date": date_str,
            "description": description,
            "expected_direction": direction,
            "return_on_day": window_returns.get(0, np.nan),
            "return_t_plus_1": window_returns.get(1, np.nan),
            "cumulative_t0_t1": window_returns.get(0, 0) + window_returns.get(1, 0),
        })
    return pd.DataFrame(rows)


def plot_events(wti: pd.Series, table: pd.DataFrame, path: str):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, table.iterrows()):
        ts = pd.Timestamp(row["date"])
        window = wti.loc[ts - pd.Timedelta(days=10): ts + pd.Timedelta(days=10)]
        ax.plot(window.index, window.values, color=ACCENT, lw=1.3)
        ax.axvline(ts, color=ACCENT_2, ls="--", lw=1.2)
        ax.set_title(f"{row['date']}\n{row['description'][:38]}",
                     fontsize=8.5, color=INK, loc="left")
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("WTI Around Six Landmark OPEC/OPEC+ Decisions", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Loading WTI prices...")
    wti = load_wti()

    table = build_event_table(wti)
    print("\nOPEC/OPEC+ event reactions:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))

    correct_direction = 0
    for _, row in table.iterrows():
        moved_as_expected = (
            (row["expected_direction"] == "bullish" and row["cumulative_t0_t1"] > 0) or
            (row["expected_direction"] == "bearish" and row["cumulative_t0_t1"] < 0)
        )
        correct_direction += int(moved_as_expected)
    print(f"\nEvents where price moved in the theoretically expected direction: "
          f"{correct_direction} of {len(table)}")

    plot_events(wti, table, f"{CHART_DIR}/opec_events.png")

    with open(f"{RESULT_DIR}/opec_summary.txt", "w") as f:
        f.write("OPEC/OPEC+ EVENT STUDY\n")
        f.write("=" * 70 + "\n")
        f.write(table.to_string(index=False))
        f.write(f"\n\nEvents matching expected direction: {correct_direction} of {len(table)}\n")

    print("Wrote charts/opec_events.png and results/opec_summary.txt")
    return table


if __name__ == "__main__":
    main()
