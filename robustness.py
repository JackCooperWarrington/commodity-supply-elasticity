"""
Robustness Checks
=================

Two checks on the main inventory-surprise finding:

  1. SUBSAMPLE STABILITY   - Does the relationship between inventory
                              surprises and WTI returns hold in both the
                              2010-2019 and 2020-2026 halves of the sample,
                              or is it driven by one period?

  2. SEASONAL WINDOW LENGTH - Does the result depend on the specific choice
                              of a 5-year trailing window for the seasonal
                              expectation, or does it hold using 3-year and
                              7-year windows as well?

Usage:  python robustness.py
"""

import warnings

import numpy as np
import pandas as pd

from inventory_elasticity import (
    load_inventory_data, load_wti, build_seasonal_surprise,
    attach_price_reaction, surprise_regression, START,
)

warnings.filterwarnings("ignore")

SPLIT_DATE = pd.Timestamp("2020-01-01")


def subsample_check(inv_full: pd.DataFrame, wti: pd.Series) -> pd.DataFrame:
    inv = build_seasonal_surprise(inv_full, window_years=5)
    inv = inv[inv.week_ending >= START].reset_index(drop=True)
    inv = attach_price_reaction(inv, wti)

    rows = []
    for label, mask in [
        ("2010-2019", inv.week_ending < SPLIT_DATE),
        ("2020-2026", inv.week_ending >= SPLIT_DATE),
    ]:
        sub = inv[mask]
        fit, d = surprise_regression(sub)
        rows.append({
            "period": label,
            "n": len(d),
            "coefficient": fit.params["surprise_mmbbl"],
            "p_value": fit.pvalues["surprise_mmbbl"],
            "corr": d.surprise_mmbbl.corr(d.return_pct),
        })
    return pd.DataFrame(rows)


def window_sensitivity_check(inv_full: pd.DataFrame, wti: pd.Series) -> pd.DataFrame:
    rows = []
    for window in [3, 5, 7]:
        inv = build_seasonal_surprise(inv_full, window_years=window)
        inv = inv[inv.week_ending >= START].reset_index(drop=True)
        inv = attach_price_reaction(inv, wti)
        fit, d = surprise_regression(inv)
        rows.append({
            "seasonal_window_years": window,
            "n": len(d),
            "coefficient": fit.params["surprise_mmbbl"],
            "p_value": fit.pvalues["surprise_mmbbl"],
            "corr": d.surprise_mmbbl.corr(d.return_pct),
        })
    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    inv_full = load_inventory_data()
    wti = load_wti()

    print("\nSubsample stability (5-year seasonal window)")
    print("=" * 66)
    sub = subsample_check(inv_full, wti)
    print(sub.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nSensitivity to seasonal window length (full sample)")
    print("=" * 66)
    win = window_sensitivity_check(inv_full, wti)
    print(win.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    return sub, win


if __name__ == "__main__":
    main()
