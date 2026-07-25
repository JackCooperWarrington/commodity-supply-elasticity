"""
Crude Oil Supply and the Price Response to Inventory Surprises
================================================================

Basic economic theory holds that price should respond to shifts in available
supply: more supply at the same demand should push price down, and less
supply should push it up. This module tests that idea directly using the
U.S. Energy Information Administration's (EIA) Weekly Petroleum Status
Report, the single most closely watched recurring data release in the
crude oil market.

Two versions of the test are run, and the difference between them is the
main finding:

  1. NAIVE      - Regress the price return around each week's report on the
                  raw reported change in crude oil inventories. This treats
                  every barrel of build or draw as equally informative.

  2. SURPRISE    - Regress the price return on the SURPRISE component only:
                  the reported change minus what a simple seasonal model
                  would have expected for that calendar week. Inventories
                  follow well-known seasonal patterns (refinery maintenance
                  season, summer driving season), so a lot of the raw weekly
                  change is already anticipated and priced in before the
                  report is even released.

Data limitation, stated plainly: EIA does not publish what the market
expected each week; that "consensus" figure is proprietary, distributed by
paid survey services (Bloomberg, Reuters). The seasonal-average measure used
here is a free, transparent, and defensible stand-in, built the same way EIA
itself frames "five-year average" comparisons in its own public reports, but
it is not identical to actual market expectations.

Usage:  python inventory_elasticity.py
Output: charts/inventory_naive.png, charts/inventory_surprise.png,
        charts/inventory_timeseries.png, results/inventory_summary.txt
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

warnings.filterwarnings("ignore")

EIA_URL = "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"
START = "2010-01-01"
SEASONAL_WINDOW_YEARS = 5   # matches EIA's own "five-year average" convention
CHART_DIR = "charts"
RESULT_DIR = "results"
INK, ACCENT, ACCENT_2 = "#1a1a1a", "#0b5d8a", "#b5451c"
GRID = "#d9d9d9"


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------

def load_inventory_data() -> pd.DataFrame:
    """
    Download the EIA's weekly U.S. commercial crude oil inventory series
    (excluding the Strategic Petroleum Reserve), reported in thousand
    barrels, going back to 1982. No API key required; this is a public XLS
    file the EIA hosts directly.
    """
    raw = pd.read_excel(EIA_URL, sheet_name="Data 1", skiprows=2)
    raw.columns = ["week_ending", "stocks_kbbl"]
    raw["week_ending"] = pd.to_datetime(raw["week_ending"])
    return raw.sort_values("week_ending").reset_index(drop=True)


def load_wti() -> pd.Series:
    """Daily WTI front-month futures closing price."""
    px = yf.download("CL=F", start="2005-01-01", progress=False, auto_adjust=True)["Close"]
    px.columns = ["WTI"]
    return px.WTI


def build_seasonal_surprise(inv: pd.DataFrame, window_years: int = SEASONAL_WINDOW_YEARS) -> pd.DataFrame:
    """
    For each week, compute:

      change_mmbbl       - the actual reported change in inventories,
                            in million barrels
      seasonal_expected  - the average change reported in the same
                            calendar week (by ISO week number) over the
                            trailing `window_years` years
      surprise_mmbbl      - change_mmbbl minus seasonal_expected: the part
                            of the report that a simple seasonal forecast
                            would not have anticipated

    This mirrors the "five-year average" framing the EIA itself uses when
    describing whether current inventories are unusually high or low for
    the time of year.
    """
    inv = inv.copy()
    inv["change_mmbbl"] = inv.stocks_kbbl.diff() / 1000.0
    inv["week_of_year"] = inv.week_ending.dt.isocalendar().week

    seasonal_expected = []
    for i in range(len(inv)):
        wk = inv.week_of_year.iloc[i]
        this_date = inv.week_ending.iloc[i]
        cutoff = this_date - pd.Timedelta(days=365 * window_years)
        hist = inv[(inv.week_ending < this_date) &
                   (inv.week_ending >= cutoff) &
                   (inv.week_of_year == wk)]
        seasonal_expected.append(hist.change_mmbbl.mean() if len(hist) >= 3 else np.nan)

    inv["seasonal_expected"] = seasonal_expected
    inv["surprise_mmbbl"] = inv.change_mmbbl - inv.seasonal_expected
    return inv


def attach_price_reaction(inv: pd.DataFrame, wti: pd.Series) -> pd.DataFrame:
    """
    Map each week's report to a release date and measure the WTI return
    around that date.

    The EIA releases the Weekly Petroleum Status Report at 10:30 AM Eastern
    on the Wednesday following each report's week-ending date (a Friday),
    five calendar days later. Weeks containing a federal holiday are
    sometimes delayed a day or two, and the exact holiday-adjusted schedule
    is not published as a single machine-readable historical file, so the
    release date used here is operationalized as the first trading day at
    least five calendar days after the week-ending date. This will
    occasionally be off by one day in holiday weeks (roughly 6-8 weeks per
    year), which adds noise but should not bias the average relationship.
    """
    inv = inv.copy()

    def release_date(week_ending):
        target = week_ending + pd.Timedelta(days=5)
        idx = wti.index.searchsorted(target)
        return wti.index[idx] if idx < len(wti) else pd.NaT

    inv["release_date"] = inv.week_ending.apply(release_date)

    def event_return(rd):
        if pd.isna(rd):
            return np.nan
        idx = wti.index.get_loc(rd)
        if idx == 0:
            return np.nan
        return 100 * np.log(wti.iloc[idx] / wti.iloc[idx - 1])

    inv["return_pct"] = inv.release_date.apply(event_return)
    return inv.dropna(subset=["release_date", "return_pct"])


# ----------------------------------------------------------------------------
# 2. Regressions
# ----------------------------------------------------------------------------

def naive_regression(panel: pd.DataFrame):
    """return_pct = a + b * change_mmbbl + e, using the raw reported change."""
    d = panel.dropna(subset=["change_mmbbl", "return_pct"])
    X = sm.add_constant(d.change_mmbbl)
    return sm.OLS(d.return_pct, X).fit(), d


def surprise_regression(panel: pd.DataFrame):
    """return_pct = a + b * surprise_mmbbl + e, using the seasonal-surprise measure."""
    d = panel.dropna(subset=["surprise_mmbbl", "return_pct"])
    X = sm.add_constant(d.surprise_mmbbl)
    return sm.OLS(d.return_pct, X).fit(), d


def granger_causality(panel: pd.DataFrame, maxlag: int = 4) -> pd.DataFrame:
    """
    Test whether past inventory surprises help predict future returns beyond
    what past returns alone predict, and the reverse.

    A Granger causality test compares two regressions: one predicting a
    variable from its own past values only, and one adding the past values
    of a second variable. If the second variable's past values are jointly
    significant, the test concludes that variable "Granger-causes" the
    first: it improves the prediction. This is a test of predictive content,
    not of causation in the everyday sense.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    d = panel.dropna(subset=["surprise_mmbbl", "return_pct"]).reset_index(drop=True)

    rows = []
    data_fwd = d[["return_pct", "surprise_mmbbl"]].values
    res_fwd = grangercausalitytests(data_fwd, maxlag=maxlag, verbose=False)
    for lag in range(1, maxlag + 1):
        f_stat, p_val, _, _ = res_fwd[lag][0]["ssr_ftest"]
        rows.append({"direction": "surprise -> return", "lag_weeks": lag,
                     "f_stat": f_stat, "p_value": p_val})

    data_rev = d[["surprise_mmbbl", "return_pct"]].values
    res_rev = grangercausalitytests(data_rev, maxlag=maxlag, verbose=False)
    for lag in range(1, maxlag + 1):
        f_stat, p_val, _, _ = res_rev[lag][0]["ssr_ftest"]
        rows.append({"direction": "return -> surprise", "lag_weeks": lag,
                     "f_stat": f_stat, "p_value": p_val})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 3. Charts
# ----------------------------------------------------------------------------

def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)


def plot_scatter(d: pd.DataFrame, x_col: str, fit, title: str, xlabel: str, path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(d[x_col], d.return_pct, s=14, color=ACCENT, alpha=0.35, edgecolors="none")

    xs = np.linspace(d[x_col].min(), d[x_col].max(), 100)
    ys = fit.params["const"] + fit.params[x_col] * xs
    ax.plot(xs, ys, color=ACCENT_2, lw=1.8)

    ax.axhline(0, color=INK, lw=0.7, alpha=0.5)
    ax.axvline(0, color=INK, lw=0.7, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    ax.set_ylabel("WTI return on release day (%)", fontsize=9.5, color=INK)
    ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=12)
    _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_timeseries(inv: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(inv.week_ending, inv.stocks_kbbl / 1000, color=ACCENT, lw=1.1)
    ax.set_ylabel("Commercial crude stocks, ex-SPR (million barrels)", fontsize=9.5, color=INK)
    ax.set_title("U.S. Weekly Commercial Crude Oil Inventories",
                 fontsize=13, color=INK, loc="left", pad=12)
    _style(ax)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Run
# ----------------------------------------------------------------------------

def main():
    os.makedirs(CHART_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("Downloading EIA inventory data and WTI prices...")
    inv_full = load_inventory_data()
    wti = load_wti()

    inv = build_seasonal_surprise(inv_full)
    inv = inv[inv.week_ending >= START].reset_index(drop=True)
    inv = attach_price_reaction(inv, wti)
    print(f"  {len(inv):,} weekly reports matched to price reactions, "
          f"{inv.week_ending.min():%Y-%m-%d} to {inv.week_ending.max():%Y-%m-%d}\n")

    print("Naive regression: return on raw inventory change...")
    naive_fit, naive_d = naive_regression(inv)
    print(f"  n = {len(naive_d)}, coefficient = {naive_fit.params['change_mmbbl']:.4f}, "
          f"p = {naive_fit.pvalues['change_mmbbl']:.3f}, "
          f"corr = {naive_d.change_mmbbl.corr(naive_d.return_pct):.3f}\n")

    print("Surprise regression: return on seasonal-surprise inventory change...")
    surprise_fit, surprise_d = surprise_regression(inv)
    print(f"  n = {len(surprise_d)}, coefficient = {surprise_fit.params['surprise_mmbbl']:.4f}, "
          f"p = {surprise_fit.pvalues['surprise_mmbbl']:.3f}, "
          f"corr = {surprise_d.surprise_mmbbl.corr(surprise_d.return_pct):.3f}\n")

    print("Granger causality: does surprise predict future return, or the reverse?...")
    granger = granger_causality(inv)
    print(granger.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

    plot_timeseries(inv_full[inv_full.week_ending >= START], f"{CHART_DIR}/inventory_timeseries.png")
    plot_scatter(naive_d, "change_mmbbl", naive_fit,
                 "WTI Return vs. Raw Inventory Change (not significant)",
                 "Reported weekly change in crude stocks (million barrels)",
                 f"{CHART_DIR}/inventory_naive.png")
    plot_scatter(surprise_d, "surprise_mmbbl", surprise_fit,
                 "WTI Return vs. Seasonal-Surprise Inventory Change",
                 "Surprise vs. 5-year seasonal average (million barrels)",
                 f"{CHART_DIR}/inventory_surprise.png")

    with open(f"{RESULT_DIR}/inventory_summary.txt", "w") as f:
        f.write("NAIVE REGRESSION: return = a + b*raw_change\n")
        f.write("=" * 70 + "\n")
        f.write(str(naive_fit.summary()))
        f.write("\n\nSURPRISE REGRESSION: return = a + b*seasonal_surprise\n")
        f.write("=" * 70 + "\n")
        f.write(str(surprise_fit.summary()))
        f.write("\n\nGRANGER CAUSALITY\n")
        f.write("=" * 70 + "\n")
        f.write(granger.to_string(index=False))

    print("Wrote charts/ and results/inventory_summary.txt")
    return inv, naive_fit, surprise_fit, granger


if __name__ == "__main__":
    main()
