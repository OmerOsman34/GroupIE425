"""Backtest framework for the IE425 Oslo bike-arrivals project (Group 11).

Information set at the origin of target day d (21:00 Oslo on d-1):
  * full-day cluster counts for days <= d-2
  * before-21:00 cluster counts for day d-1
Every forecaster below is written as a function of shifted series only, so a
forecast for d can never see y[d] or the full count of d-1.
"""
import os
import numpy as np
import pandas as pd

CLUSTERS = [1, 2, 3, 4, 5]
NAMES = {1: "Center", 2: "East", 3: "North", 4: "North-west", 5: "West"}
# The uncut tables that the self-check kit writes to its delivered/ folder (never committed).
DATA = os.environ.get("KIT_DELIVERED", os.path.join("..", "..", "project-self-check-kit-main", "delivered"))


# ----------------------------------------------------------------------------- loading
def load():
    """Return (full, eve): days x clusters frames on a gap-free daily index."""
    cd = pd.read_csv(os.path.join(DATA, "cluster_daily_inflow.csv"), parse_dates=["date"])
    full = cd.set_index("date")[["cluster%d" % j for j in CLUSTERS]]
    full.columns = CLUSTERS
    sd = pd.read_csv(os.path.join(DATA, "station_day_arrivals.csv"), parse_dates=["date"])
    st = pd.read_csv(os.path.join(DATA, "stations.csv"))
    st = st[(st["scored"] == 1) & st["cluster"].notna()]
    cl = dict(zip(st["station_id"], st["cluster"].astype(int)))
    sd = sd[sd["station_id"].isin(cl)]
    sd["cluster"] = sd["station_id"].map(cl)
    eve = sd.pivot_table(index="date", columns="cluster", values="arrivals_before_21",
                         aggfunc="sum")
    chk = sd.pivot_table(index="date", columns="cluster", values="arrivals_full", aggfunc="sum")
    days = pd.date_range(full.index.min(), full.index.max(), freq="D")
    full, eve, chk = full.reindex(days), eve.reindex(days), chk.reindex(days)
    gap = (chk[CLUSTERS] - full[CLUSTERS]).abs().to_numpy()
    print("station sums vs cluster file: max abs diff %.1f" % np.nanmax(gap))
    full.index.name = eve.index.name = "date"
    return full.astype(float), eve[CLUSTERS].astype(float)


# ----------------------------------------------------------------------------- windows
def window(year, start, end):
    return list(pd.date_range("%d-%s" % (year, start), "%d-%s" % (year, end), freq="D"))


def folds(years=(2022, 2023, 2024, 2025)):
    """The test-round calendar windows replayed in past years."""
    out = {}
    for y in years:
        out["R1-%d" % y] = window(y, "11-09", "11-20")
        out["R2-%d" % y] = window(y, "11-23", "12-01")
    return out


def autumn(years=(2022, 2023, 2024, 2025)):
    """The brief's yardstick window: 8 Oct - 16 Dec."""
    return {"Aut-%d" % y: window(y, "10-08", "12-16") for y in years}


# ----------------------------------------------------------------------------- scoring
def score(fc, actual, days):
    """WMAPE over five clusters, bias, per-cluster WMAPE, city WMAPE on `days`."""
    days = [d for d in days if d in fc.index and d in actual.index]
    a = actual.loc[days, CLUSTERS].to_numpy()
    f = fc.loc[days, CLUSTERS].to_numpy()
    ok = ~np.isnan(f).any(axis=1) & ~np.isnan(a).any(axis=1)
    a, f = a[ok], f[ok]
    out = {"days": int(ok.sum()),
           "wmape": np.abs(a - f).sum() / a.sum(),
           "bias": f.sum() / a.sum(),
           "city": np.abs(a.sum(1) - f.sum(1)).sum() / a.sum()}
    for k, j in enumerate(CLUSTERS):
        out["c%d" % j] = np.abs(a[:, k] - f[:, k]).sum() / a[:, k].sum()
    return out


def table(fcs, actual, fold_map):
    """Rows = (model, fold); pooled rows per model over all folds."""
    rows = []
    for name, fc in fcs.items():
        for fold, days in fold_map.items():
            s = score(fc, actual, days)
            s.update(model=name, fold=fold)
            rows.append(s)
        alld = sorted({d for ds in fold_map.values() for d in ds})
        s = score(fc, actual, alld)
        s.update(model=name, fold="ALL")
        rows.append(s)
    t = pd.DataFrame(rows).set_index(["model", "fold"])
    return t[["days", "wmape", "bias", "city"] + ["c%d" % j for j in CLUSTERS]]


# ----------------------------------------------------------------------------- baselines
def b1(full):
    """Brief's B1, vectorised: four * (m7 / m28), complete days up to d-2."""
    four = sum(full.shift(k) for k in (7, 14, 21, 28)) / 4.0
    m7 = full.shift(2).rolling(7).mean()
    m28 = full.shift(2).rolling(28).mean()
    r = (m7 / m28).where(m28 > 0, 1.0)
    return four * r


def last_week(full):
    return full.shift(7)
