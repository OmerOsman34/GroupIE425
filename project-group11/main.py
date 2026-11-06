"""IE 425 term project, Group 11 -- B1-evening x seasonal factor.

The forecast for target day d (origin: 21:00 Oslo on d-1) has two parts.

1. B1-evening, a version of the brief's B1 whose level ratio also uses day d-1:
       four    = mean(y[d-7], y[d-14], y[d-21], y[d-28])
       y~[d-1] = e[d-1] / p^,   e = arrivals before 21:00 on d-1,
                 p^ = mean share before 21:00 over the last 8 complete days with d-1's weekday
       r~      = mean(y~[d-1], y[d-2..d-7]) / mean(y~[d-1], y[d-2..d-28])
       E[d]    = four * r~
2. A seasonal factor learned from earlier years only:
       c(d) = median of y[t] / E[t] over the days t of earlier years that lie within
              +-HALF days of d's calendar date (days with the whole system closed left out)
   and the forecast is c(d) * E[d], per cluster.

fit_model() learns c for every calendar date of the target's year and p^ is recomputed from
the snapshot in make_forecast(), so the model stays valid if it was fitted on an earlier evening.
Validation (rolling origin, 9-20 Nov and 23 Nov-1 Dec of 2022-2025) is in analysis/.
"""
import numpy as np
import pandas as pd

import helper

CLUSTERS = [1, 2, 3, 4, 5]
LAGS = (7, 14, 21, 28)
N_SAME = 8          # same-weekday days behind the before-21:00 share
HALF = 14           # +-days around the same calendar date; chosen on the 2022-2024 folds
FIRST_YEAR = 2019   # first year of history that may enter the seasonal factor


def wide(by_day, last_day):
    """{day: {cluster: value}} as a days x clusters frame on a gap-free daily index.

    Gap-free matters: shift() moves rows, not dates, so one missing day would quietly turn a
    7-day lag into a 6-day one. The index runs to `last_day` (the target), so rows the snapshot
    does not hold yet are NaN rather than absent.
    """
    f = pd.DataFrame(by_day).T.sort_index()
    f.index = pd.to_datetime(f.index)
    f = f.reindex(columns=CLUSTERS).astype(float)
    days = pd.date_range(f.index.min(), pd.Timestamp(last_day), freq="D")
    return f.reindex(days)


def series(target):
    """Complete days (to d-2) and before-21:00 counts (to d-1), on one daily index."""
    full = wide(helper.cluster_daily(), target)
    eve = wide(helper.cluster_before21(), target).reindex(full.index)
    return full, eve


def b1(full):
    """The brief's B1 for every row of `full` (complete days up to d-2 only)."""
    four = sum(full.shift(k) for k in LAGS) / 4.0
    m7 = full.shift(2).rolling(7).mean()
    m28 = full.shift(2).rolling(28).mean()
    return four * (m7 / m28).where(m28 > 0, 1.0)


def b1_evening(full, eve):
    """B1 with day d-1 in the level ratio, estimated from its before-21:00 count."""
    share = (eve / full).where(full > 0)
    weekday = full.index.dayofweek
    p_hat = pd.DataFrame(np.nan, index=full.index, columns=full.columns)
    for k in range(7):
        rows = weekday == k
        # shift(1) within one weekday: the share of row t uses t-7, t-14, ... only
        p_hat.loc[rows] = share.loc[rows].rolling(N_SAME, min_periods=4).mean().shift(1).to_numpy()
    est_prev = (eve / p_hat).shift(1)                  # estimated whole day d-1, aligned to d
    four = sum(full.shift(k) for k in LAGS) / 4.0
    m7 = (est_prev + full.shift(2).rolling(6).sum()) / 7.0
    m28 = (est_prev + full.shift(2).rolling(27).sum()) / 28.0
    return four * (m7 / m28).where(m28 > 0, 1.0)


def seasonal_factors(full, eve, year):
    """{(month, day): {cluster: c}} for every calendar date of `year`, from earlier years only."""
    base = b1_evening(full, eve)
    ratio = (full / base).replace([np.inf, -np.inf], np.nan)
    open_days = full.sum(axis=1, min_count=1) > 0
    ratio = ratio.where(open_days, axis=0)
    ratio = ratio[(ratio.index.year >= FIRST_YEAR) & (ratio.index.year < year)]

    table = {}
    for day in pd.date_range("%d-01-01" % year, "%d-12-31" % year, freq="D"):
        parts = []
        for y in range(FIRST_YEAR, year):
            try:
                centre = pd.Timestamp(year=y, month=day.month, day=day.day)
            except ValueError:                      # 29 February in a common year
                continue
            parts.append(ratio.loc[centre - pd.Timedelta(days=HALF):
                                   centre + pd.Timedelta(days=HALF)])
        if parts:
            med = pd.concat(parts).median()
            table[(day.month, day.day)] = {c: float(med[c]) for c in CLUSTERS
                                           if np.isfinite(med[c]) and med[c] > 0}
    return table


def fit_model():
    """Learn the seasonal factors. Small: 366 dates x 5 clusters of floats."""
    target = pd.Timestamp(helper.target_date())
    full, eve = series(target)
    factors = seasonal_factors(full, eve, target.year)
    known = full.dropna(how="all").index.max()
    near = factors.get((target.month, target.day), {})
    print("fit_model: complete days to %s, factors for %d dates; for %s: %s"
          % (known.date(), len(factors), target.date(),
             {c: round(v, 3) for c, v in near.items()}))
    return {"factors": factors, "year": int(target.year)}


def make_forecast(model):
    target = pd.Timestamp(helper.target_date())
    full, eve = series(target)
    row_e = b1_evening(full, eve).loc[target]
    row_b = b1(full).loc[target]
    row_w = full.shift(7).loc[target]

    factors = (model or {}).get("factors", {})
    near = factors.get((target.month, target.day), {})

    out, used = {}, {}
    for c in CLUSTERS:
        # A missing input is a decision, not an accident: B1, then last week's value.
        base, how = row_e[c], "evening"
        if not np.isfinite(base):
            base, how = row_b[c], "B1"
        if not np.isfinite(base):
            base, how = row_w[c], "last week"
        if not np.isfinite(base):
            base, how = 0.0, "zero"
        f = near.get(c, 1.0)
        if not np.isfinite(f) or f <= 0:
            f = 1.0
        out[c] = max(0.0, float(base) * float(f))
        used[c] = "%s x %.3f" % (how, f)

    print("make_forecast: %s -> %s  (%s)"
          % (target.date(), {c: round(v, 1) for c, v in out.items()}, used))
    return out


if __name__ == "__main__":        # never runs in the box or in the evaluation
    m = fit_model()
    print(make_forecast(m))
