"""Candidate forecasters, each a function of information available at the origin."""
import numpy as np
import pandas as pd
import bt

C = bt.CLUSTERS
HALF = 14          # chosen on the 2022-2024 folds only (3..28 tried)


def b1e(full, eve, n_same=8):
    """B1 whose level ratio also uses day d-1, estimated from its before-21:00 count.

    p_hat = mean share of a day's inflow that arrives before 21:00, over the last `n_same`
    complete days with the same weekday as d-1 (all of them <= d-8, so complete at the origin).
    """
    frac = (eve / full).where(full > 0)
    dow = pd.Series(full.index.dayofweek, index=full.index)
    p_est = pd.DataFrame(index=full.index, columns=C, dtype=float)
    for k in range(7):
        m = dow == k
        p_est.loc[m] = frac[m].rolling(n_same, min_periods=4).mean().shift(1)
    est_dm1 = (eve / p_est).shift(1)                      # estimated full count of d-1
    four = sum(full.shift(k) for k in (7, 14, 21, 28)) / 4.0
    m7 = (est_dm1 + full.shift(2).rolling(6).sum()) / 7.0       # d-1 .. d-7
    m28 = (est_dm1 + full.shift(2).rolling(27).sum()) / 28.0    # d-1 .. d-28
    return four * (m7 / m28).where(m28 > 0, 1.0)


def seasonal_corr(F, full, half=7, exclude=(2020,), months=(10, 11, 12)):
    """c_i(d) = median over earlier years of y/F within +-half days of the same date."""
    q = (full / F).replace([np.inf, -np.inf], np.nan)
    q = q.where(full.sum(axis=1) > 0, axis=0)       # a closed system (city total 0) says nothing
    out = pd.DataFrame(index=F.index, columns=C, dtype=float)
    for d in F.index:
        if d.month not in months:
            continue
        rows = []
        for y in range(2019, d.year):
            if y in exclude:
                continue
            try:
                c0 = pd.Timestamp(year=y, month=d.month, day=d.day)
            except ValueError:
                continue
            rows.append(q.loc[c0 - pd.Timedelta(days=half): c0 + pd.Timedelta(days=half)])
        if rows:
            out.loc[d] = pd.concat(rows).median()
    return out


def all_models(full, eve):
    B1 = bt.b1(full)
    E = b1e(full, eve)
    return {
        "Last week": bt.last_week(full),
        "B1": B1,
        "B1 x seasonal": B1 * seasonal_corr(B1, full, half=HALF, exclude=()),
        "B1-evening x seasonal": E * seasonal_corr(E, full, half=HALF, exclude=()),
    }
