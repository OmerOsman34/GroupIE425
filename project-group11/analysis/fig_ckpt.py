import os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import bt, models
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoint")
full, eve = bt.load()
fcs = models.all_models(full, eve)
B1 = fcs["B1"]; M = fcs["B1-evening x seasonal"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 8.5, "legend.fontsize": 7, "legend.frameon": False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1.55, 1]})
# panel a: city total, autumn 2025
a = ax[0]
rng = pd.date_range("2025-10-08", "2025-12-16")
city = full.loc[rng].sum(1); b = B1.loc[rng].sum(1); m = M.loc[rng].sum(1)
for s, e, lab in [("2025-11-09", "2025-11-20", "R1 days"), ("2025-11-23", "2025-12-01", "R2 days")]:
    a.axvspan(pd.Timestamp(s) - pd.Timedelta(hours=12), pd.Timestamp(e) + pd.Timedelta(hours=12), color="0.9", lw=0)
    a.text(pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2, 5450, lab, ha="center", fontsize=7, color="0.35")
a.plot(city.index, city, color="black", lw=1.3, label="actual")
a.plot(b.index, b, color="#c0392b", lw=1.0, ls="--", label="B1")
a.plot(m.index, m, color="#2471a3", lw=1.0, label="B1-evening x seasonal")
a.set_ylabel("bikes returned per day (city)")
a.set_title("(a) Autumn 2025, sum of the five cluster forecasts")
a.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2)); a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
a.set_ylim(0, 5900); a.legend(loc="lower left")
# panel b: median realised ratio y / B1 by date, 2022-2025 (city)
bb = ax[1]
q = (full.sum(1) / B1.sum(1))
rows = []
for y in (2022, 2023, 2024, 2025):
    r = q.loc["%d-10-01" % y: "%d-12-20" % y]
    r.index = r.index.map(lambda d: pd.Timestamp(2000, d.month, d.day))
    rows.append(r.rename(y))
Q = pd.concat(rows, axis=1)
for y in Q.columns:
    bb.plot(Q.index, Q[y], color="0.75", lw=0.6)
bb.plot(Q.index, Q.median(1).rolling(15, center=True, min_periods=5).median(), color="#2471a3", lw=1.6, label="median 2022-25, smoothed")
bb.axhline(1, color="black", lw=0.6)
for s, e in [("2000-11-09", "2000-11-20"), ("2000-11-23", "2000-12-01")]:
    bb.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="0.9", lw=0, zorder=0)
bb.set_ylim(0.3, 1.7); bb.set_ylabel(r"actual / B1 forecast (city)")
bb.set_title("(b) Actual / B1 by date, 2022-2025")
bb.xaxis.set_major_locator(mdates.MonthLocator()); bb.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
bb.legend(loc="lower left")
fig.tight_layout(pad=0.3, w_pad=1.2)
fig.savefig(os.path.join(OUT, "fig1.pdf"))
print("ok")
