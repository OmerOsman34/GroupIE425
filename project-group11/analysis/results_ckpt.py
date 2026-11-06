import pandas as pd, numpy as np, json
import bt, models
pd.set_option("display.width", 220); pd.set_option("display.precision", 3)
full, eve = bt.load()
fcs = models.all_models(full, eve)
f = bt.folds()
groups = {
  "R1 2022-24 (tune)": [d for k, v in f.items() if k.startswith("R1") and not k.endswith("2025") for d in v],
  "R2 2022-24 (tune)": [d for k, v in f.items() if k.startswith("R2") and not k.endswith("2025") for d in v],
  "R1 2025 (check)": f["R1-2025"],
  "R2 2025 (check)": f["R2-2025"],
  "All 84 days": [d for v in f.values() for d in v],
}
rows = []
for m, F in fcs.items():
    for g, days in groups.items():
        s = bt.score(F, full, days); s.update(model=m, group=g); rows.append(s)
T = pd.DataFrame(rows).set_index(["model", "group"])
print(T[["days","wmape","bias","city","c1","c2","c3","c4","c5"]])
W = T["wmape"].unstack("group")[list(groups)]
Bb = T["bias"].unstack("group")[list(groups)]
print(W.round(3)); print(Bb.round(2))
# per fold detail
print(bt.table(fcs, full, f)["wmape"].unstack("fold").round(3))
# spread across the 8 folds
per = bt.table(fcs, full, f)["wmape"].unstack("fold").drop(columns="ALL")
print("min/max across folds"); print(pd.DataFrame({"min": per.min(1), "max": per.max(1)}).round(3))
T.to_csv("results_ckpt.csv")
# brief yardstick
aut = bt.autumn()
Y = bt.table({k: fcs[k] for k in ["Last week", "B1"]}, full, aut).xs("ALL", level="fold")
print(Y)
Y.to_csv("yardstick.csv")
