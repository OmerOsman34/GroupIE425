"""The one tuning search behind main.py: the half-width h of the seasonal-factor window.

Scored on the 2022-2024 folds only (the tune folds). The 2025 folds are the final check and
are not used here. Run from this folder with KIT_DELIVERED pointing at the kit's delivered/.

Result on 6 Nov 2026 (WMAPE over the 63 tune days, B1-evening base):
    h =  3: 0.175   h =  5: 0.170   h =  7: 0.169   h = 10: 0.162
    h = 14: 0.154   h = 21: 0.161   h = 28: 0.165          -> h = 14
"""
import pandas as pd

import bt
import models

full, eve = bt.load()
folds = bt.folds()
tune = [d for k, v in folds.items() if not k.endswith("2025") for d in v]
bases = {"B1": bt.b1(full), "B1-evening": models.b1e(full, eve)}

rows = []
for name, F in bases.items():
    for half in (3, 5, 7, 10, 14, 21, 28):
        c = models.seasonal_corr(F, full, half=half, exclude=())
        s = bt.score(F * c, full, tune)
        rows.append({"base": name, "half": half, "wmape_tune": round(s["wmape"], 4),
                     "bias_tune": round(s["bias"], 3)})
print(pd.DataFrame(rows).to_string(index=False))
