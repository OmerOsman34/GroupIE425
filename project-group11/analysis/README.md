# Analysis (Group 11)

Validation code behind `checkpoint.pdf` and `main.py`. None of it runs in the evaluation; only
`main.py` does. No data is committed: every script reads the uncut tables that the self-check
kit writes to its `delivered/` folder after `python selfcheck.py --rebuild-data`.

```
set KIT_DELIVERED=C:\path\to\project-self-check-kit-main\delivered     (Windows cmd)
export KIT_DELIVERED=/path/to/project-self-check-kit-main/delivered    (macOS / Linux)
cd analysis
python results_ckpt.py     # Table 1 of the checkpoint, and the brief's yardstick check
python tune_h.py           # the search that fixed HALF = 14 (2022-2024 folds only)
python fig_ckpt.py         # Figure 1 -> ../checkpoint/fig1.pdf
```

| file | what it does |
|---|---|
| `bt.py` | loads the tables, defines the folds (9-20 Nov and 23 Nov-1 Dec of 2022-2025), WMAPE / bias, B1 and last week's value |
| `models.py` | B1-evening (adds the before-21:00 count of d-1) and the seasonal factor learned from earlier years |
| `tune_h.py` | chooses the window half-width h on the tune folds |
| `results_ckpt.py` | scores every model on every fold; reproduces the brief's yardstick (B1 0.204, last week 0.252) |
| `fig_ckpt.py` | the checkpoint figure |

**Leakage rule used everywhere:** a forecast for day d uses complete days up to d-2 and the
before-21:00 count of d-1 only (every series is shifted accordingly), and the seasonal factor for
a day in year Y uses years before Y only.

**Check of the submission:** `main.py`, run through the self-check kit at the origins
2025-11-12, 2025-11-25 and 2024-11-28, gives the same forecasts as `models.py` to four decimals.
