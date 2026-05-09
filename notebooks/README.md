# `notebooks/` — Jupyter notebooks

| File | Purpose |
|---|---|
| `main.ipynb` | Data exploration, training (calls `train.py`), results, plots |

## How to run

```bash
# From the project root:
jupyter notebook notebooks/main.ipynb
```

The notebook calls `train.py` via `!python -m src.train ...` subprocesses.
Results are saved to `results/*.json` and loaded back for plotting.
