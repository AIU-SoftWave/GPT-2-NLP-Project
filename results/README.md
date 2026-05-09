# `results/` — Experiment outputs

Each experiment run by `train.py` saves a JSON file here:

```json
{
  "name": "sst_frozen",
  "best_dev": 0.4512,
  "test_acc": 0.4421,
  "history": {
    "train_loss": [...],
    "train_acc": [...],
    "dev_loss": [...],
    "dev_acc": [...]
  }
}
```

These are tracked in git so results are never lost between runs.
The notebook loads them automatically for plotting.
