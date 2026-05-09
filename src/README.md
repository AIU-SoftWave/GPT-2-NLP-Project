# `src/` — Source code

| File | Purpose |
|---|---|
| `dataset.py` | `SentimentDataset` class + `load_sst_data()`, `load_cfimdb_data()` |
| `model.py` | `GPT2Classifier` + `MODEL_REGISTRY` for adding new models |
| `train.py` | Training loop, called via `python -m src.train` |

## Import convention

Both the notebook and `train.py` import from `src.*`:
```python
from src.dataset import SentimentDataset, load_sst_data, load_cfimdb_data
from src.model import get_model, MODEL_REGISTRY
```

## Adding a new model

1. Create your model class in `model.py`
2. Register it in `MODEL_REGISTRY`
3. Run: `python -m src.train --model your_model_name --dataset sst`
