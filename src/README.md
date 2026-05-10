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

## Training commands — `python -m src.train`

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | `sst` \| `cfimdb` | **(required)** | Dataset to train on |
| `--frozen` | flag | off | Freeze GPT-2 (train classifier head only) |
| `--epochs` | int | 10 | Number of epochs |
| `--batch-size` | int | 32 | Batch size per step |
| `--lr` | float | 3e-3 | Learning rate |
| `--workers` | int | 6 | DataLoader worker processes |
| `--accum` | int | 1 | Gradient accumulation steps |
| `--model` | str | `baseline` | Model name from `MODEL_REGISTRY` |
| `--name` | str | auto | Custom results filename (saved to `results/<name>.json`) |

### Examples

```bash
# SST-5 frozen baseline
python -m src.train --dataset sst --frozen --epochs 10 --batch-size 64 --lr 3e-3

# SST-5 finetune
python -m src.train --dataset sst --epochs 5 --batch-size 16 --lr 1e-5 --accum 1

# CFIMDB frozen baseline
python -m src.train --dataset cfimdb --frozen --epochs 10 --batch-size 32 --lr 3e-3

# CFIMDB finetune (gradient accumulation for smaller batch)
python -m src.train --dataset cfimdb --epochs 10 --batch-size 8 --lr 1e-5 --accum 2

# Custom model from registry
python -m src.train --model my_model --dataset sst --frozen --epochs 10

# Custom results filename
python -m src.train --dataset sst --frozen --name my_experiment
```

### Output

- Best checkpoint → `checkpoints/best_model_<name>.pt`
- Per-epoch history + test accuracy → `results/<name>.json`

## Adding a new model

1. Create your model class in `model.py`
2. Register it in `MODEL_REGISTRY`
3. Run: `python -m src.train --model your_model_name --dataset sst`
