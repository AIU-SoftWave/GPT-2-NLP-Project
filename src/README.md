# `src/` — Source code

| File | Purpose |
|---|---|
| `dataset.py` | `SentimentDataset` class + `load_sst_data()`, `load_cfimdb_data()` |
| `model.py` | All model classes + `MODEL_REGISTRY` for adding new models |
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
| `--accum` | int | 1 | Gradient accumulation steps (effective batch = batch_size × accum) |
| `--model` | str | `baseline` | Model name from `MODEL_REGISTRY` |
| `--name` | str | auto | Custom experiment folder name (default: `<dataset>_frozen/finetune`) |
| `--sample` | int | — | Use N training samples for fast debugging |

### Available models

| Model name | Class | Pooling | Params added |
|---|---|---|---|
| `baseline` | `GPT2Classifier` | Last token | 0 |
| `mean_pool` | `GPT2MeanPoolClassifier` | Uniform average | 0 |
| `attention_pool` | `GPT2AttentionPoolClassifier` | Learned weighted average | 769 |

### Examples

```bash
# SST-5 frozen baseline
python -m src.train --dataset sst --frozen --epochs 10 --batch-size 32 --lr 3e-3

# SST-5 finetune
python -m src.train --dataset sst --epochs 10 --batch-size 32 --lr 1e-5

# CFIMDB frozen baseline
python -m src.train --dataset cfimdb --frozen --epochs 10 --batch-size 32 --lr 3e-3

# CFIMDB finetune (gradient accumulation for memory-constrained GPU)
python -m src.train --dataset cfimdb --epochs 10 --batch-size 8 --lr 1e-5 --accum 2

# Custom model from registry
python -m src.train --model attention_pool --dataset cfimdb --frozen --epochs 10

# Subset training data for quick debugging
python -m src.train --dataset sst --frozen --sample 100

# Custom experiment name
python -m src.train --dataset sst --frozen --name my_experiment
```

### Output

All results saved to `experiments/<name>/`:
- `checkpoint.pt` — Best model weights (highest dev accuracy)
- `results.json` — Full results: config, best_dev, test_acc, per-epoch history
- `config.json` — Experiment configuration (duplicated from results.json for quick access)

## Adding a new model

1. Create your model class in `model.py`
2. Add it to `MODEL_REGISTRY`
3. Run: `python -m src.train --model your_model_name --dataset sst`
