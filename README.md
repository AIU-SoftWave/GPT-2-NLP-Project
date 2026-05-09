# GPT-2 Sentiment Analysis — NLP Project

Two-stage sentiment analysis using frozen and fine-tuned GPT-2 on SST (5-class) and CFIMDB (binary) datasets.

## Project structure

```
├── src/               ← source code (dataset.py, model.py, train.py)
├── notebooks/         ← Jupyter notebooks (main.ipynb)
├── data/              ← datasets (gitignored)
├── checkpoints/       ← .pt model weights (gitignored)
├── results/           ← experiment JSON outputs (git-tracked)
├── docs/              ← project documentation
└── requirements.txt   ← Python dependencies
```

## Quick start

```bash
pip install -r requirements.txt
python -m src.train --dataset sst --frozen --model baseline --epochs 10
jupyter notebook notebooks/main.ipynb
```

## Adding a model

1. Add your model class to `src/model.py`
2. Register it in `MODEL_REGISTRY`
3. Train: `python -m src.train --model your_model --dataset sst`

## Results

All experiment results are saved to `results/*.json` and tracked in git.
