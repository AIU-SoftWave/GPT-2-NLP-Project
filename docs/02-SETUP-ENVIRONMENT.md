# 02 — Setup Environment

## 1. Check Python Version

You need **Python 3.10 or later**.

```bash
python --version
```

If you don't have it, install Python 3.10+ from [python.org](https://python.org).

---

## 2. Create a Virtual Environment

Isolate your project dependencies:

```bash
# From the Project folder
python -m venv venv
```

Activate it:

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

---

## 3. Install Required Packages

Create a file `requirements.txt` with:

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
pytreebank
pandas
numpy
matplotlib
seaborn
scikit-learn
tqdm
ipython
jupyter
```

Then install:

```bash
pip install -r requirements.txt
```

### What each package is for:

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch — deep learning framework |
| `transformers` | HuggingFace — pretrained models (GPT-2) |
| `datasets` | HuggingFace — dataset loading utilities |
| `pytreebank` | Parse SST treebank format |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `matplotlib` / `seaborn` | Plotting & visualization |
| `scikit-learn` | Metrics (accuracy, confusion matrix) |
| `tqdm` | Progress bars for training loops |
| `jupyter` | Running the notebook |

---

## 4. Verify Installation

Run this in Python:

```python
import torch
import transformers
import pytreebank
import pandas as pd

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")

# Check GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Expected output:**
```
PyTorch version: 2.x.x
CUDA available: True   # (or False if CPU-only)
Transformers version: 4.x.x
GPU: NVIDIA GeForce ... (if available)
```

---

## 5. Verify Dataset Paths

```python
import os

# Check SST data
sst_tree = "./Datasets/SST2Data/trainDevTestTrees_PTB/trees/train.txt"
print(f"SST train trees exist: {os.path.exists(sst_tree)}")

# Check CFIMDB data
cfimdb = "./Datasets/CFIMDB/IMDB Dataset.csv"
print(f"CFIMDB exists: {os.path.exists(cfimdb)}")
```

**Expected output:**
```
SST train trees exist: True
CFIMDB exists: True
```

---

## 6. Launch Jupyter

```bash
jupyter notebook
```

Or for newer Jupyter:

```bash
jupyter lab
```

Open `main.ipynb` and you're ready to go.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `torch` installs CPU-only on Linux | Install CUDA version: `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| `pytreebank` fails to install | Try `pip install pytreebank==0.2.7` |
| Out of memory on GPU | Reduce batch size (start with 8 or 16) |
| File not found errors | Check you're running from the `Project/` directory with `pwd` |
