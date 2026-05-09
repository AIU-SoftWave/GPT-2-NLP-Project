# 02 — Setup Environment (Beginner-Friendly)

## What You'll Learn

- How to install all the software we need (Python, PyTorch, etc.)
- What each piece of software does (in plain English)
- How to check that everything is working
- How to launch the notebook

---

## Prerequisites

- **Python** installed (version 3.10 or later)
- A computer with internet access

---

## 1. Check Your Python Version

Python is the programming language we'll write our code in.

Open your terminal (command prompt) and type:

```bash
python --version
```

**What you should see:** `Python 3.10.x` or higher (like `Python 3.12.3`)

> If you don't have Python installed, go to [python.org](https://python.org) and download it. Get version 3.10 or later.

---

## 2. Create a Virtual Environment

### What is a virtual environment? — An analogy

Think of a virtual environment as a **separate kitchen** for this project.

- You wouldn't cook a French recipe in a kitchen full of Indian spices and sauces (they'd get in the way)
- Similarly, each Python project needs its own "kitchen" with just the ingredients (packages) it needs
- A virtual environment keeps this project's packages separate from other projects on your computer

### Create it:

```bash
# From the Project folder
python -m venv venv
```

This creates a folder called `venv` in your project directory. That folder will hold all the packages we install.

### Activate it (turn on this kitchen):

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

After activation, your terminal will show `(venv)` at the start of the line. That means you're now inside your project's "kitchen."

---

## 3. Install Required Packages

### What are packages?

Packages are **pre-written code** that other people created so we don't have to reinvent the wheel. Think of them like **appliances in your kitchen** — you don't build a blender from scratch, you just use one.

Create a file called `requirements.txt` in your project folder with this content:

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

Then run:

```bash
pip install -r requirements.txt
```

This will download and install all the packages. It might take a few minutes.

### What each package does (plain English):

| Package | What it is (analogy) | What it does for us |
|---------|---------------------|-------------------|
| `torch` | **The engine** | PyTorch — runs all the math for deep learning. Like a car engine that powers everything else |
| `transformers` | **The chef** | HuggingFace library — gives us GPT-2 ready to use, like a pre-made cake mix instead of baking from scratch |
| `pytreebank` | **The translator** | Reads the SST dataset's special format (trees) and turns it into a simple table |
| `pandas` | **The spreadsheet** | Lets us work with data in table form (like Excel) |
| `numpy` | **The calculator** | Does fast math with numbers, especially arrays and matrices |
| `matplotlib` / `seaborn` | **The artist** | Creates graphs and charts so we can visualize our results |
| `scikit-learn` | **The referee** | Gives us accuracy scores, confusion matrices — ways to measure how good our model is |
| `tqdm` | **The progress bar** | Shows a progress bar during training so you know how long it'll take |
| `jupyter` | **The workspace** | Lets us run the notebook (main.ipynb) where we'll write and run our code |

---

## 4. Verify Installation

Let's make sure everything installed correctly. Run this Python code:

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

### What does CUDA mean?

**CUDA** is NVIDIA's technology that lets us use the **GPU** (graphics card) for training. 

Think of it this way:
- **CPU** (your computer's brain) = A few very smart chefs who can cook anything, but slowly
- **GPU** (graphics card) = 1000 junior chefs who can only do simple tasks, but do them all at once, very fast

Training AI models requires doing millions of simple math operations. GPUs are PERFECT for this — they can do thousands of calculations at the same time.

**Expected output:**
```
PyTorch version: 2.x.x
CUDA available: True   # Good! You have a GPU!
(or False if CPU-only — that's okay, training will just be slower)

Transformers version: 4.x.x
GPU: NVIDIA GeForce ... (if available)
```

---

## 5. Verify Dataset Paths

Check that the data files actually exist where we expect them:

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

If either shows `False`, you're in the wrong folder or the datasets aren't downloaded.

---

## 6. Launch Jupyter

Jupyter Notebook is where we'll run our code. Think of it as a **digital lab notebook** where you can mix explanations (text) with experiments (code) in the same document.

```bash
jupyter notebook
```

This will open a web browser with a file browser. Click on `main.ipynb` and you're ready!

> **Tip:** If your browser doesn't open automatically, look at the terminal output for a URL that starts with `http://localhost:8888/` and copy-paste it into your browser.

---

## Troubleshooting

| Problem | What's happening | Solution |
|---------|-----------------|----------|
| `torch` installs CPU-only on Linux | PyTorch didn't detect your GPU | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| `pytreebank` fails to install | Version conflict | Try `pip install pytreebank==0.2.7` |
| Out of memory on GPU | Your GPU can't fit the model + data | Reduce batch size (start with 8 instead of 16) |
| File not found errors | You're running from the wrong folder | Run `pwd` (Linux/Mac) or `cd` (Windows) to check your location, then navigate to the Project folder |

---

## Quick Summary

```
1. Check Python version          → python --version
2. Create virtual environment    → python -m venv venv
3. Activate it                   → source venv/bin/activate (Linux/Mac)
                                  or venv\Scripts\activate (Windows)
4. Install packages              → pip install -r requirements.txt
5. Launch Jupyter                → jupyter notebook
6. Open main.ipynb               → Click the file in the browser
```
