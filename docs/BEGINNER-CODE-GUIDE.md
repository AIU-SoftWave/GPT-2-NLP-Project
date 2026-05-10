# Beginner's Guide: NLP Concepts & Every Line of Code Explained

**Who this is for:** You've never done NLP before. You know basic Python (functions, classes, loops). You want to understand *everything* in this project — every concept, every file, every line — so you can modify it with confidence.

**How to use this guide:** Read Part 1 first (concepts). Then open each source file alongside Part 2 and read along. Each code snippet tells you the file and line numbers so you can find it.

---

## Part 1: The Absolute Minimum NLP Concepts

These are the concepts you need before any line of code makes sense. Each one is explained using *this project's code* as the example.

### 1.1 Token IDs vs. Tokens vs. Words

A **word** is what humans read ("great"). A **token** is how the computer splits text. GPT-2 doesn't see words — it sees **token IDs** (integers).

```
"The movie was great!"
        ↓  [GPT-2 Tokenizer]
[464, 3181, 373, 3734,  0]     ← token IDs
  The  movie  was  great  !     ← decoded back to text
```

GPT-2's vocabulary has 50,257 possible token IDs. Each ID maps to a piece of text (a word, part of a word, or punctuation).

**Where in the code:** `src/dataset.py:25-31` — the tokenizer converts text → token IDs in one call:
```python
encodings = tokenizer(
    df['text'].tolist(),
    truncation=True,
    padding='max_length',
    max_length=max_length,
    return_tensors='pt',
)
```

### 1.2 Embedding Dimension (768)

Every token ID is immediately converted to a vector of 768 numbers. This is the **embedding dimension** (also called `hidden_size`).

Why 768? It's a design choice from the GPT-2 authors — large enough to encode rich meaning, small enough to compute efficiently.

**An embedding is a lookup table:** row 0 = vector for token ID 0, row 1 = vector for token ID 1, etc. The model *learns* these vectors during pre-training.

**Where in the code:** `src/model.py:29`:
```python
hidden_size = self.gpt2.config.hidden_size  # 768
```

Every hidden state in this project is a vector of 768 numbers. This number shows up everywhere — the classifier input size, the attention layer size, etc.

### 1.3 Hidden States (The Core Data Structure)

When you send a batch of token IDs through GPT-2, you get back **hidden states** — a 3D tensor of shape:

```
(batch_size, sequence_length, hidden_size)
  e.g. (8, 512, 768)
```

Think of it as a spreadsheet:
- **Dimension 0 (batch):** Which movie review in the batch
- **Dimension 1 (sequence):** Which token position in that review
- **Dimension 2 (hidden):** Which of the 768 learned features

So `outputs.last_hidden_state[2, 150, 300]` means "the 300th feature of the 150th token in the 3rd review."

**Where in the code:** `src/model.py:41`:
```python
last_hidden = outputs.last_hidden_state  # shape: (batch, seq_len, 768)
```

### 1.4 Causal (Left-to-Right) Attention

GPT-2 is a **causal language model**. This means when it processes token #5, it can only "see" tokens #1 through #5. It cannot peek at token #6.

Why does this matter for sentiment analysis? The **last token** has seen everything before it. So its hidden state is the richest — it contains information about the entire input. This is why last-token pooling works at all.

```
"The movie was absolutely fantastic"
  ↑     ↑     ↑      ↑          ↑
  #1    #2    #3     #4         #5
                                 └── can see #1-#5 (full context)
  #1 can only see itself (no context)
```

**Where in the code:** Not explicit — it's baked into GPT-2's architecture. But line `src/model.py:42` depends on it:
```python
last_token_idx = attention_mask.sum(dim=1) - 1  # pick last real token
```

If GPT-2 weren't causal, the last token wouldn't be special — every token would see everything.

### 1.5 The Logits Vector

The final layer of our classifier produces **logits**: raw scores for each class.

```python
logits = self.classifier(sentence_repr)  # shape: (batch, num_classes)
# e.g. (8, 5) for SST with 5 classes
```

For one review, logits might look like:
```
[2.1, 1.3, -0.5, 0.8, -1.2]
  ↑     ↑     ↑    ↑     ↑
class0 class1 class2 class3 class4
```

The largest number = the model's prediction. `torch.argmax(logits)` = `0` (predicts "very negative").

**Where in the code:**
- `src/model.py:46`: produces logits
- `src/train.py:73`: makes predictions `torch.argmax(logits, dim=1)`

### 1.6 Cross-Entropy Loss (in 2 Sentences)

Cross-entropy loss compares the model's predicted probabilities against the true label. If the true label is class 3, it wants the logit for class 3 to be high and all others low.

- **Loss = 0:** Perfect prediction (logit for correct class = +∞, others = -∞)
- **Loss = high:** Wrong prediction or low confidence
- **Loss decreases** during training as the model learns

**Where in the code:** `src/train.py:143`:
```python
crit = nn.CrossEntropyLoss()
```
And line 67:
```python
loss = crit(logits, labs)  # compare predictions to true labels
```

### 1.7 Train / Dev / Test Split

We never evaluate on the training data (the model has seen it). We hold out:

- **Dev set:** Used during development to check progress (evaluate after every epoch)
- **Test set:** Only used at the very end, to report final numbers

If you tune hyperparameters on the test set, you're cheating — the test numbers lose meaning.

**CFIMDB sizes:** 1,701 train / 245 dev / 488 test

**Where in the code:** `src/dataset.py:79-94`: loads separate files for each split.

### 1.8 Overfitting (The Most Important Concept)

**Overfitting** = the model memorizes the training data but fails on new data.

**Signs of overfitting:**
- Training accuracy keeps rising (e.g., 0.99)
- Dev accuracy plateaus then drops (e.g., peaks at 0.955 then falls to 0.943)

**Why it happens in this project:** GPT-2 has 124 million parameters. CFIMDB has only 1,701 training examples. The model is powerful enough to memorize every single review. The solution is either (a) freeze the encoder, or (b) use early stopping.

**Real example from our experiments:**

| Epoch | Train Acc | Dev Acc |
|-------|-----------|---------|
| 1 | 0.559 | 0.706 |
| 3 | 0.947 | 0.943 |
| **7** | **0.990** | **0.955** ← best dev |
| 10 | 0.995 | 0.943 |

After epoch 7, dev accuracy *drops* while training accuracy keeps rising. Classic overfitting.

**Where in the code:** `src/model.py:32-34` (freezing prevents overfitting by limiting trainable params).

---

## Part 2: Every Line of Code Explained

### 2.1 `src/__init__.py`

```python
# (empty)
```

**Why it exists:** Python needs `__init__.py` to treat `src/` as a package. Without it, `from src.dataset import ...` wouldn't work. The file is empty because we don't need any package-level setup.

**NLP concept:** None. Pure Python mechanics.

---

### 2.2 `src/dataset.py` — Data Loading (96 lines)

This file loads text data from disk and converts it to PyTorch tensors.

```python
# Lines 1-6: Docstring
"""
Data loading and pre-tokenized Dataset for multiprocessing-safe DataLoader workers.

All datasets are tokenized once at construction time (not per-batch),
so DataLoader workers never need access to the tokenizer object.
This avoids pickling issues with HuggingFace tokenizers.
"""
```

**Why pre-tokenize at construction time?** If we tokenized inside `__getitem__`, every DataLoader worker would need a copy of the tokenizer. HuggingFace tokenizers don't serialize (pickle) well. By tokenizing *once* when the Dataset is created, workers only access pre-computed tensors — no tokenizer needed.

```python
# Lines 9-12: Imports
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
```

`pandas` holds data in tables (DataFrames). `torch` provides tensor operations. `Dataset` is the base class for all PyTorch datasets.

```python
# Lines 15-43: SentimentDataset class
class SentimentDataset(Dataset):
    """PyTorch Dataset that pre-tokenizes all text at init."""
    def __init__(self, df, tokenizer, max_length=128):
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)
```

`torch.tensor(..., dtype=torch.long)` converts the label column (integers like 0, 1, 2, 3, 4) from a pandas Series to a PyTorch tensor. `torch.long` = 64-bit integer (required by `CrossEntropyLoss`).

```python
        encodings = tokenizer(
            df['text'].tolist(),      # list of strings, e.g. ["great movie", "terrible film"]
            truncation=True,           # cut off text longer than max_length
            padding='max_length',      # pad shorter texts to max_length with 0s
            max_length=max_length,     # SST=128, CFIMDB=512
            return_tensors='pt',       # return PyTorch tensors, not Python lists
        )
```

**This single call does ALL the work:**
1. Takes a list of N strings
2. Converts each to token IDs using GPT-2's vocabulary
3. Truncates any text longer than `max_length`
4. Pads any text shorter than `max_length` with 0s (padding token)
5. Returns two tensors: `input_ids` (shape N × max_length) and `attention_mask` (same shape)

```python
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
```

**`attention_mask`** is critical: 1 = real token, 0 = padding token. Without it, the model would try to classify padding tokens as if they were real words.

```python
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'label': self.labels[idx],
        }
```

PyTorch `DataLoader` calls `__getitem__` to get one example at a time. It collates them into batches. Returning a dictionary with consistent keys makes the training loop readable.

**NLP concept — Padding & Truncation:** In any batch, all sequences must be the same length. Padding fills short sequences with 0s; truncation cuts long sequences. `attention_mask` tells the model which tokens to ignore.

---

#### `load_sst_data()` (lines 46-64)

```python
def load_sst_data():
    import pytreebank
    data = []
    tree_path = './data/Datasets/SST2Data/trainDevTestTrees_PTB/trees/'
```

SST (Stanford Sentiment Treebank) stores data as **Penn Treebank trees** — nested structures like `(4 (2 The) (3 movie))` where numbers are sentiment scores (0-4). `pytreebank` parses these trees.

```python
    for split in ['train', 'dev', 'test']:
        full_path = os.path.join(tree_path, f'{split}.txt')
        if not os.path.exists(full_path):
            print(f'Warning: {full_path} not found. Skipping...')
            continue
        trees = pytreebank.import_tree_corpus(full_path)
        for tree in trees:
            data.append({'text': tree.to_lines()[0], 'label': tree.label, 'split': split})
    return pd.DataFrame(data)
```

`tree.to_lines()[0]` flattens the tree to plain text (e.g., `"The movie was great"`). `tree.label` extracts the root sentiment label (0-4).

**NLP concept — Treebank Format:** In linguistic treebanks, sentences are stored as parse trees showing grammatical structure. SST adds sentiment labels to each node. We only use the root label (the whole sentence's sentiment).

---

#### `load_cfimdb_data()` (lines 67-96)

```python
def load_cfimdb_data():
    base = './data/Datasets/CFIMDB_CS224N/'
    parts = []

    for split, fname in [('train', 'ids-cfimdb-train.csv'),
                          ('dev', 'ids-cfimdb-dev.csv')]:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            print(f'Warning: {path} not found. Skipping...')
            continue
        df = pd.read_csv(path, sep='\t')          # tab-separated
        df = df.rename(columns={'sentence': 'text', 'sentiment': 'label'})
        df['split'] = split
        parts.append(df[['text', 'label', 'split']])
```

Reads CSV files. Renames columns to match the standard format (`text`, `label`, `split`).

```python
    path = os.path.join(base, 'ids-cfimdb-test-student.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t', header=None, skiprows=1, usecols=[3], names=['text'])
        df['label'] = -1        # -1 means "unknown" — test labels are hidden
        df['split'] = 'test'
        parts.append(df[['text', 'label', 'split']])
    return pd.concat(parts, ignore_index=True)
```

Test set has no labels (they're held out by the course staff). `label = -1` signals "don't evaluate on test." The `train.py` checks for this (`src/train.py:164`).

**NLP concept — Held-out Test Set:** In real evaluations, test labels are kept secret to prevent overfitting to the test set. You submit predictions and get scores back. In this project, if test labels are -1, the code uses dev accuracy as the final metric instead.

---

### 2.3 `src/model.py` — The Models (207 lines)

This is the heart of the project. Let's go through each model.

#### `GPT2Classifier` — Baseline (lines 16-46)

```python
class GPT2Classifier(nn.Module):
    """Baseline: last-token pooling."""
    def __init__(self, num_classes=5, dropout=0.1, freeze=True):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
```

`GPT2Model.from_pretrained('gpt2')` downloads GPT-2 small (124M params) from HuggingFace and loads its pre-trained weights. This is the **transfer learning** step — we start with a model that already "knows" English.

```python
        hidden_size = self.gpt2.config.hidden_size  # = 768
        self.dropout = nn.Dropout(dropout)           # randomly zeroes 10% of neurons
        self.classifier = nn.Linear(hidden_size, num_classes)  # 768 → 5
```

`nn.Linear(768, 5)` is a simple matrix multiplication: input 768 numbers, output 5 numbers (one per class). This is the **classification head** — it learns to map GPT-2's features to sentiment scores.

```python
        if freeze:
            for param in self.gpt2.parameters():
                param.requires_grad = False
```

**Freezing** sets `requires_grad = False` on all GPT-2 parameters. PyTorch's optimizer skips parameters where `requires_grad = False`. So only the classifier head (3,845 params) gets updated.

```python
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
```

GPT-2 processes all tokens in parallel (thanks to the transformer architecture) and returns hidden states for every token position. `last_hidden_state` is the output of the final layer.

```python
        # Find the last REAL token (not padding)
        last_token_idx = attention_mask.sum(dim=1) - 1
```

`attention_mask.sum(dim=1)` counts how many real tokens are in each sequence. Subtract 1 to get the index of the last real token. For a sequence of 128 tokens where 100 are real and 28 are padding, `sum = 100`, index = 99 (0-indexed).

```python
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        sentence_repr = last_hidden[batch_indices, last_token_idx]
```

This uses **fancy indexing** to grab one token per sequence: for batch item 0, take token `last_token_idx[0]`; for batch item 1, take token `last_token_idx[1]`, etc.

```python
        sentence_repr = self.dropout(sentence_repr)
        return self.classifier(sentence_repr)
```

Dropout prevents overfitting. The linear layer produces logits of shape `(batch, num_classes)`.

**NLP concept — Last-Token Pooling:** In a causal LM, the last token's hidden state is a summary of the entire input because it attended to everything before it.

---

#### `GPT2MeanPoolClassifier` — Mean Pooling Baseline (lines 49-78)

```python
class GPT2MeanPoolClassifier(nn.Module):
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
```

`attention_mask` has shape `(batch, seq_len)`. We need shape `(batch, seq_len, 1)` to multiply with `last_hidden` which is `(batch, seq_len, 768)`. `unsqueeze(-1)` adds a dimension at the end. `.float()` converts from int to float for multiplication.

```python
        masked_hidden = last_hidden * mask
```

Padding tokens (mask=0) get zeroed out. Real tokens (mask=1) keep their values.

```python
        sentence_repr = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
```

Sum across all tokens, divide by the number of real tokens. `.clamp(min=1e-9)` prevents division by zero for empty sequences. Result: average of all non-padding token hidden states.

**NLP concept — Mean Pooling:** Every token gets equal weight. Useful when sentiment is distributed evenly across all words. In practice, it dilutes strong sentiment signals.

---

#### `AttentionPooling` — Softmax Attention Module (lines 81-101)

```python
class AttentionPooling(nn.Module):
    """Learned attention pooling module."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)  # 768 → 1
```

A single linear layer that learns to score each token's relevance to sentiment. 769 parameters total (768 weights + 1 bias).

```python
    def forward(self, hidden_states, attention_mask):
        scores = self.attn(hidden_states).squeeze(-1)
```

`self.attn(hidden_states)` gives shape `(batch, seq_len, 1)`. `.squeeze(-1)` removes the last dimension → `(batch, seq_len)`. Each token now has a single relevance score.

```python
        scores = scores.masked_fill(attention_mask == 0, -1e9)
```

Padding tokens get a very negative score (-1e9), so after softmax they get ~0 weight.

```python
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)
```

Softmax converts scores to probabilities that sum to 1 across all tokens. Then: weighted sum of hidden states.

**NLP concept — Softmax Attention:** Forces competition between tokens. If one token gets high weight, others must get lower weight (because probabilities sum to 1). This is semantically bad for long reviews with multiple sentiment phrases.

---

#### `GatedAttentionPooling` — Our Innovation (lines 135-155)

```python
class GatedAttentionPooling(nn.Module):
    """Gated Attention Pooling module (innovation)."""
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1)  # same 769 params as attention
```

Identical parameter count to softmax attention. Fair comparison.

```python
    def forward(self, hidden_states, attention_mask):
        gates = torch.sigmoid(self.gate(hidden_states)).squeeze(-1)
```

**`torch.sigmoid`** is the key difference. Sigmoid outputs values between 0 and 1 *independently* for each token. Unlike softmax, sigmoid gates don't compete — multiple tokens can all have high weights.

```
Softmax: [0.1, 0.7, 0.2] → sums to 1 (competitive)
Sigmoid: [0.8, 0.9, 0.7] → each is 0-1 (independent)
```

```python
        gates = gates * attention_mask.float()
        total = gates.sum(dim=1, keepdim=True).clamp(min=1e-9)
        weights = (gates / total).unsqueeze(-1)
```

**L1 normalization:** Divide each gate by the sum of all gates. This preserves the independent nature (no token gets penalized) while keeping the scale stable for training.

```python
        return (hidden_states * weights).sum(dim=1)
```

Same weighted sum as softmax attention, but the weights come from normalized sigmoid gates instead of softmax.

**NLP concept — Gating:** Independent control signals. Each token decides its own relevance without being forced to compete with others. For sentiment: "terrible acting" AND "waste of time" AND "poor direction" can ALL contribute fully.

---

#### Full Classifier Wrappers (lines 104-187)

`GPT2AttentionPoolClassifier` and `GPT2GatedAttentionPoolClassifier` follow the same pattern:

```python
class GPT2GatedAttentionPoolClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.1, freeze=True):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        hidden_size = self.gpt2.config.hidden_size
        self.pool = GatedAttentionPooling(hidden_size)  # ← THE INNOVATION
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        # ... freeze logic ...

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        sentence_repr = self.pool(hidden, attention_mask)  # pooling instead of last-token
        sentence_repr = self.dropout(sentence_repr)
        return self.classifier(sentence_repr)
```

The only difference from `GPT2Classifier` is `self.pool(hidden, attention_mask)` replaces the manual last-token indexing. Everything else is identical — this is the power of modular design.

---

#### Model Registry (lines 190-207)

```python
MODEL_REGISTRY = {
    'baseline': GPT2Classifier,
    'mean_pool': GPT2MeanPoolClassifier,
    'attention_pool': GPT2AttentionPoolClassifier,
    'gated_attention_pool': GPT2GatedAttentionPoolClassifier,
}

def get_model(model_name, num_classes, freeze):
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f'Unknown model "{model_name}". Available: {available}')
    cls = MODEL_REGISTRY[model_name]
    return cls(num_classes=num_classes, freeze=freeze)
```

**Design pattern — Registry:** Maps string names to model classes. Adding a new model means: (1) write the class, (2) add it to the dict. No other code changes needed. The `--model` flag in `train.py` uses this.

---

### 2.4 `src/train.py` — The Training Pipeline (189 lines)

This is the orchestrator. It loads data, creates the model, trains it, and saves results.

```python
# Lines 1-9: Module docstring
"""GPT-2 Sentiment Analysis — single-experiment runner.
Usage:
    python -m src.train --dataset sst --frozen --epochs 10
    python -m src.train --dataset cfimdb --epochs 10 --batch-size 8 --lr 1e-5 --accum 2
"""
```

The `-m` flag means "run as a module." Python finds `src/train.py` and executes it.

```python
# Lines 11-26: Imports + settings
import json, os, argparse, datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
```

Key imports and what they do:
- `json`: Save/load results files
- `argparse`: Parse command-line arguments
- `AdamW`: The optimizer (like Adam but with decoupled weight decay)
- `GPT2Tokenizer`: Converts text ↔ token IDs
- `DataLoader`: Batches data for training
- `accuracy_score`: Simple metric (correct / total)
- `tqdm`: Progress bars

```python
from src.dataset import SentimentDataset, load_sst_data, load_cfimdb_data
from src.model import get_model
```

Imports from our own code. This is why `src/` needs `__init__.py`.

```python
os.makedirs('experiments', exist_ok=True)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
```

- `os.makedirs`: Ensures `experiments/` exists
- `torch.set_float32_matmul_precision('high')`: Enables TensorFloat-32 on supported GPUs (~2x faster matrix multiplies with no precision loss for deep learning)
- `cudnn.benchmark`: Auto-tunes CUDA kernels for your GPU

#### `make_loaders()` — Data Preparation (lines 30-47)

```python
def make_loaders(df, tokenizer, bs, max_len, workers=6):
    loaders = {}
    for split in ['train', 'dev', 'test']:
        sd = df[df['split'] == split].reset_index(drop=True)
        ds = SentimentDataset(sd, tokenizer, max_len)
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=(split == 'train'),
            num_workers=workers, pin_memory=True,
            persistent_workers=True if workers > 0 else False,
            prefetch_factor=2 if workers > 0 else None,
        )
    return loaders
```

Creates three DataLoaders. Only the training set is shuffled (important — shuffling dev/test would change the evaluation every time).

`pin_memory=True` speeds up GPU transfer. `num_workers>0` loads data in parallel. `persistent_workers` keeps workers alive between epochs (avoids restart overhead).

**NLP concept — Batch Processing:** You never process one example at a time (too slow). You process N examples together as a batch. The batch size is a key hyperparameter — larger = more stable gradients but more memory.

#### `train_one_epoch()` (lines 50-76)

```python
def train_one_epoch(model, loader, opt, crit, device, accum=1):
    model.train()
    total_loss, preds, labels = 0, [], []
    opt.zero_grad()
```

`model.train()` tells PyTorch to enable dropout and batch norm (they behave differently during training vs evaluation). `opt.zero_grad()` clears gradients from the previous step.

```python
    pbar = tqdm(loader, desc='Train')
    for i, b in enumerate(pbar):
        ids = b['input_ids'].to(device, non_blocking=True)
        mask = b['attention_mask'].to(device, non_blocking=True)
        labs = b['label'].to(device, non_blocking=True)
```

Moves batch tensors to GPU. `non_blocking=True` overlaps data transfer with computation.

```python
        logits = model(ids, mask)
        loss = crit(logits, labs) / accum
        loss.backward()
```

Forward pass (compute predictions) → calculate loss → backpropagate (compute gradients).

**`loss / accum`** is gradient accumulation (see below).

```python
        if (i + 1) % accum == 0:
            opt.step()
            opt.zero_grad()
```

**Gradient accumulation:** When GPU memory is limited (e.g., batch size 8 instead of 32), you run multiple forward/backward passes, accumulating gradients, then take one optimizer step. Effective batch size = batch_size × accum.

```python
        total_loss += loss.item() * accum
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels.extend(labs.cpu().numpy())
```

Tracks running loss and predictions for accuracy calculation.

```python
    return total_loss / len(loader), accuracy_score(labels, preds)
```

Returns average loss and accuracy for this epoch.

#### `evaluate()` — Evaluation (lines 79-97)

```python
@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    total_loss, preds, labels = 0, [], []
    for b in tqdm(loader, desc='Eval'):
        ids = b['input_ids'].to(device, non_blocking=True)
        mask = b['attention_mask'].to(device, non_blocking=True)
        labs = b['label'].to(device, non_blocking=True)
        logits = model(ids, mask)
        total_loss += crit(logits, labs).item()
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels.extend(labs.cpu().numpy())
    return total_loss / len(loader), accuracy_score(labels, preds)
```

`@torch.no_grad()` disables gradient computation — this saves memory and speeds up inference because we don't need gradients for evaluation. `model.eval()` disables dropout (we always want the full model during evaluation, not a randomly dropped-out version).

#### `main()` — The Full Pipeline (lines 100-185)

```python
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['sst', 'cfimdb'], required=True)
    p.add_argument('--frozen', action='store_true')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--model', type=str, default='baseline')
    p.add_argument('--name', type=str, default=None)
    p.add_argument('--sample', type=int, default=None)
    args = p.parse_args()
```

All command-line arguments. The defaults are set for frozen mode (higher LR, larger batch). Fine-tuning overrides with `--batch-size 8 --lr 1e-5 --accum 2`.

```python
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
```

Sets up GPU (or CPU fallback) and tokenizer. **Crucial:** GPT-2 doesn't have a pad token by default. We set `pad_token = eos_token` (end-of-sequence token, ID 50256) so padding doesn't confuse the model.

```python
    if args.dataset == 'sst':
        df = load_sst_data(); num_classes = 5; max_len = 128
    elif args.dataset == 'cfimdb':
        df = load_cfimdb_data(); num_classes = 2; max_len = 512
```

SST has 5 classes (0-4) and max 128 tokens. CFIMDB has 2 classes (0/1) and max 512 tokens (reviews are longer).

```python
    if args.sample and args.sample < len(df[df['split'] == 'train']):
        train_idx = df[df['split'] == 'train'].sample(args.sample, random_state=42).index
        other_idx = df[df['split'] != 'train'].index
        df = df.loc[train_idx.union(other_idx)]
```

`--sample` is a debugging feature: train on only N examples to verify the code runs fast before the full training.

```python
    loaders = make_loaders(df, tokenizer, args.batch_size, max_len, args.workers)
    model = get_model(args.model, num_classes, freeze=args.frozen).to(device)
```

Creates data loaders and model. `.to(device)` moves the model to GPU.

```python
    mode = 'frozen' if args.frozen else 'finetune'
    name = args.name or f'{args.dataset}_{mode}'
    save_dir = f'experiments/{name}'
    os.makedirs(save_dir, exist_ok=True)
    crit = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
```

Sets up experiment directory, loss function, and optimizer. `AdamW` with `weight_decay=0.0` means no L2 regularization (common for fine-tuning).

```python
    best_dev, history = 0, {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}
    for epoch in range(args.epochs):
        print(f'\n--- Epoch {epoch+1}/{args.epochs} ---')
        tl, ta = train_one_epoch(model, loaders['train'], opt, crit, device, args.accum)
        dl, da = evaluate(model, loaders['dev'], crit, device)
```

The main training loop: for each epoch, train → evaluate on dev → record.

```python
        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['dev_loss'].append(dl)
        history['dev_acc'].append(da)
        if da > best_dev:
            best_dev = da
            torch.save(model.state_dict(), f'experiments/{name}/checkpoint.pt')
```

Saves the model whenever dev accuracy improves. At the end, `checkpoint.pt` contains the best model (highest dev accuracy), not necessarily the last one.

**NLP concept — Early Stopping via Checkpointing:** The model might overfit in later epochs. By saving the best checkpoint (not the last), we get the best possible performance.

```python
    model.load_state_dict(torch.load(f'experiments/{name}/checkpoint.pt'))
    test_labels = loaders['test'].dataset.labels
    if (test_labels == -1).any():
        dl_final, ta_final = evaluate(model, loaders['dev'], crit, device)
        print(f'\nDev Loss: {dl_final:.4f} | Dev Acc: {ta_final:.4f} (test labels unavailable)')
    else:
        tl_final, ta_final = evaluate(model, loaders['test'], crit, device)
        print(f'\nTest Loss: {tl_final:.4f} | Test Acc: {ta_final:.4f}')
```

If test labels are -1 (CFIMDB test set), use dev accuracy as the final metric. Otherwise report test accuracy.

```python
    config = vars(args)
    config.pop('sample', None)
    results = {
        'name': name,
        'timestamp': datetime.datetime.now().isoformat(),
        'config': config,
        'best_dev': best_dev,
        'test_acc': ta_final,
        'history': history,
    }
    with open(f'{save_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
```

Saves everything to `results.json` — the source of truth for all experiment numbers.

```python
if __name__ == '__main__':
    main()
```

Standard Python entry point. Only runs when the file is executed directly (not imported).

---

### 2.5 `notebooks/length_analysis.py` — Analysis Script

This script generates Figure 5 in the report (length distribution + gated attention gain).

```python
# Lines 1-9: Setup
import os, sys
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import load_sst_data, load_cfimdb_data
```

`matplotlib.use('Agg')` is important — without it, matplotlib tries to open a window, which fails on servers without displays. `sys.path.insert(0, ...)` adds the project root to Python's import path so `from src.dataset import ...` works.

```python
# Lines 11-16: Tokenize and count
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def compute_lengths(texts, max_length=512):
    lengths = []
    for t in texts:
        tokens = tokenizer.encode(t, truncation=True, max_length=max_length)
        lengths.append(len(tokens))  # number of real tokens (no padding)
    return np.array(lengths)
```

`tokenizer.encode()` returns a list of token IDs. `len(tokens)` is the actual length of the text in tokens (without padding). This tells us how long each review actually is.

```python
# Lines 18-36: Compute lengths for both datasets
datasets = {}
for dataset_name in ['CFIMDB', 'SST']:
    df = load_cfimdb_data() if dataset_name == 'CFIMDB' else load_sst_data()
    max_len = 512 if dataset_name == 'CFIMDB' else 128
    lens = {}
    for split in ['train', 'dev', 'test']:
        subset = df[df['split'] == split]
        lens[split] = compute_lengths(subset['text'].tolist(), max_length=max_len)
    datasets[dataset_name] = lens
```

**What this reveals:**
- CFIMDB reviews: mean 214 tokens, median 199 (long documents)
- SST sentences: mean 22 tokens, median 21 (short phrases)

This difference explains why attention pooling helps CFIMDB but not SST — short sentences fit entirely in the last token's receptive field.

```python
# Lines 38-47: Hardcoded results for the bar chart
results = {
    'CFIMDB': {
        'frozen': {'Last-token': 0.886, 'Gated Attn': 0.951, 'Gain': 0.0733},
        'finetune': {'Last-token': 0.971, 'Gated Attn': 0.955, 'Gain': -0.016},
    },
    'SST': {
        'frozen': {'Last-token': 0.481, 'Gated Attn': 0.459, 'Gain': -0.022},
        'finetune': {'Last-token': 0.513, 'Gated Attn': 0.515, 'Gain': 0.002},
    },
}
```

These numbers come from `experiments/*/results.json`. They're hardcoded here because reading all JSON files in the plotting script would be overkill for a simple bar chart.

```python
# Lines 49-81: Create the 2×2 figure
fig, axes = plt.subplots(2, 2, figsize=(10, 8),
                         gridspec_kw={'width_ratios': [3, 1]})
```

**Subplot layout:**
- Top row: CFIMDB analysis
  - Left: Histogram of token lengths (3: width ratio)
  - Right: Bar chart of gated attention gain (1: width ratio)
- Bottom row: Same for SST

```python
for row, dataset_name in enumerate(['CFIMDB', 'SST']):
    ax_hist = axes[row, 0]
    ax_bar = axes[row, 1]
    lens = datasets[dataset_name]
    all_lens = np.concatenate([lens['train'], lens['dev'], lens['test']])

    # Histogram
    ax_hist.hist(all_lens, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    median_l = np.median(all_lens)
    mean_l = np.mean(all_lens)
    ax_hist.axvline(median_l, color='red', linestyle='--', linewidth=1.5)
    ax_hist.axvline(mean_l, color='darkorange', linestyle=':', linewidth=1.5)
```

A **histogram** groups data into bins (40 bins here) and counts how many reviews fall in each bin. The vertical lines show the median (red dashed) and mean (orange dotted).

```python
    # Bar chart
    modes = results[dataset_name]
    x = np.arange(len(modes))
    gains = [modes[m]['Gain'] * 100 for m in modes]  # convert to percentage
    colors = ['#2ecc71' if g >= 0 else '#e74c3c' for g in gains]
    ax_bar.barh(x, gains, color=colors, height=0.5)
    ax_bar.set_yticks(x)
    ax_bar.set_yticklabels(list(modes.keys()))
    ax_bar.axvline(0, color='black', linewidth=0.8)
```

Horizontal bar chart. Green bars = gated attention helps (positive gain). Red bars = gated attention hurts (negative gain). The gain is computed as `(gated - baseline) / baseline`.

```python
plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), '..', 'LatexReport', 'figures')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'length_analysis.pdf'), bbox_inches='tight')
```

Saves the figure as a PDF vector graphic. `bbox_inches='tight'` removes unnecessary whitespace.

**NLP concept — Distribution Analysis:** Understanding the data distribution (how long are reviews?) helps explain *why* certain models work. This is the kind of analysis that scores high on the rubric.

---

## Part 3: Connecting Code → Experiments → Report

### 3.1 How a Single Experiment Flows

Here's the complete path from running a command to printing a number in the report:

**Step 1 — Run:**
```bash
python -m src.train --dataset cfimdb --frozen --model gated_attention_pool
```

**Step 2 — `train.py` creates:** `experiments/cfimdb_frozen_gated/results.json`

**Step 3 — The JSON contains:**
```json
{
  "best_dev": 0.9510204081632653,
  "test_acc": 0.9510204081632653,
  "history": {
    "train_acc": [0.588, 0.736, ..., 0.949],
    "dev_acc": [0.808, 0.833, ..., 0.951]
  }
}
```

**Step 4 — The report reads this data and produces:**

> "...gated attention pooling achieves 95.10% dev accuracy versus 88.57% for the baseline (+7.33%)..."

The number `95.10` comes from `best_dev: 0.9510...` in `results.json`. The `+7.33%` is computed as `(0.9510 - 0.8857) / 0.8857`.

**Line in the report:** Each result row in the table maps to one experiment's `results.json`.

### 3.2 Reading a Results File End to End

Let's walk through `experiments/cfimdb_finetune_gated/results.json`:

```json
{
  "name": "cfimdb_finetune_gated",
```

From `args.name` or auto-generated: `{dataset}_{frozen|finetune}`

```json
  "best_dev": 0.9551020408163265,
```

The highest dev accuracy seen during training. Saved at `src/train.py:158-159` when `da > best_dev`.

```json
  "test_acc": 0.9551020408163265,
```

Final evaluation on the test set (or dev set if test labels are hidden). Set at `src/train.py:166` or `src/train.py:168`.

```json
  "history": {
    "train_loss": [0.891, 0.341, ..., 0.017],
    "train_acc": [0.559, 0.868, ..., 0.995],
    "dev_loss": [0.583, 0.186, ..., 0.175],
    "dev_acc": [0.706, 0.922, ..., 0.943]
  }
}
```

Each array has one entry per epoch (10 epochs = 10 entries). These create the training curves in Figure 4 of the report.

**Overfitting analysis from this file:**
- Epoch 7: train_acc=0.990, dev_acc=0.955 (best dev)
- Epoch 10: train_acc=0.995, dev_acc=0.943 (dev DECLINED)

This gap → overfitting. The report's Limitations section discusses this.

### 3.3 Which Lines of Code Produce Which Report Sections

| Report Section | Source Code | Data File |
|---|---|---|
| Table 1 (results) | `experiments/*/results.json` → `best_dev`, `test_acc` | All JSONs |
| Figure 4 (curves) | `results.json` → `history` | `cfimdb_curves.pdf` |
| Figure 5 (length) | `notebooks/length_analysis.py` | `length_analysis.pdf` |
| Figure 6 (attention) | Model's attention weights | `attention_weights.pdf` |
| Overfitting discussion | `results.json` → train_acc vs dev_acc | CFIMDB fine-tune JSONs |
| Gated vs softmax comparison | `src/model.py` lines 135-155 vs 81-101 | Both CFIMDB frozen JSONs |

### 3.4 Understanding the Overfitting Pattern

**Experiment: CFIMDB fine-tune, gated attention**

```
Epoch  Train Acc  Dev Acc
1      0.559      0.706
2      0.868      0.922
3      0.947      0.943
4      0.972      0.935
5      0.981      0.943
6      0.983      0.951
7      0.990      0.955  ← best dev, saved as checkpoint
8      0.997      0.947
9      0.995      0.931
10     0.995      0.943
```

**What's happening:**
- Epochs 1-3: Model learns quickly (dev climbs)
- Epochs 4-7: Model continues learning (dev slowly climbs)
- Epochs 8-10: Model memorizes training data (train hits 0.997) but loses generalization (dev drops)

**Why it happens:** GPT-2 has 124M parameters. CFIMDB has 1,701 training examples. That's 73,000 parameters per example — the model can literally memorize every review.

**Why frozen avoids this:** With only 769 + classifier head parameters trainable, the model has only ~4,600 degrees of freedom — too few to memorize 1,701 examples. It must learn general patterns.

**NLP concept — The Parameter-to-Data Ratio:** More parameters than data points = likely overfitting. Rule of thumb: you want at least 10-100x more data points than parameters in your classification head.

---

## Part 4: Quick Reference

### Dimension Cheat Sheet

| Tensor | Shape | Created At |
|--------|-------|-----------|
| `input_ids` | `(batch, seq_len)` | `dataset.py:32` |
| `attention_mask` | `(batch, seq_len)` | `dataset.py:33` |
| `last_hidden_state` | `(batch, seq_len, 768)` | `model.py:41` |
| `sentence_repr` (last-token) | `(batch, 768)` | `model.py:44` |
| `sentence_repr` (mean) | `(batch, 768)` | `model.py:76` |
| `sentence_repr` (attention) | `(batch, 768)` | `model.py:101` |
| `sentence_repr` (gated) | `(batch, 768)` | `model.py:155` |
| `logits` | `(batch, num_classes)` | `model.py:46` |
| `loss` | scalar (0D) | `train.py:67` |

### Dimension Flow Diagram

```
input_ids          attention_mask
(batch, seq_len)   (batch, seq_len)
        \               /
         GPT-2 Model
              |
    last_hidden_state
    (batch, seq_len, 768)
              |
        Pooling layer
              |
    sentence_repr
       (batch, 768)
              |
        Dropout
              |
        Linear(768, num_classes)
              |
          logits
     (batch, num_classes)
              |
      torch.argmax(dim=1)
              |
       predicted class
          (batch,)
```

### Glossary with Line Numbers

| Term | Definition | First appears at |
|------|-----------|-----------------|
| **token ID** | Integer representing a word/subword in GPT-2's vocabulary | `dataset.py:32` |
| **attention_mask** | 1=real token, 0=padding token | `dataset.py:33` |
| **hidden_size** | 768 — dimension of each token's representation | `model.py:29` |
| **embedding** | Lookup table mapping token IDs → vectors | `model.py:28` (inside GPT2Model) |
| **hidden state** | Vector of 768 numbers representing a token in context | `model.py:41` |
| **pooling** | Combining per-token vectors into one sentence vector | `model.py:42` |
| **last-token pooling** | Using final token's hidden state as sentence representation | `model.py:42-44` |
| **mean pooling** | Averaging all non-padding token hidden states | `model.py:74-76` |
| **softmax attention** | Learned weighted average where weights sum to 1 | `model.py:100` |
| **gated attention** | Learned weighted average with independent sigmoid gates | `model.py:151-154` |
| **logits** | Raw class scores before softmax | `model.py:46` |
| **cross-entropy loss** | Measures difference between predicted and true labels | `train.py:67` |
| **gradient accumulation** | Simulating larger batch size by summing gradients over N steps | `train.py:67-71` |
| **overfitting** | Model memorizes training data, fails on new data | `model.py:32-34` (freeze prevents it) |
| **frozen** | GPT-2 parameters locked, only classifier head trains | `model.py:32-34` |
| **fine-tuning** | All parameters (124M) are updated during training | `model.py:36-37` |
| **dropout** | Randomly disables 10% of neurons during training | `model.py:30` |
| **AdamW** | Optimizer that adapts learning rate per parameter | `train.py:144` |
| **causal attention** | Left-to-right: each token only sees previous tokens | `model.py:28` (built into GPT2Model) |
| **batch_size** | Number of examples processed simultaneously | `train.py:105` |
| **epoch** | One complete pass through the training data | `train.py:148` |
| **checkpoint** | Saved model weights (best dev accuracy) | `train.py:159` |

### Dataset Comparison

| Feature | SST | CFIMDB |
|---------|-----|--------|
| Task | 5-class sentiment | Binary (positive/negative) |
| Train examples | 8,544 | 1,701 |
| Dev examples | 1,101 | 245 |
| Test examples | 2,210 | 488 |
| Mean token length | 22 | 214 |
| Max tokens | 128 | 512 |
| Data format | Penn Treebank trees | CSV files |
| Difficulty | Hard (5 classes, short) | Easy (2 classes, long) |
| Random baseline | 20% | 50% |
| Frozen baseline | ~48% | ~89% |
| Fine-tune baseline | ~52% | ~97% |

### Common Mistakes & How to Avoid Them

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Not freezing | GPU out of memory | Add `--frozen` flag |
| Wrong learning rate | Loss doesn't decrease | Frozen: `--lr 3e-3`, fine-tune: `--lr 1e-5` |
| Batch too large | CUDA out of memory | Use `--batch-size 8 --accum 2` |
| Train on test data | Inflated scores | Never look at test until final evaluation |
| Tokenizer not working | "pad_token is not set" error | `tokenizer.pad_token = tokenizer.eos_token` |
| Overfitting | Train acc >> dev acc | Freeze the encoder, or reduce epochs |
| Missing baseline | Can't measure improvement | Always run `baseline` first, under same conditions |

### Adding a New Model: Checklist

1. **Write the class** in `src/model.py` — follow the `GPT2Classifier` pattern
2. **Add pooling logic** in `forward()` — modify how hidden states → sentence_repr
3. **Register in `MODEL_REGISTRY`** (`src/model.py:193-198`)
4. **Train:** `python -m src.train --model your_name --dataset cfimdb --frozen`
5. **Compare:** Open `experiments/your_name/results.json` and check `best_dev`
6. **Don't forget the baseline:** You need `baseline` numbers for fair comparison

---

## Summary: What You Should Know Now

After reading this guide and the code alongside it, you should be able to:

1. **Explain** what each of the 4 pooling strategies does and when it helps
2. **Modify** the code to add a new pooling strategy
3. **Run** experiments with `python -m src.train`
4. **Read** `results.json` to understand what happened
5. **Diagnose** overfitting by comparing train_acc and dev_acc
6. **Connect** experiment numbers to the LaTeX report
7. **Understand** why gated attention works better on long documents with frozen encoders

The key insight to take away: **pooling strategy matters most when the encoder is frozen and documents are long.** When you fine-tune, the encoder adapts to the task regardless of pooling. When documents are short, the last token "sees" everything anyway.
