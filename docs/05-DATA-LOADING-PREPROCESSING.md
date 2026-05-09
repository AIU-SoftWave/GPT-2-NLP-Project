# 05 — Data Loading & Preprocessing (Beginner-Friendly)

## What You'll Learn

- How to load SST and CFIMDB datasets into a format the model can use
- What **tokenization** is and why it matters
- What **padding** and **attention masks** are
- How PyTorch DataLoaders feed data to the model in batches

---

## Key Terms (Defined Simply)

| Term | Simple definition | Analogy |
|------|------------------|---------|
| **Tokenization** | Splitting text into smaller pieces and converting them to numbers | Translating English words into secret code numbers |
| **Token** | A piece of text (a word or part of a word) | A single word in a sentence, or part of a long word |
| **Padding** | Adding dummy tokens to make all sentences the same length | Adding empty seats on a bus so everyone has a spot |
| **Attention mask** | A list of 1s and 0s that tells the model which tokens are real (1) and which are padding (0) | A seatbelt sign — 1 means "person sitting here," 0 means "empty seat" |
| **Truncation** | Cutting off text that's too long | Folding a long letter to fit in an envelope |
| **Batch** | A group of examples processed at once | Instead of serving one customer at a time, serving a table of 16 at once |

---

## 1. Imports (What We Need)

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytreebank
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
import re
import os
from tqdm import tqdm
```

**What each import does:**

| Import | What it is | What we use it for |
|--------|-----------|-------------------|
| `torch` | PyTorch — the deep learning engine | Runs the model, stores data as tensors |
| `Dataset`, `DataLoader` | PyTorch data tools | Organize and feed data to the model |
| `GPT2Tokenizer` | HuggingFace's tokenizer | Converts text into numbers GPT-2 understands |
| `train_test_split` | scikit-learn tool | Splits data into train/dev/test |
| `tqdm` | Progress bar | Shows how training is progressing |

---

## 2. Loading SST with pytreebank

### What we're doing:

1. Read the tree files (train.txt, dev.txt, test.txt)
2. Parse each tree to extract the sentence text and its label
3. Store everything in a pandas DataFrame (like a spreadsheet)

```python
def load_sst_data():
    """Load SST data from tree files into a DataFrame."""
    data = []
    
    for split in ['train', 'dev', 'test']:
        path = f"./Datasets/SST2Data/trainDevTestTrees_PTB/trees/{split}.txt"
        trees = pytreebank.import_tree_corpus(path)
        
        for tree in trees:
            sentence = tree.to_lines()[0]
            label = tree.label
            data.append({
                'text': sentence,
                'label': label,       # 0-4 integer
                'split': split
            })
    
    df = pd.DataFrame(data)
    return df

sst_df = load_sst_data()
print(f"SST rows: {len(sst_df)}")
print(sst_df.head())
```

**What the output looks like:**
```
                                                  text  label  split
0  The Rock is destined to be the 21st Century 's ...      3  train
1  The gorgeously elaborate continuation of `` Th ...      4  train
...
```

A **DataFrame** is like an Excel sheet in Python. Each row is one example with:
- **text**: The movie review sentence
- **label**: The sentiment (0-4)
- **split**: Which set it belongs to (train/dev/test)

---

## 3. Loading CFIMDB / IMDB

### What we're doing:

1. Read the CSV file
2. Clean HTML tags from reviews
3. Convert "positive"/"negative" to numbers (1/0)
4. Create train/dev/test splits

```python
def load_cfimdb_data():
    """Load IMDB CSV and create train/dev/test splits."""
    df = pd.read_csv("./Datasets/CFIMDB/IMDB Dataset.csv")
    
    # Clean HTML tags
    def clean_text(text):
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    df['text'] = df['review'].apply(clean_text)
    df['label'] = (df['sentiment'] == 'positive').astype(int)  # 0 or 1
    
    # Split into train (80%), dev (10%), test (10%)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )
    
    train_df['split'] = 'train'
    dev_df['split'] = 'dev'
    test_df['split'] = 'test'
    
    result = pd.concat([train_df, dev_df, test_df])
    return result[['text', 'label', 'split']]

cfimdb_df = load_cfimdb_data()
print(f"CFIMDB rows: {len(cfimdb_df)}")
```

> **Note:** `random_state=42` means we get the same split every time we run this. It doesn't matter that it's 42 specifically — it's just a way to make the random split reproducible.

---

## 4. Tokenization with GPT-2 Tokenizer

### What is tokenization? (The most important concept)

**Tokenization** is the process of converting text into numbers that the model can process.

Here's the problem: Computers understand numbers, not words. We need a way to map every word (or word piece) to a unique number.

### How GPT-2's tokenizer works

GPT-2 uses something called **Byte-Pair Encoding (BPE)**. Here's how it works:

1. Common words like "the", "movie", "was" get their own token ID:
   - "the" → 464
   - "movie" → 3181
   - "was" → 373

2. Rare words get split into smaller pieces:
   - "fantastic" → "fant" + "astic" (because "fantastic" is less common as a whole)
   - "Googling" → "Goog" + "ling"

3. Each piece gets a unique number

> **Analogy:** Think of it like building with LEGO bricks. Common words are standard bricks (you have lots of them). Rare words are custom pieces you might not have, so you build them from smaller standard bricks.

```python
# Load the tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# GPT-2 doesn't have a padding token by default — we add one
tokenizer.pad_token = tokenizer.eos_token

# Test it
text = "The movie was fantastic!"
tokens = tokenizer(text, return_tensors="pt")
print(f"Text: {text}")
print(f"Tokens: {tokens['input_ids'][0].tolist()}")
print(f"Decoded: {tokenizer.decode(tokens['input_ids'][0])}")
print(f"Length: {len(tokens['input_ids'][0])}")
```

**Expected output:**
```
Text: The movie was fantastic!
Tokens: [464, 3181, 373, 35715, 0]
Decoded: The movie was fantastic!
Length: 5
```

**Breaking down the output:**
- `[464, 3181, 373, 35715, 0]` — these are the token IDs
- `464` = "The", `3181` = " movie", `373` = " was", `35715` = " fantastic", `0` = "!"
- The tokenizer also added an end-of-sequence token (50256) at the end, which we see as 0 because our example was too short
- "Decoded" proves we can convert numbers back to text without losing meaning

### Important: Max Length

GPT-2 can only handle up to **1024 tokens** at once. This is called its **context length**.

```python
# Check lengths for both datasets
sst_df['token_count'] = sst_df['text'].apply(lambda x: len(tokenizer.encode(x)))
cfimdb_df['token_count'] = cfimdb_df['text'].apply(lambda x: len(tokenizer.encode(x)))

print(f"SST - Max tokens: {sst_df['token_count'].max()}, Avg: {sst_df['token_count'].mean():.0f}")
print(f"CFIMDB - Max tokens: {cfimdb_df['token_count'].max()}, Avg: {cfimdb_df['token_count'].mean():.0f}")

# How many CFIMDB reviews exceed 1024 tokens?
over_limit = (cfimdb_df['token_count'] > 1024).sum()
print(f"CFIMDB reviews > 1024 tokens: {over_limit} ({over_limit/len(cfimdb_df)*100:.1f}%)")
```

SST sentences are short (~22 tokens), so no problem. But some CFIMDB reviews are very long (~285 tokens avg, up to 3000+). For those, we'll **truncate** (cut off the extra).

---

## 5. PyTorch Dataset Class

### What is a Dataset?

A PyTorch **Dataset** is a class that knows how to get one example at a time. Think of it like a **recipe card box** — you ask for card #5, and it gives you the ingredients and instructions for that recipe.

```python
class SentimentDataset(Dataset):
    """Turns our DataFrame into a format PyTorch can use."""
    
    def __init__(self, df, tokenizer, max_length=512):
        self.texts = df['text'].values
        self.labels = df['label'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """How many total examples do we have?"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get one example (by index) and tokenize it."""
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,         # Cut off text that's too long
            padding='max_length',    # Pad short texts to max_length
            max_length=self.max_length,
            return_tensors='pt'      # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
```

### Understanding __getitem__

When we ask for example #5, this function:

1. Gets the text and label for example #5
2. **Tokenizes** the text — converts words to numbers
3. **Truncates** if longer than max_length
4. **Pads** if shorter than max_length (fills the rest with 0s)
5. Returns a dictionary with:
   - **input_ids**: The token numbers (e.g., `[464, 3181, 373, 35715, 0, 0, 0, ...]`)
   - **attention_mask**: Which tokens are real (1) vs padding (0) (e.g., `[1, 1, 1, 1, 1, 0, 0, 0, ...]`)
   - **label**: The sentiment score (e.g., `3`)

### Why padding and attention_mask?

**Padding:** We need to process examples in batches. All examples in a batch must be the same length. So we pad shorter sentences with 0s.

**Attention mask:** The model needs to know which tokens are real words and which are just padding. The attention mask is a list of 1s (real) and 0s (padding).

> **Analogy:** Imagine packing items into boxes of a fixed size. If your item is small, you add packing peanuts (padding). The attention mask is like a note saying "ignore the packing peanuts."

---

## 6. Data Loaders

### What is a DataLoader?

A **DataLoader** takes a Dataset and adds two features:
1. **Batching**: Groups examples into batches of a fixed size
2. **Shuffling**: Randomizes the order (training only)

Think of it like a **conveyor belt** that brings examples to the model in organized groups.

```python
def create_dataloaders(df, tokenizer, batch_size=16, max_length=512):
    dataloaders = {}
    
    for split in ['train', 'dev', 'test']:
        split_df = df[df['split'] == split].reset_index(drop=True)
        dataset = SentimentDataset(split_df, tokenizer, max_length)
        
        # Shuffle only the training set
        shuffle = (split == 'train')
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0   # 0 = main process (safest for notebooks)
            # If running interactively, try num_workers=4 for faster loading
        )
    
    return dataloaders

# Create dataloaders
sst_loaders = create_dataloaders(sst_df, tokenizer, batch_size=64, max_length=128)
cfimdb_loaders = create_dataloaders(cfimdb_df, tokenizer, batch_size=32, max_length=512)
```

### Why different sizes for SST and CFIMDB?

| Parameter | SST | CFIMDB | Why? |
|-----------|-----|--------|------|
| `max_length` | 128 | 512 | SST sentences are short; CFIMDB reviews are long |
| `batch_size` | 64 | 32 | Larger batch fits in GPU memory for short texts |

### Verifying the DataLoaders

```python
for split, loader in sst_loaders.items():
    batch = next(iter(loader))
    print(f"SST {split}: input_ids {batch['input_ids'].shape}, labels {batch['label'].shape}")
```

**Expected output:**
```
SST train: input_ids torch.Size([64, 128]), labels torch.Size([64])
SST dev: input_ids torch.Size([64, 128]), labels torch.Size([64])
SST test: input_ids torch.Size([64, 128]), labels torch.Size([64])
```

**What this tells us:**
- `[64, 128]` means: 64 examples (batch), each 128 tokens long
- `[64]` means: 64 labels, one per example
- Everything is working correctly!

---

## 7. Full Data Loading Function

Putting it all together:

```python
def load_all_data(tokenizer, batch_size=64):
    """Load both SST and CFIMDB datasets and return dataloaders."""
    sst_df = load_sst_data()
    cfimdb_df = load_cfimdb_data()
    
    sst_loaders = create_dataloaders(sst_df, tokenizer, batch_size, max_length=128)
    cfimdb_loaders = create_dataloaders(cfimdb_df, tokenizer, batch_size, max_length=512)
    
    return {
        'sst': (sst_df, sst_loaders),
        'cfimdb': (cfimdb_df, cfimdb_loaders)
    }

data = load_all_data(tokenizer, batch_size=64)
```

**Expected output:**
```
SST train batches: 534
CFIMDB train batches: 2500
```

This means during one epoch (one full pass through the data):
- For SST: 134 batches of 64 examples each = 8,544 total examples
- For CFIMDB: 1,250 batches of 32 = 40,000 total examples

---

## Summary

**The pipeline from raw text to model-ready data:**

```
Raw text → Tokenizer → Token IDs → Pad/Truncate → Dataset → DataLoader → Model
                              ↓
                      Attention Mask
                         (1=real, 0=pad)
```

**Key parameters:**

| Parameter | SST | CFIMDB |
|-----------|-----|--------|
| `max_length` | 128 | 512 |
| `batch_size` | 64 | 32 |
| `num_classes` | 5 | 2 |
| Train examples | 8,544 | 40,000 |
