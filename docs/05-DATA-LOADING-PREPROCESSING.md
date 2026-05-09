# 05 — Data Loading & Preprocessing

This guide covers loading both datasets and preparing them for GPT-2 training.

---

## 1. Imports

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

---

## 2. Load SST with pytreebank

```python
def load_sst_data():
    """Load SST data from tree files into a DataFrame."""
    data = []
    
    for split in ['train', 'dev', 'test']:
        path = f"./Datasets/SST2Data/trainDevTestTrees_PTB/trees/{split}.txt"
        trees = pytreebank.load_tree_file(path, format="custom")
        
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
print(f"\nLabel distribution:\n{sst_df.groupby('split')['label'].value_counts()}")
```

---

## 3. Load CFIMDB / IMDB

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
    
    # Split
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
print(cfimdb_df.head())
print(f"\nSplit sizes:\n{cfimdb_df['split'].value_counts()}")
```

---

## 4. Tokenization with GPT-2 Tokenizer

GPT-2 uses **Byte-Pair Encoding (BPE)**. It has its own tokenizer.

```python
# Load the tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# GPT-2 doesn't have a padding token by default — add one
tokenizer.pad_token = tokenizer.eos_token
# Or: tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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

### Important: Max Length

GPT-2 has a maximum context length of **1024 tokens**. Reviews can be longer.

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

If many reviews exceed 1024, you have options:
- **Truncate** to 1024 (lose the end of the review)
- **Truncate to first N + last N** (take head + tail)
- **Use a model with longer context** (GPT-2-medium is also 1024)

---

## 5. PyTorch Dataset Class

```python
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.texts = df['text'].values
        self.labels = df['label'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),      # (max_length,)
            'attention_mask': encoding['attention_mask'].squeeze(0),  # (max_length,)
            'label': torch.tensor(label, dtype=torch.long)
        }
```

---

## 6. Data Loaders

```python
def create_dataloaders(df, tokenizer, batch_size=16, max_length=512):
    dataloaders = {}
    
    for split in ['train', 'dev', 'test']:
        split_df = df[df['split'] == split].reset_index(drop=True)
        dataset = SentimentDataset(split_df, tokenizer, max_length)
        
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2
        )
    
    return dataloaders

# Create dataloaders for SST (shorter, use max_length=128)
sst_loaders = create_dataloaders(sst_df, tokenizer, batch_size=16, max_length=128)

# Create dataloaders for CFIMDB (longer, use max_length=512)
cfimdb_loaders = create_dataloaders(cfimdb_df, tokenizer, batch_size=16, max_length=512)

# Verify
for split, loader in sst_loaders.items():
    batch = next(iter(loader))
    print(f"SST {split}: input_ids {batch['input_ids'].shape}, labels {batch['label'].shape}")
```

**Expected output:**
```
SST train: input_ids torch.Size([16, 128]), labels torch.Size([16])
SST dev: input_ids torch.Size([16, 128]), labels torch.Size([16])
SST test: input_ids torch.Size([16, 128]), labels torch.Size([16])
```

---

## 7. Full Data Loading Function

Put it all together:

```python
def load_all_data(tokenizer, batch_size=16):
    """Load both SST and CFIMDB datasets and return dataloaders."""
    # Load data
    sst_df = load_sst_data()
    cfimdb_df = load_cfimdb_data()
    
    # Create dataloaders
    sst_loaders = create_dataloaders(sst_df, tokenizer, batch_size, max_length=128)
    cfimdb_loaders = create_dataloaders(cfimdb_df, tokenizer, batch_size, max_length=512)
    
    return {
        'sst': (sst_df, sst_loaders),
        'cfimdb': (cfimdb_df, cfimdb_loaders)
    }

# Test
data = load_all_data(tokenizer, batch_size=16)
print(f"SST train batches: {len(data['sst'][1]['train'])}")
print(f"CFIMDB train batches: {len(data['cfimdb'][1]['train'])}")
```

**Expected output:**
```
SST train batches: 534
CFIMDB train batches: 2500
```

---

## Key Parameters Summary

| Parameter | SST | CFIMDB |
|-----------|-----|--------|
| `max_length` | 128 (sentences are short) | 512 (reviews are long) |
| `batch_size` | Start with 16 | Start with 8-16 (depends on GPU) |
| `num_classes` | 5 | 2 |
| Train examples | 8,544 | 40,000 |
