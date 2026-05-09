# 07 — Baseline: Frozen Feature Extractor

This is your first baseline. GPT-2 is **frozen** — only the classification head is trained.

---

## The Model

```
Sentence → GPT-2 (frozen) → hidden states → last token → Dropout → Linear → logits
                                                                    ↑
                                                            Only this is trained
```

---

## Full Training Script

### 1. Setup

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import Dataset, DataLoader
import pytreebank
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import os
import re
from sklearn.model_selection import train_test_split
```

### 2. Data Loading (from previous guide)

```python
def load_sst_data():
    data = []
    for split in ['train', 'dev', 'test']:
        path = f"./Datasets/SST2Data/trainDevTestTrees_PTB/trees/{split}.txt"
        trees = pytreebank.load_tree_file(path, format="custom")
        for tree in trees:
            data.append({
                'text': tree.to_lines()[0],
                'label': tree.label,
                'split': split
            })
    return pd.DataFrame(data)

def load_cfimdb_data():
    df = pd.read_csv("./Datasets/CFIMDB/IMDB Dataset.csv")
    def clean_text(text):
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    df['text'] = df['review'].apply(clean_text)
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    train_df['split'] = 'train'
    dev_df['split'] = 'dev'
    test_df['split'] = 'test'
    return pd.concat([train_df, dev_df, test_df])[['text', 'label', 'split']]

class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['text'].values
        self.labels = df['label'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
```

### 3. Model Definition

```python
class GPT2Classifier(nn.Module):
    def __init__(self, model_name="gpt2", num_classes=5, dropout=0.1, freeze=True):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        hidden_size = self.gpt2.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        if freeze:
            for param in self.gpt2.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        # Last token pooling
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        sentence_repr = last_hidden[batch_indices, last_token_idx]
        
        sentence_repr = self.dropout(sentence_repr)
        logits = self.classifier(sentence_repr)
        return logits
```

### 4. Training Function

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy
```

### 5. Main Training Loop

```python
def run_experiment(dataset_name="sst", freeze=True, batch_size=16, lr=1e-3, epochs=10):
    print(f"\n{'='*50}")
    print(f"Experiment: {dataset_name.upper()}, freeze={freeze}")
    print(f"{'='*50}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    if dataset_name == "sst":
        df = load_sst_data()
        num_classes = 5
        max_length = 128
    else:
        df = load_cfimdb_data()
        num_classes = 2
        max_length = 512
    
    # Create dataloaders
    dataloaders = {}
    for split in ['train', 'dev', 'test']:
        split_df = df[df['split'] == split].reset_index(drop=True)
        dataset = SentimentDataset(split_df, tokenizer, max_length)
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Model
    model = GPT2Classifier(
        model_name="gpt2",
        num_classes=num_classes,
        dropout=0.1,
        freeze=freeze
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Training loop
    best_dev_acc = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, dataloaders['train'], optimizer, criterion, device)
        dev_loss, dev_acc = evaluate(model, dataloaders['dev'], criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Dev Loss:   {dev_loss:.4f} | Dev Acc:   {dev_acc:.4f}")
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), f"best_model_{dataset_name}_{'frozen' if freeze else 'finetune'}.pt")
            print(f"✓ New best dev acc: {best_dev_acc:.4f} (saved)")
    
    # Test with best model
    model.load_state_dict(torch.load(f"best_model_{dataset_name}_{'frozen' if freeze else 'finetune'}.pt"))
    test_loss, test_acc = evaluate(model, dataloaders['test'], criterion, device)
    print(f"\n=== Final Results ===")
    print(f"Best Dev Acc: {best_dev_acc:.4f}")
    print(f"Test Acc:     {test_acc:.4f}")
    
    return best_dev_acc, test_acc
```

### 6. Run It

```python
# SST baseline (frozen)
sst_dev, sst_test = run_experiment(
    dataset_name="sst",
    freeze=True,
    batch_size=16,
    lr=1e-3,
    epochs=10
)
```

**Expected output:** ~0.45 (45%) dev accuracy

---

## CFIMDB Baseline

```python
# CFIMDB baseline (frozen) — use smaller batch if GPU memory is limited
cfimdb_dev, cfimdb_test = run_experiment(
    dataset_name="cfimdb",
    freeze=True,
    batch_size=8,   # Reviews are longer, use smaller batch
    lr=1e-3,
    epochs=5        # CFIMDB is easier, converges faster
)
```

**Expected output:** ~0.83-0.90 (83-90%) dev accuracy

---

## Experiment Tracking

Use a dict to track all results:

```python
results = {}

results['sst_frozen'] = run_experiment("sst", freeze=True, batch_size=16, lr=1e-3, epochs=10)
results['cfimdb_frozen'] = run_experiment("cfimdb", freeze=True, batch_size=8, lr=1e-3, epochs=5)

# Display summary
print("\n\n=== RESULTS SUMMARY ===")
for name, (dev, test) in results.items():
    print(f"{name:20s} | Dev: {dev:.4f} | Test: {test:.4f}")
```

---

## Hyperparameter Notes

| Parameter | Frozen SST | Frozen CFIMDB |
|-----------|-----------|---------------|
| Learning rate | 1e-3 (higher is OK since only head trains) | 1e-3 |
| Batch size | 16-32 | 8-16 |
| Dropout | 0.1 | 0.1 |
| Epochs | 10 | 5 |
| Training time | ~5 min (GPU) | ~10 min (GPU) |

---

## Common Issues

| Problem | Fix |
|---------|-----|
| Loss not decreasing | Check that GPT-2 is actually frozen (no gradients) |
| CUDA out of memory | Reduce batch_size or max_length |
| Low accuracy (~20% for SST) | Make sure labels are 0-4 not 1-5 |
| Tokenizer errors | Ensure you set `tokenizer.pad_token = tokenizer.eos_token` |
