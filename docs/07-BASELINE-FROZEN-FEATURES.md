# 07 — Baseline: Frozen Feature Extractor (Beginner-Friendly)

## What You'll Learn

- How to train the frozen GPT-2 baseline (first experiment)
- What the training loop actually does, step by step
- What loss, optimizer, and accuracy mean in practice
- Expected results and what they tell us

---

## Key Terms (Defined Simply)

| Term | Simple definition | Analogy |
|------|------------------|---------|
| **Loss** | A number that measures how wrong the model's predictions are | The number of points you lose in a game — lower is better |
| **Cross-entropy loss** | The standard loss for classification problems | Like a scoring system that penalizes confident wrong answers more |
| **Optimizer** | The algorithm that decides how to update model parameters | A teacher who adjusts your answers — tells you which direction to move |
| **Learning rate** | How big of a step the optimizer takes when updating parameters | The size of each step when walking downhill — too big and you overshoot, too small and it takes forever |
| **Epoch** | One complete pass through ALL training examples | Reading your textbook cover to cover once |
| **Batch** | A small group of examples processed at once | Studying 16 flashcards at a time instead of all 8,544 at once |
| **Accuracy** | The percentage of predictions that are correct | Your score on a test — number right divided by total questions |
| **Overfitting** | When the model memorizes the training data but fails on new data | Cramming for a test by memorizing the answers, then failing when a question is slightly different |

---

## The Model

```
Sentence → GPT-2 (frozen) → hidden states → last token → Dropout → Linear → logits
                                                                   ↑
                                                           Only this is trained
                                                           (3,845 parameters)
```

**Frozen** means GPT-2's 124 million parameters stay exactly as they are. Only the tiny classifier head (3,845 parameters) is trained.

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

These are all the tools we need. Let's go through them:
- `torch` — PyTorch, the engine that runs everything
- `AdamW` — The optimizer (decides how to update the model's weights)
- `accuracy_score` — Calculates what percentage of predictions are correct
- `tqdm` — Shows a progress bar during training

---

### 2. Data Loading

```python
def load_sst_data():
    data = []
    for split in ['train', 'dev', 'test']:
        path = f"./Datasets/SST2Data/trainDevTestTrees_PTB/trees/{split}.txt"
        trees = pytreebank.import_tree_corpus(path)
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

---

### 3. Training Function (What the Loop Does)

#### Understanding train_epoch

Let's break down what happens every time we train for one epoch:

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # Tell the model: "we're training now" (enables dropout, etc.)
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        # Step 1: Move data to GPU (if available)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Step 2: Clear old gradients
        optimizer.zero_grad()
        
        # Step 3: Forward pass — get predictions
        logits = model(input_ids, attention_mask)
        
        # Step 4: Calculate loss — how wrong are we?
        loss = criterion(logits, labels)
        
        # Step 5: Backward pass — calculate gradients
        loss.backward()
        
        # Step 6: Update weights — make the model slightly less wrong
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy
```

**Step by step, in plain English:**

1. **`.to(device)`** — Move data to the GPU. GPUs are specialized for the kind of math neural networks do.

2. **`optimizer.zero_grad()`** — Clear the gradients from the previous batch. Imagine writing on a whiteboard — you erase before writing new information.

3. **Forward pass** — Feed the input through the model to get predictions. This is the model "thinking" about the input.

4. **Loss** — Compare predictions to the correct answers. The loss is a single number saying how wrong the model is. Cross-entropy loss is standard for classification.

5. **Backward pass** — Calculate how each parameter contributed to the error. The model figures out: "If I adjust this knob slightly, will I be less wrong?"

6. **Optimizer step** — Actually adjust the parameters. The optimizer takes the gradients and updates all the parameters.

#### Understanding evaluate

```python
def evaluate(model, dataloader, criterion, device):
    model.eval()  # Tell the model: "evaluation mode" (disables dropout)
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Don't track gradients (saves memory)
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

**Difference from training:**
- `model.eval()` — Disables dropout (we want consistent predictions)
- `torch.no_grad()` — Don't track gradients (no training, so no need for gradients)
- No `optimizer.step()` — We're not updating anything, just measuring

> **Analogy:** Training is like practicing a sport with a coach (you get feedback and adjust). Evaluation is like a referee watching a game and keeping score (no coaching, just measurement).

---

### 4. Main Training Loop

```python
def run_experiment(dataset_name="sst", freeze=True, batch_size=64, lr=3e-3, epochs=10):
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

### What happens during training:

```
Epoch 1:  Train → calculate error → update weights → check dev accuracy
Epoch 2:  Train → calculate error → update weights → check dev accuracy
...
Epoch 10: Train → calculate error → update weights → check dev accuracy

After all epochs: Load the model with the BEST dev accuracy → test on test set
```

**Why save the best model?** The model might get worse after a certain point (overfitting). We keep the version that performed best on the dev set.

**What is "best"?** Highest accuracy on the dev set (not training set). We don't care about training accuracy — we care about how well the model generalizes to new data.

---

### 5. Run SST Frozen Baseline

```python
sst_dev, sst_test = run_experiment(
    dataset_name="sst",
    freeze=True,       # GPT-2 stays frozen
    batch_size=64,
    lr=3e-3,           # Higher learning rate (only 3,845 params to train)
    epochs=10
)

**Expected output:** ~0.45 (45%) dev accuracy

**Training time:** ~1-2 min on GPU

**What does 45% mean?** Random guessing would give 20% (since there are 5 classes). So 45% is much better than random, but not great. SST is a hard dataset — short sentences with 5 subtle sentiment levels.

---

### 6. Run CFIMDB Frozen Baseline

```python
cfimdb_dev, cfimdb_test = run_experiment(
    dataset_name="cfimdb",
    freeze=True,
    batch_size=32,     # Smaller batch because reviews are longer
    lr=3e-3,
    epochs=5           # CFIMDB converges faster (easier task)
)

**Expected output:** ~0.83-0.90 (83-90%) dev accuracy

**Training time:** ~3-5 min on GPU

**Why is CFIMDB so much higher than SST?**
1. **Binary classification** — only 2 choices instead of 5
2. **Longer text** — more context to work with (~234 words vs ~19 words)
3. **More data** — 40,000 training examples vs 8,544

---

## Reading the Training Output

During training, you'll see output like:

```
Epoch 1/10
Training: 100%|██████████| 134/134 [00:07<00:00, 17.8it/s]
Evaluating: 100%|██████████| 18/18 [00:01<00:00, 20.5it/s]
Train Loss: 1.5234 | Train Acc: 0.3245
Dev Loss:   1.4876 | Dev Acc:   0.3512
```

**What to look for:**

| Signal | Good sign | Bad sign |
|--------|-----------|----------|
| Train Loss | Decreasing each epoch | Increasing or staying flat |
| Dev Loss | Decreasing (parallels training) | Starts increasing while train loss decreases (OVERFITTING!) |
| Train Accuracy | Increasing each epoch | Stuck at same value |
| Dev Accuracy | Increasing (parallels training) | Stays low or decreases |

**For the frozen baseline:** The loss should steadily decrease and accuracy should increase. Since only the classifier head is training, convergence is fast (5-10 epochs is enough).

---

## Expected Results Summary

| Experiment | Dev Accuracy | Test Accuracy | Time |
|-----------|-------------|---------------|------|
| SST Frozen | ~45% | ~44% | ~1-2 min |
| CFIMDB Frozen | ~83-90% | ~82-89% | ~3-5 min |

---

## Hyperparameter Notes

| Parameter | SST Frozen | CFIMDB Frozen |
|-----------|-----------|---------------|
| Learning rate | 3e-3 | 3e-3 |
| Batch size | 64 | 32 |
| Dropout | 0.1 | 0.1 |
| Epochs | 10 | 5 |

**Why is the learning rate 3e-3 (0.003)?** Since we're only training 3,845 parameters (not 124 million), we can use a higher learning rate. The model can take bigger steps because there's less risk of "breaking" anything.

---

## Common Issues

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Loss not decreasing | GPT-2 not actually frozen (gradients flowing) | Check `param.requires_grad` is False for GPT-2 params |
| CUDA out of memory | Batch too large for GPU | Reduce batch_size or max_length |
| Accuracy stuck at ~20% (SST) | Label values wrong | Check labels are 0-4, not 1-5 |
| Tokenizer errors | Padding token not set | Ensure `tokenizer.pad_token = tokenizer.eos_token` |
| Accuracy much lower than expected | Training loop bug | Compare your results with the expected values |

---

## What These Baseline Results Tell You

Your frozen baseline results are the **starting point**. Your innovation should try to improve on these numbers.

| Dataset | Baseline (Frozen) | Your Goal |
|---------|-------------------|-----------|
| SST | ~45% | Beat 45% |
| CFIMDB | ~83-90% | Beat 83-90% |

**Note:** If you get CFIMDB accuracy much higher than 83% (e.g., 90%), that's because our dataset has 40K training examples vs the original 1.7K. Your innovation will need to improve on YOUR baseline, not the paper's.
