# 08 — Fine-Tuning

In this setup, **all GPT-2 parameters are updated** during training. This gives better accuracy but uses more memory and takes longer.

---

## Key Differences from Frozen Baseline

| Aspect | Frozen | Fine-Tuning |
|--------|--------|-------------|
| Trainable params | ~3.8K (0.003%) | 124M (100%) |
| Learning rate | 1e-3 | 2e-5 to 5e-5 (much smaller!) |
| Batch size | 16-32 | 8-16 (smaller due to memory) |
| Training time (SST) | ~5 min | ~30-60 min |
| Memory usage | ~2 GB | ~6-10 GB |
| Dev accuracy (SST) | ~45% | ~52% |
| Dev accuracy (CFIMDB) | ~83% | ~97% |

---

## The Model (same architecture, different freeze flag)

```python
model = GPT2Classifier(
    model_name="gpt2",
    num_classes=5,     # 5 for SST, 2 for CFIMDB
    dropout=0.1,
    freeze=False       # ← This is the key difference
)
```

---

## Training Modifications for Fine-Tuning

### 1. Smaller Learning Rate

Fine-tuning pretrained models requires a much smaller learning rate. Large updates can destroy the learned representations.

```python
# Good learning rates for fine-tuning
lr = 5e-5   # Try: 5e-5, 3e-5, 2e-5, 1e-5

optimizer = AdamW(model.parameters(), lr=lr)
```

### 2. Learning Rate Scheduler

A linear warmup + decay schedule helps stable fine-tuning:

```python
from transformers import get_linear_schedule_with_warmup

total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

### 3. Gradient Clipping

Prevents exploding gradients (common with large models):

```python
# After loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()
```

---

## Full Fine-Tuning Loop

```python
def train_epoch_finetune(model, dataloader, optimizer, criterion, scheduler, device):
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
        
        # Gradient clipping (important for fine-tuning!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy


def run_finetune_experiment(dataset_name="sst", batch_size=8, lr=5e-5, epochs=5):
    print(f"\n{'='*50}")
    print(f"Fine-Tuning Experiment: {dataset_name.upper()}")
    print(f"{'='*50}")
    
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
    
    # Dataloaders
    dataloaders = {}
    for split in ['train', 'dev', 'test']:
        split_df = df[df['split'] == split].reset_index(drop=True)
        dataset = SentimentDataset(split_df, tokenizer, max_length)
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Model — freeze=False for fine-tuning
    model = GPT2Classifier(
        model_name="gpt2",
        num_classes=num_classes,
        dropout=0.1,
        freeze=False
    ).to(device)
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with smaller LR
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Scheduler with warmup
    total_steps = len(dataloaders['train']) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training
    best_dev_acc = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch_finetune(
            model, dataloaders['train'], optimizer, criterion, scheduler, device
        )
        dev_loss, dev_acc = evaluate(model, dataloaders['dev'], criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Dev Loss:   {dev_loss:.4f} | Dev Acc:   {dev_acc:.4f}")
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), f"best_model_{dataset_name}_finetune.pt")
            print(f"✓ New best dev acc: {best_dev_acc:.4f}")
    
    # Test
    model.load_state_dict(torch.load(f"best_model_{dataset_name}_finetune.pt"))
    test_loss, test_acc = evaluate(model, dataloaders['test'], criterion, device)
    print(f"\n=== Final Results ===")
    print(f"Best Dev Acc: {best_dev_acc:.4f}")
    print(f"Test Acc:     {test_acc:.4f}")
    
    return best_dev_acc, test_acc
```

---

## Run Fine-Tuning

### SST Fine-Tuning
```python
sst_dev_ft, sst_test_ft = run_finetune_experiment(
    dataset_name="sst",
    batch_size=8,
    lr=5e-5,
    epochs=5
)
```

**Expected output:** ~0.52 (52%) dev accuracy
**Training time:** ~30-45 min on GPU

### CFIMDB Fine-Tuning
```python
cfimdb_dev_ft, cfimdb_test_ft = run_finetune_experiment(
    dataset_name="cfimdb",
    batch_size=4,    # Reviews are long, small batch
    lr=3e-5,
    epochs=3         # CFIMDB converges fast
)
```

**Expected output:** ~0.97 (97%) dev accuracy
**Training time:** ~45-60 min on GPU

---

## Memory Optimization Tips

If you run out of GPU memory:

### 1. Gradient Accumulation
Simulate larger batch by accumulating gradients over multiple steps:

```python
accumulation_steps = 4  # Effective batch = batch_size * accumulation_steps

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Gradient Checkpointing
Trades compute for memory — reduces memory by ~3x, adds ~30% time:

```python
model.gpt2.gradient_checkpointing_enable()
```

### 3. Mixed Precision (FP16)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

---

## Results Comparison

```python
# After running both experiments:
print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)
print(f"{'Dataset':<10} {'Approach':<15} {'Dev Acc':<10} {'Test Acc':<10}")
print("-"*60)
print(f"{'SST':<10} {'Frozen':<15} {0.451:<10.4f} {0.448:<10.4f}")  # replace with your actual
print(f"{'SST':<10} {'Fine-tuned':<15} {0.518:<10.4f} {0.512:<10.4f}")
print(f"{'CFIMDB':<10} {'Frozen':<15} {0.829:<10.4f} {0.825:<10.4f}")
print(f"{'CFIMDB':<10} {'Fine-tuned':<15} {0.976:<10.4f} {0.971:<10.4f}")
```
