# 08 — Fine-Tuning (Beginner-Friendly)

## What You'll Learn

- What fine-tuning means and why it's different from frozen
- Why we need different settings (lower learning rate, gradient clipping, scheduler)
- How to train the fine-tuned baseline
- Expected results and how they compare to frozen

---

## Key Terms (Defined Simply)

| Term | Simple definition | Analogy |
|------|------------------|---------|
| **Fine-tuning** | Updating ALL model parameters during training (not just the classifier) | A chef who already knows how to cook Italian cuisine now practices specifically making lasagna |
| **Learning rate scheduler** | Gradually reduces the learning rate over time | Taking smaller steps as you get closer to your destination |
| **Gradient clipping** | Caps the maximum size of gradients to prevent wild updates | A seatbelt that prevents you from flying forward when the car stops suddenly |
| **Weight decay** | Adds a small penalty for large parameter values | Keeping your backpack organized — prevents it from getting too messy |
| **Warmup** | Starts with a very small learning rate and gradually increases it | Warming up before exercise instead of sprinting immediately |

---

## Key Differences: Frozen vs Fine-Tuning

| Aspect | Frozen | Fine-Tuning |
|--------|--------|-------------|
| **Trainable params** | ~3,845 (0.003%) | 124 million (100%) |
| **Learning rate** | 3e-3 (higher is fine) | 5e-5 (much smaller!) |
| **Batch size** | 32-64 | 8-16 (smaller due to memory) |
| **Training time (SST)** | ~1-2 min | ~15-20 min |
| **Memory usage** | ~2 GB | ~6-10 GB |
| **Dev accuracy (SST)** | ~45% | ~52% |
| **Dev accuracy (CFIMDB)** | ~83% | ~97% |

### Why are the numbers so different?

Fine-tuning updates **all 124 million parameters**. This is powerful but dangerous — like performing surgery instead of just putting on a bandage. You need to be much more careful.

---

## The Model

Same architecture, but with `freeze=False`:

```python
model = GPT2Classifier(
    model_name="gpt2",
    num_classes=5,     # 5 for SST, 2 for CFIMDB
    dropout=0.1,
    freeze=False       # ← This is the key difference!
)
```

---

## Why Fine-Tuning Needs Special Care

### 1. Smaller Learning Rate

**The problem:** GPT-2 already "knows" English really well. If we update its parameters too aggressively, we can **destroy** what it's learned — like scribbling over a painting instead of adding to it.

**The solution:** Use a much smaller learning rate (5e-5 instead of 1e-3).

> **Analogy:** Imagine adjusting the focus on a camera. With a frozen model, you're turning the focus knob a full turn. With fine-tuning, you're making tiny micro-adjustments — just a millimeter at a time. Any larger and you'd blur the image.

```python
# Good learning rates for fine-tuning
lr = 5e-5   # Try: 5e-5, 3e-5, 2e-5, 1e-5

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
```

**Note:** `weight_decay=0.01` adds a small penalty for large parameter values. This prevents overfitting by keeping the model's parameters "modest" — think of it as a preference for simpler explanations.

### 2. Learning Rate Scheduler

**The problem:** At the start of fine-tuning, the model needs to explore — try different adjustments to find what works. But near the end, it should fine-tune — make tiny corrections.

**The solution:** A scheduler that:
1. **Warms up** (gradually increases LR from 0 to 5e-5 over the first 10% of steps)
2. **Decays** (gradually decreases LR back to 0 over the remaining 90% of steps)

> **Analogy:** When learning to throw darts:
> - First, you experiment with different angles and forces (high LR)
> - Once you're close to the bullseye, you make tiny adjustments (low LR)
> - A scheduler automates this — high LR early, decreasing over time

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

```
Learning Rate
    ^
    |     __________________
    |    /                  \
    |   /                    \
    |  /                      \
    | /                        \
    +-----------------------------> Steps
     ↑ warmup (10%)   ↑ decay (90%)
```

### 3. Gradient Clipping

**The problem:** Sometimes the gradients (which tell us how to update parameters) can become **explosively large** — like a microphone feedback loop. This can ruin the model in one step.

**The solution:** Clip gradients to a maximum size. If the gradient tries to exceed the limit, it gets scaled down.

> **Analogy:** Gradient clipping is like a **seatbelt**. Most of the time you don't need it, but when something goes wrong, it prevents disaster.

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
        scheduler.step()  # Update learning rate
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy
```

Compare to the frozen training loop:
- **New:** `clip_grad_norm_()` — prevents gradient explosion
- **New:** `scheduler.step()` — adjusts learning rate after each batch
- **Same:** Everything else

```python
def run_finetune_experiment(dataset_name="sst", batch_size=16, lr=5e-5, epochs=5):
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
    batch_size=16,     # Balanced for GPU memory vs training speed
    lr=5e-5,           # Much smaller than frozen (3e-3 → 5e-5)
    epochs=5           # Fewer epochs than frozen (10→5) because faster convergence
)

**Expected output:** ~0.52 (52%) dev accuracy

**Training time:** ~15-20 min on GPU (FP16 enabled)

**Why only 5 epochs?** Fine-tuning converges faster because all 124M parameters are being optimized simultaneously. More epochs might cause overfitting.

### CFIMDB Fine-Tuning

```python
cfimdb_dev_ft, cfimdb_test_ft = run_finetune_experiment(
    dataset_name="cfimdb",
    batch_size=8,     # Reviews are long, need small batch for memory
    lr=5e-5,           # Standard fine-tuning LR
    epochs=3           # CFIMDB is easy, converges very fast
)

**Expected output:** ~0.97 (97%) dev accuracy

**Training time:** ~20-30 min on GPU (FP16 + gradient accumulation)

---

## Expected Results

| Model | SST Dev | CFIMDB Dev |
|-------|---------|------------|
| Frozen (baseline) | ~45.1% | ~82.9% |
| Fine-tuned | ~51.8% | ~97.6% |
| **Improvement** | **+6.7%** | **+14.7%** |

### Why is CFIMDB improvement so dramatic?

1. **CFIMDB is simpler** — binary classification (2 choices vs 5)
2. **Longer text** — more context for the model to work with
3. **Fine-tuning unlocks the full model** — GPT-2 can learn subtle patterns in long reviews that the frozen classifier couldn't access
4. **More data** — 40K examples is enough to fine-tune effectively

### Why is SST improvement smaller?

1. **Harder task** — 5 classes, subtle differences between adjacent classes
2. **Shorter text** — less context for the model to work with
3. **Limited data** — 8.5K examples is decent but not huge

---

## Memory Optimization (If You Run Out of GPU Memory)

Fine-tuning all 124M parameters requires a lot of GPU memory (6-10 GB). The main.ipynb notebook uses FP16 mixed precision and gradient accumulation by default to fit comfortably in 8 GB. If your GPU doesn't have enough:

### 1. Gradient Accumulation

Simulate a larger batch by accumulating gradients over multiple steps:

```python
accumulation_steps = 4  # Effective batch = batch_size × accumulation_steps

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

> **Analogy:** Instead of carrying 16 boxes at once (which doesn't fit), carry 4 boxes 4 times. The effect is the same, just spread out.

### 2. Gradient Checkpointing

Saves memory by recomputing activations during backward pass instead of storing them:

```python
model.gpt2.gradient_checkpointing_enable()
```

This reduces memory by ~3x but adds ~30% training time. A trade-off: **time for memory**.

### 3. Mixed Precision (FP16) — Enabled by Default

Use 16-bit floating point numbers instead of 32-bit — halves memory usage and speeds up training on RTX 4060 Tensor Cores:

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

After running both experiments, compare:

```python
print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)
print(f"{'Dataset':<10} {'Approach':<15} {'Dev Acc':<10} {'Test Acc':<10}")
print("-"*60)
print(f"{'SST':<10} {'Frozen':<15} {0.451:<10.4f} {0.448:<10.4f}")
print(f"{'SST':<10} {'Fine-tuned':<15} {0.518:<10.4f} {0.512:<10.4f}")
print(f"{'CFIMDB':<10} {'Frozen':<15} {0.829:<10.4f} {0.825:<10.4f}")
print(f"{'CFIMDB':<10} {'Fine-tuned':<15} {0.976:<10.4f} {0.971:<10.4f}")
```

---

## Summary

| What | Frozen | Fine-Tuning |
|------|--------|-------------|
| What's trained? | Classifier only (3,845 params) | Everything (124M params) |
| Learning rate | 3e-3 | 5e-5 |
| Batch size | 32-64 | 8-16 |
| Training time | ~1-5 min | ~15-30 min |
| GPU memory | ~2 GB | ~6-10 GB |
| Risk of overfitting | Very low | Higher (need to monitor) |
| SST accuracy | ~45% | ~52% |
| CFIMDB accuracy | ~83% | ~97% |

**Key takeaway:** Fine-tuning gives much better results but requires careful tuning (lower LR, scheduler, gradient clipping) and more resources.
