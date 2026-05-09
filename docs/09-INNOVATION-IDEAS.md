# 09 — Innovation Ideas

This is the most important part of your project. **You must improve upon the baselines** with your own idea.

Below are 15+ concrete innovation ideas, organized by category.

---

## Category A: Pooling Strategies

Instead of taking just the last token's hidden state, aggregate all token representations.

### A1. Attention Pooling

Learn a weighted sum of all token representations.

```python
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, last_hidden, attention_mask):
        # last_hidden: (batch, seq_len, hidden_size)
        # attention_mask: (batch, seq_len)
        scores = self.attention(last_hidden).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding tokens
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax over sequence dimension
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        pooled = (last_hidden * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_size)
        return pooled
```

**Expected improvement**: +1-3% on SST

### A2. Multi-Layer Pooling

GPT-2 has 12 layers. Use representations from multiple layers instead of just the last.

```python
class MultiLayerPooling(nn.Module):
    def __init__(self, model, num_layers=4):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        hidden_size = model.config.hidden_size
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, input_ids, attention_mask):
        # Get hidden states from all layers
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # outputs.hidden_states is a tuple of (embedding, layer1, ..., layer12)
        all_hidden = outputs.hidden_states  # 13 items
        
        # Use last num_layers
        selected = torch.stack(all_hidden[-self.num_layers:])  # (num_layers, batch, seq, hidden)
        
        # Weighted combination
        weights = torch.softmax(self.weights, dim=0)  # (num_layers,)
        combined = (selected * weights.view(-1, 1, 1, 1)).sum(dim=0)  # (batch, seq, hidden)
        
        # Last token pooling on combined
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        sentence_repr = combined[batch_indices, last_token_idx]
        
        return sentence_repr
```

**Expected improvement**: +1-4% on SST

### A3. CLS-style Token Pooling

Add a special `[CLS]` token (like BERT) at the beginning and use its representation.

```python
# Add [CLS] token
tokenizer.add_special_tokens({'cls_token': '[CLS]'})
model.gpt2.resize_token_embeddings(len(tokenizer))

# In forward: just take the first token's hidden state
cls_repr = last_hidden[:, 0, :]  # (batch, hidden_size)
```

---

## Category B: Training Techniques

### B1. Gradual Unfreezing

Start with the classifier head only, then gradually unfreeze GPT-2 layers from top to bottom.

```python
def gradual_unfreeze(model, epoch, total_epochs):
    """
    Unfreeze layers progressively.
    epoch 0: only classifier
    epoch 1: classifier + last 4 GPT-2 layers
    epoch 2: classifier + last 8 GPT-2 layers
    epoch 3+: all layers
    """
    if epoch == 0:
        # Only classifier
        for param in model.gpt2.parameters():
            param.requires_grad = False
    elif epoch == 1:
        # Unfreeze last 4 layers (layers 8-11)
        for i, layer in enumerate(model.gpt2.h):
            layer.requires_grad = (i >= 8)
    elif epoch == 2:
        # Unfreeze last 8 layers (layers 4-11)
        for i, layer in enumerate(model.gpt2.h):
            layer.requires_grad = (i >= 4)
    else:
        # Unfreeze all
        for param in model.gpt2.parameters():
            param.requires_grad = True
```

**Expected improvement**: +1-2%, more stable training

### B2. Layer-wise Learning Rate Decay

Lower layers get smaller learning rates (they learn general features), higher layers get larger rates (task-specific).

```python
def get_layerwise_lr_params(model, base_lr=5e-5, decay=0.95):
    """
    Each GPT-2 layer gets LR = base_lr * decay^depth
    Layer 0 (embedding): lowest LR
    Layer 11: highest LR
    Classifier: base_lr
    """
    param_groups = []
    
    # Embedding layer
    param_groups.append({
        'params': model.gpt2.wte.parameters(),
        'lr': base_lr * (decay ** 12)
    })
    
    # Transformer layers (bottom to top)
    for i, layer in enumerate(model.gpt2.h):
        param_groups.append({
            'params': layer.parameters(),
            'lr': base_lr * (decay ** (11 - i))  # top layers get higher LR
        })
    
    # Final layer norm
    param_groups.append({
        'params': model.gpt2.ln_f.parameters(),
        'lr': base_lr
    })
    
    # Classifier head (highest LR)
    param_groups.append({
        'params': model.classifier.parameters(),
        'lr': base_lr
    })
    
    return param_groups
```

**Expected improvement**: +1-2%, more stable fine-tuning

### B3. Adapters (Parameter-Efficient Fine-Tuning)

Instead of fine-tuning all 124M parameters, insert small bottleneck layers and only train those.

```python
class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.activation(x)
        x = self.up(x)
        return x + residual  # residual connection

# Insert into each GPT-2 block (after the MLP)
def add_adapters(model, bottleneck_size=64):
    for layer in model.gpt2.h:
        adapter = BottleneckAdapter(model.gpt2.config.hidden_size, bottleneck_size)
        layer.adapter = adapter
        
        # Override forward to include adapter
        original_forward = layer.forward
        def adapter_forward(x, *args, **kwargs):
            x = original_forward(x, *args, **kwargs)
            x = layer.adapter(x)
            return x
        layer.forward = adapter_forward
    
    # Freeze GPT-2, only train adapters + classifier
    for param in model.gpt2.parameters():
        param.requires_grad = False
    for layer in model.gpt2.h:
        for param in layer.adapter.parameters():
            param.requires_grad = True
```

**Expected improvement**: +0.5-2%, much more parameter-efficient than full fine-tuning

---

## Category C: Data & Augmentation

### C1. Data Augmentation via Back-Translation

Translate sentences to French and back to English to create variations.

```python
# Using HuggingFace translation models
from transformers import pipeline

# Load translation pipeline (run once)
translator_en_to_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
translator_fr_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

def back_translate(text):
    fr = translator_en_to_fr(text)[0]['translation_text']
    en = translator_fr_to_en(fr)[0]['translation_text']
    return en

# Augment training data (be careful with time!)
# sst_df['text_aug'] = sst_df['text'].apply(back_translate)
```

**Expected improvement**: +1-3% (especially for small datasets)

### C2. Mixup Augmentation

Create virtual training examples by mixing two sentences and their labels.

```python
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# In training loop:
if use_mixup:
    lam = np.random.beta(0.5, 0.5)
    index = torch.randperm(batch_size).to(device)
    
    mixed_input_ids = lam * input_ids + (1 - lam) * input_ids[index]
    # Note: Mixup on token IDs is unusual. 
    # Better approach: mixup on hidden states or embeddings
```

**Expected improvement**: +1-2% (more robust model)

### C3. CFIMDB Subsampling for Fair Comparison

The original CS224N CFIMDB has only 1,701 training examples. To match the paper, create a smaller subset.

```python
# Create a small CFIMDB matching original size
small_cfimdb = train_df.sample(n=1701, random_state=42, stratify=train_df['label'])
```

---

## Category D: Multi-Task & Ensemble

### D1. Multi-Task Learning (SST + CFIMDB)

Train one model on both datasets simultaneously. The model shares GPT-2 and has two classification heads.

```python
class MultiTaskGPT2(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        hidden_size = self.gpt2.config.hidden_size
        
        self.dropout = nn.Dropout(0.1)
        self.sst_head = nn.Linear(hidden_size, 5)       # 5 classes
        self.cfimdb_head = nn.Linear(hidden_size, 2)     # 2 classes
    
    def forward(self, input_ids, attention_mask, task):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        sentence_repr = last_hidden[batch_indices, last_token_idx]
        
        sentence_repr = self.dropout(sentence_repr)
        
        if task == 'sst':
            return self.sst_head(sentence_repr)
        else:
            return self.cfimdb_head(sentence_repr)
```

**Expected improvement**: +0.5-2% (shared representations benefit both tasks)

### D2. Model Ensemble

Average predictions from multiple models.

```python
models = [
    GPT2Classifier(num_classes=5, freeze=False),
    GPT2Classifier(num_classes=5, freeze=False),
]
models = [m.to(device) for m in models]

# Load different checkpoints
models[0].load_state_dict(torch.load("best_model_sst_finetune_run1.pt"))
models[1].load_state_dict(torch.load("best_model_sst_finetune_run2.pt"))

# Ensemble prediction
all_logits = []
for m in models:
    m.eval()
    with torch.no_grad():
        logits = m(input_ids, attention_mask)
        all_logits.append(logits)

avg_logits = torch.stack(all_logits).mean(dim=0)
preds = torch.argmax(avg_logits, dim=1)
```

**Expected improvement**: +1-3% (but requires training multiple models)

---

## Category E: Advanced Architectures

### E1. BiLSTM on Top of GPT-2

Add a BiLSTM layer to process GPT-2's token representations before pooling.

```python
class GPT2BiLSTM(nn.Module):
    def __init__(self, model_name="gpt2", num_classes=5, lstm_hidden=256):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        hidden_size = self.gpt2.config.hidden_size
        
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True
        )
        
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq, 768)
        
        lstm_out, _ = self.bilstm(last_hidden)  # (batch, seq, 512)
        
        # Use last LSTM output
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        sentence_repr = lstm_out[batch_indices, last_token_idx]
        
        sentence_repr = self.dropout(sentence_repr)
        return self.classifier(sentence_repr)
```

**Expected improvement**: +1-3%

### E2. Hierarchical Attention

For long CFIMDB reviews, split into sentences, encode each, then attend over sentences.

```python
# Split review into sentences
sentences = review.split('. ')
# Encode each sentence separately with GPT-2
# Then use attention to combine sentence representations
```

**Expected improvement**: +1-2% on CFIMDB (handles long reviews better)

---

## Summary Table

| # | Innovation | Difficulty | Est. Improvement | Best For |
|---|-----------|-----------|-----------------|----------|
| A1 | Attention Pooling | Easy | +1-3% | SST |
| A2 | Multi-Layer Pooling | Medium | +1-4% | SST |
| A3 | CLS Token Pooling | Easy | +0-2% | Both |
| B1 | Gradual Unfreezing | Medium | +1-2% | Both |
| B2 | Layer-wise LR Decay | Medium | +1-2% | Both |
| B3 | Adapters (PEFT) | Hard | +0.5-2% | Both |
| C1 | Back-Translation Aug | Hard | +1-3% | SST |
| C2 | Mixup | Medium | +1-2% | Both |
| C3 | CFIMDB Subsampling | Easy | N/A (fairness) | CFIMDB |
| D1 | Multi-Task Learning | Hard | +0.5-2% | Both |
| D2 | Model Ensemble | Medium | +1-3% | Both |
| E1 | BiLSTM on Top | Medium | +1-3% | SST |
| E2 | Hierarchical Attention | Hard | +1-2% | CFIMDB |

**Recommendation**: Start with **A1 (Attention Pooling)** + **B1 (Gradual Unfreezing)** — both are relatively easy to implement and give clear improvements. If you want more impact, add **D1 (Multi-Task)** or **E1 (BiLSTM)**.
