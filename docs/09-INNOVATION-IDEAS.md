# 09 — Innovation Ideas (Beginner-Friendly)

## What You'll Learn

- 15+ ways to improve upon the baselines (your original contribution!)
- Why each idea might help (the intuition)
- How to implement each one (with code)

---

## Key Terms (Defined Simply)

| Term | Simple definition | Analogy |
|------|------------------|---------|
| **Pooling** | Combining multiple token representations into one | Summarizing a book into one paragraph |
| **Attention** | Learning which tokens are more important | A spotlight that shines brighter on important words |
| **Adapter** | Small trainable modules inserted into a frozen model | Plug-in modules for a synthesizer that add new sounds |
| **Ensemble** | Combining predictions from multiple models | Asking 3 doctors for a diagnosis and going with the majority |
| **Ablation** | Testing what happens when you remove a component | Baking a cake without sugar to see how important sugar is |
| **Augmentation** | Creating modified versions of training data | Practicing a sport in different weather conditions |

---

## Category A: Pooling Strategies

### The Problem with Last-Token Pooling

Our baseline uses the **last token's hidden state** as the sentence representation. This works, but it might miss important information in the middle of the sentence.

> **Analogy:** Last-token pooling is like judging a movie based on the last 5 minutes only. You get some context, but you miss the build-up, the plot twists, and the character development.

---

### A1. Attention Pooling (Recommended for Beginners)

**The intuition:** Instead of using only the last token, calculate a **weighted average** of ALL tokens. The model learns which tokens are more important for sentiment.

> **Analogy:** A spotlight operator at a theater. The operator (the attention mechanism) learns to shine the spotlight on the most important actors (words) and dim it on less important ones.

**How it works:**
1. For each token, calculate an "importance score"
2. Mask out padding tokens (make their score very negative)
3. Convert scores to probabilities (softmax)
4. Multiply each token's hidden state by its probability
5. Sum them up → weighted average

```python
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # One small neural network layer that learns importance scores
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, last_hidden, attention_mask):
        # last_hidden: (batch, seq_len, hidden_size)
        # attention_mask: (batch, seq_len)
        
        # Step 1: Calculate importance score for each token
        scores = self.attention(last_hidden).squeeze(-1)  # (batch, seq_len)
        
        # Step 2: Mask padding tokens (make them -infinity so softmax ignores them)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Step 3: Convert scores to probabilities
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Step 4: Weighted sum of all token representations
        pooled = (last_hidden * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_size)
        return pooled
```

**Expected improvement:** +1-3% on SST

**Difficulty:** ⭐ Easy (modify the pooling method, that's it)

---

### A2. Multi-Layer Pooling (Medium)

**The intuition:** GPT-2 has 12 layers. Early layers capture grammar, middle layers capture phrases, and late layers capture overall meaning. Maybe combining multiple layers gives a richer representation than just the last one.

> **Analogy:** When describing a person, you might consider their appearance (early layer), their personality (middle layer), and their reputation (late layer). Each gives useful information that the others miss.

```python
class MultiLayerPooling(nn.Module):
    def __init__(self, model, num_layers=4):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        hidden_size = model.config.hidden_size
        # Learnable weights — the model decides which layers matter most
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, input_ids, attention_mask):
        # Get hidden states from ALL layers
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True  # This gives us all layers
        )
        
        # outputs.hidden_states is a tuple: (embedding, layer1, ..., layer12)
        all_hidden = outputs.hidden_states  # 13 items
        
        # Take the last num_layers (4, in this case)
        selected = torch.stack(all_hidden[-self.num_layers:])  # (4, batch, seq, 768)
        
        # Weighted combination of layers
        weights = torch.softmax(self.weights, dim=0)  # (4,)
        combined = (selected * weights.view(-1, 1, 1, 1)).sum(dim=0)  # (batch, seq, 768)
        
        # Last token pooling on combined representation
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        sentence_repr = combined[batch_indices, last_token_idx]
        
        return sentence_repr
```

**Expected improvement:** +1-4% on SST

**Difficulty:** ⭐⭐ Medium (requires understanding GPT-2's layer structure)

---

### A3. CLS-Style Token Pooling (Easy)

**The intuition:** BERT (another famous model) uses a special `[CLS]` token at the beginning of every sequence. The model is trained to put the sentence meaning in this token's representation. We can do the same thing with GPT-2.

> **Analogy:** A dedicated mailbox at your house where all important mail is delivered. Instead of searching through all the mail, you just check the mailbox.

```python
# Add [CLS] token to vocabulary
tokenizer.add_special_tokens({'cls_token': '[CLS]'})

# Resize model to accommodate the new token
model.gpt2.resize_token_embeddings(len(tokenizer))

# In forward: just take the first token's hidden state
cls_repr = last_hidden[:, 0, :]  # (batch, hidden_size) — first token = [CLS]
```

**Expected improvement:** +0-2% on both

**Difficulty:** ⭐ Easy (just change which token you pool from)

---

## Category B: Training Techniques

### B1. Gradual Unfreezing (Medium)

**The intuition:** Start with the model frozen (only classifier trains), then gradually unfreeze layers from the top down. This is gentler than full fine-tuning and prevents catastrophic forgetting.

> **Analogy:** When learning a new sport, you don't change your entire technique at once. First you fix your stance, then your grip, then your swing. Gradual unfreezing does the same for the model.

**How it works:**
- Epoch 0: Only classifier head trains
- Epoch 1: Classifier + last 4 GPT-2 layers
- Epoch 2: Classifier + last 8 GPT-2 layers
- Epoch 3+: All layers

```python
def gradual_unfreeze(model, epoch, total_epochs):
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

**Expected improvement:** +1-2%, more stable training

**Difficulty:** ⭐⭐ Medium (need to understand GPT-2's layer numbering)

---

### B2. Layer-wise Learning Rate Decay (Medium)

**The intuition:** Different layers of GPT-2 learn different things. Early layers (basic grammar) should change less than late layers (task-specific meaning). So we give early layers a smaller learning rate.

> **Analogy:** When renovating a house, you make small changes to the foundation but big changes to the paint color. Early layers = foundation, late layers = paint.

```python
def get_layerwise_lr_params(model, base_lr=5e-5, decay=0.95):
    param_groups = []
    
    # Embedding layer — lowest LR
    param_groups.append({
        'params': model.gpt2.wte.parameters(),
        'lr': base_lr * (decay ** 12)
    })
    
    # Transformer layers (bottom to top, increasing LR)
    for i, layer in enumerate(model.gpt2.h):
        param_groups.append({
            'params': layer.parameters(),
            'lr': base_lr * (decay ** (11 - i))  # top layers = higher LR
        })
    
    # Final layer norm
    param_groups.append({
        'params': model.gpt2.ln_f.parameters(),
        'lr': base_lr
    })
    
    # Classifier head — highest LR
    param_groups.append({
        'params': model.classifier.parameters(),
        'lr': base_lr
    })
    
    return param_groups
```

**Expected improvement:** +1-2%, more stable fine-tuning

**Difficulty:** ⭐⭐ Medium (need to understand GPT-2's module names)

---

### B3. Adapters (Parameter-Efficient Fine-Tuning) (Hard)

**The intuition:** Instead of fine-tuning all 124M parameters, insert small "bottleneck" layers (adapters) into each GPT-2 layer. Only train these adapters (a few thousand parameters each) + the classifier.

> **Analogy:** Instead of rebuilding an entire car engine to make it faster, you add a turbocharger — a small add-on that makes a big difference.

**Why this is exciting:** Adapters give you close to fine-tuning performance but with only ~5-10% of the trainable parameters. This means:
- Faster training
- Less GPU memory
- Less risk of overfitting

```python
class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        # Compress from 768 → 64 → 768
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x              # Save original
        x = self.down(x)          # Compress (768 → 64)
        x = self.activation(x)    # Non-linearity
        x = self.up(x)            # Expand back (64 → 768)
        return x + residual       # Add original back (residual connection)

# Insert adapter into each GPT-2 layer
def add_adapters(model, bottleneck_size=64):
    for layer in model.gpt2.h:
        adapter = BottleneckAdapter(model.gpt2.config.hidden_size, bottleneck_size)
        layer.adapter = adapter
        
        # Wrap the layer's forward method to include the adapter
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

**Expected improvement:** +0.5-2% (but with far fewer trainable params than full fine-tuning)

**Difficulty:** ⭐⭐⭐ Hard (need to modify GPT-2's internal structure)

---

## Category C: Data & Augmentation

### C1. Data Augmentation via Back-Translation (Hard)

**The intuition:** Create new training examples by translating sentences to another language and back. The meaning stays the same, but the wording changes — giving the model more variety to learn from.

> **Analogy:** Telling a story to a friend, who tells it to another friend, who tells it back to you. The core story is the same, but the words change slightly. This helps you understand the story better because you hear it in multiple versions.

```python
from transformers import pipeline

# Load translation models (run once, takes a minute)
translator_en_to_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
translator_fr_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

def back_translate(text):
    fr = translator_en_to_fr(text)[0]['translation_text']
    en = translator_fr_to_en(fr)[0]['translation_text']
    return en

# Example
original = "This movie was absolutely fantastic!"
augmented = back_translate(original)
# Might become: "This film was truly wonderful!"
```

**Expected improvement:** +1-3% (especially for small datasets like SST)

**Difficulty:** ⭐⭐⭐ Hard (requires downloading translation models, slow)

---

### C2. Mixup Augmentation (Medium)

**The intuition:** Create "hybrid" training examples by mixing two sentences and their labels. This makes the model more robust — it learns to handle ambiguous cases better.

> **Analogy:** If you study both "the cat sat" and "the dog ran," mixup creates "the cat ran" — helping you understand that either animal can perform either action.

```python
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss = weighted combination of both labels"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# In training loop:
lam = np.random.beta(0.5, 0.5)  # Random mixing weight
index = torch.randperm(batch_size).to(device)

# Mix embeddings (not token IDs — that wouldn't make sense)
mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]
```

**Expected improvement:** +1-2% (more robust model)

**Difficulty:** ⭐⭐ Medium (need to modify training loop)

---

### C3. CFIMDB Subsampling for Fair Comparison (Easy)

**The intuition:** The original CS224N project used only 1,701 training examples for CFIMDB. If you want to compare your results directly to the paper, create a similarly-sized subset.

```python
# Create a small CFIMDB matching original paper size
small_cfimdb = train_df.sample(n=1701, random_state=42, stratify=train_df['label'])
```

**Expected improvement:** N/A (this is for fair comparison, not improvement)

**Difficulty:** ⭐ Very easy

---

## Category D: Multi-Task & Ensemble

### D1. Multi-Task Learning (Hard)

**The intuition:** Train ONE model on BOTH datasets simultaneously. GPT-2 is shared, but there are two separate classification heads (one for SST with 5 outputs, one for CFIMDB with 2 outputs).

> **Analogy:** Learning two related subjects at the same time (e.g., Spanish and Italian). They're different, but what you learn in one helps with the other.

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

**Expected improvement:** +0.5-2% (shared representations benefit both tasks)

**Difficulty:** ⭐⭐⭐ Hard (complex training loop with alternating tasks)

---

### D2. Model Ensemble (Medium)

**The intuition:** Train 3 separate models (same architecture, different random seeds) and average their predictions. The "wisdom of the crowd" — if 2 out of 3 models agree, it's probably right.

> **Analogy:** When doctors disagree, a second (or third) opinion is often more reliable. Each model makes different mistakes, and averaging cancels them out.

```python
models = [
    GPT2Classifier(num_classes=5, freeze=False),
    GPT2Classifier(num_classes=5, freeze=False),
]
models = [m.to(device) for m in models]

# Train each model separately (different random seeds)...

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

**Expected improvement:** +1-3% (but requires training multiple models)

**Difficulty:** ⭐⭐ Medium (straightforward but expensive)

---

## Category E: Advanced Architectures

### E1. BiLSTM on Top of GPT-2 (Medium)

**The intuition:** Add a BiLSTM (Bidirectional LSTM) layer on top of GPT-2's token representations. The BiLSTM processes tokens in BOTH directions, capturing more context than GPT-2's left-to-right only approach.

> **Analogy:** GPT-2 reads a sentence left to right (like reading a book). A BiLSTM reads it left to right AND right to left — like reading the book forwards and backwards to fully understand the plot.

```python
class GPT2BiLSTM(nn.Module):
    def __init__(self, model_name="gpt2", num_classes=5, lstm_hidden=256):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        hidden_size = self.gpt2.config.hidden_size
        
        self.bilstm = nn.LSTM(
            input_size=hidden_size,    # 768 from GPT-2
            hidden_size=lstm_hidden,    # 256
            bidirectional=True,          # Left-to-right AND right-to-left
            batch_first=True
        )
        
        # 2 directions × 256 = 512
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

**Expected improvement:** +1-3%

**Difficulty:** ⭐⭐ Medium (need to understand LSTMs)

---

### E2. Hierarchical Attention for Long Reviews (Hard)

**The intuition:** For long CFIMDB reviews, split them into sentences. Encode each sentence separately with GPT-2, then use attention to figure out which sentences matter most.

> **Analogy:** When summarizing a long book, you first read each chapter (sentence), then decide which chapters are most important (attention), and write your summary based on the important chapters.

**When it helps:** CFIMDB reviews can be very long (up to 2,470 words). GPT-2's 1024-token limit forces us to truncate. Hierarchical attention avoids this by processing each sentence separately.

**Expected improvement:** +1-2% on CFIMDB

**Difficulty:** ⭐⭐⭐ Hard (complex architecture, custom data preparation)

---

## Summary Table

| # | Innovation | Difficulty | Est. Improvement | Best For |
|---|-----------|-----------|-----------------|----------|
| **A1** | Attention Pooling | ⭐ Easy | +1-3% | SST |
| **A2** | Multi-Layer Pooling | ⭐⭐ Medium | +1-4% | SST |
| **A3** | CLS Token Pooling | ⭐ Easy | +0-2% | Both |
| **B1** | Gradual Unfreezing | ⭐⭐ Medium | +1-2% | Both |
| **B2** | Layer-wise LR Decay | ⭐⭐ Medium | +1-2% | Both |
| **B3** | Adapters (PEFT) | ⭐⭐⭐ Hard | +0.5-2% | Both |
| **C1** | Back-Translation Aug | ⭐⭐⭐ Hard | +1-3% | SST |
| **C2** | Mixup | ⭐⭐ Medium | +1-2% | Both |
| **C3** | CFIMDB Subsampling | ⭐ Easy | N/A | CFIMDB |
| **D1** | Multi-Task Learning | ⭐⭐⭐ Hard | +0.5-2% | Both |
| **D2** | Model Ensemble | ⭐⭐ Medium | +1-3% | Both |
| **E1** | BiLSTM on Top | ⭐⭐ Medium | +1-3% | SST |
| **E2** | Hierarchical Attention | ⭐⭐⭐ Hard | +1-2% | CFIMDB |

## Recommendation

If you're new to ML/AI, start with:

1. **A1 (Attention Pooling)** — Easy to implement, clear intuition, gives +1-3%
2. **B1 (Gradual Unfreezing)** — Well-known technique, works reliably
3. **B2 (Layer-wise LR Decay)** — Easy addition to B1

These three together can give +3-7% improvement and show you understand multiple aspects of the model (architecture + training).
