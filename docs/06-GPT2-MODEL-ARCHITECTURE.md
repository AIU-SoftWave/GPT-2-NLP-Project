# 06 — GPT-2 Model Architecture

## What Is GPT-2?

GPT-2 (Generative Pretrained Transformer 2) is a **decoder-only transformer** language model. It's trained to predict the next word in a sequence.

**Key facts:**
- Architecture: Decoder-only transformer
- Training objective: Next token prediction (autoregressive)
- Context length: 1024 tokens
- Available sizes: small (124M), medium (355M), large (774M), XL (1.5B)
- For this project: **GPT-2 small** (124M parameters) is sufficient

---

## How GPT-2 Encodes Text

When you pass a sentence through GPT-2:

```
Input:  "The movie was great"
           ↓
      [tokenizer]
           ↓
Tokens:  [464, 3181, 373, 3734]
           ↓
      [GPT-2 model]
           ↓
Hidden states for each token:
  h_0 = vector for "The"     (768-dim for GPT-2 small)
  h_1 = vector for "movie"   (768-dim)
  h_2 = vector for "was"     (768-dim)
  h_3 = vector for "great"   (768-dim)
```

Each hidden state is a **contextual representation** — it encodes information from all preceding tokens (because GPT-2 is causal/left-to-right).

---

## Last-Token Pooling (The Standard Approach)

For classification, GPT-2 models typically use the **last token's hidden state** as the sentence representation.

Why the last token? In a causal language model, the last token has "seen" all previous tokens, so its representation contains information about the entire sequence.

```
Tokens:  [The] [movie] [was] [great]
            ↓      ↓      ↓      ↓
States:   h_0    h_1    h_2    h_3
                                 └── Use this for classification
```

---

## Model Code

```python
from transformers import GPT2Model, GPT2Config
import torch.nn as nn

class GPT2Classifier(nn.Module):
    def __init__(self, model_name="gpt2", num_classes=5, dropout=0.1, freeze=True):
        super().__init__()
        
        # Load pretrained GPT-2
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.hidden_size = self.gpt2.config.hidden_size  # 768 for gpt2
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Freeze GPT-2 parameters if needed
        if freeze:
            for param in self.gpt2.parameters():
                param.requires_grad = False
            print("GPT-2 frozen: Only classifier head will be trained")
        else:
            print("GPT-2 unfrozen: Full fine-tuning")
    
    def forward(self, input_ids, attention_mask):
        # Get GPT-2 outputs
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden = outputs.last_hidden_state
        
        # Take the last non-padding token for each sequence
        # attention_mask: (batch_size, seq_len) — 1 for real tokens, 0 for padding
        last_token_idx = attention_mask.sum(dim=1) - 1  # index of last real token
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        
        # Gather the last token's hidden state
        sentence_repr = last_hidden[batch_indices, last_token_idx]  # (batch_size, hidden_size)
        
        # Classify
        sentence_repr = self.dropout(sentence_repr)
        logits = self.classifier(sentence_repr)  # (batch_size, num_classes)
        
        return logits
```

---

## Architecture Diagram

```
Input IDs:        [464, 3181, 373, 3734, 50256, 50256, ...]
                    │      │     │     │      │       │
Attention Mask:    [ 1,     1,    1,    1,     0,      0, ...]
                    │      │     │     │      │       │
                    └──────┴─────┴─────┴──────┴───────┘
                                      │
                              GPT-2 Model
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  last_hidden_state (batch, seq, 768) │
                    └─────────────────┬─────────────────┘
                                      │
                          Pick last real token
                          (sum(attention_mask) - 1)
                                      │
                            sentence_repr (batch, 768)
                                      │
                                Dropout + Linear
                                      │
                              logits (batch, num_classes)
```

---

## Testing the Model

```python
# Test with frozen GPT-2
model_frozen = GPT2Classifier(model_name="gpt2", num_classes=5, freeze=True)

# Create dummy input
dummy_input = torch.randint(0, 50000, (2, 10))  # (batch=2, seq_len=10)
dummy_mask = torch.ones(2, 10)

# Forward pass
with torch.no_grad():
    logits = model_frozen(dummy_input, dummy_mask)

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {logits.shape}")  # (2, 5) — 2 samples, 5 classes
print(f"Output (logits):\n{logits}")

# Count trainable parameters
trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_frozen.parameters())
print(f"\nTrainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
```

**Expected output:**
```
GPT-2 frozen: Only classifier head will be trained
Input shape: torch.Size([2, 10])
Output shape: torch.Size([2, 5])
Output (logits):
tensor([[..., ..., ..., ..., ...],
        [..., ..., ..., ..., ...]])

Trainable params: 3,845 / 124,439,808 (0.003%)
```

When frozen, only the classifier head (~3.8K params) is trainable out of 124M total.

---

## Fine-Tuning Mode

```python
# Test with unfrozen GPT-2
model_unfrozen = GPT2Classifier(model_name="gpt2", num_classes=5, freeze=False)

trainable = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_unfrozen.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
```

**Expected output:**
```
GPT-2 unfrozen: Full fine-tuning
Trainable params: 124,439,808 / 124,439,808 (100.00%)
```

All 124M parameters are trainable. This is why fine-tuning uses more GPU memory.

---

## Alternative Pooling Strategies (for your innovation)

Instead of last-token pooling, you could try:

1. **Mean pooling**: Average all token hidden states
2. **Max pooling**: Take element-wise max across tokens
3. **Weighted pooling**: Learnable attention weights over tokens
4. **Multi-layer pooling**: Combine hidden states from multiple GPT-2 layers

```python
def mean_pooling(last_hidden, attention_mask):
    """Average all non-padding token representations."""
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

def max_pooling(last_hidden, attention_mask):
    """Max over all non-padding token representations."""
    mask = attention_mask.unsqueeze(-1).float()
    mask[mask == 0] = -1e9  # Set padding to very negative
    return (last_hidden + mask).max(dim=1).values
```
