# 06 — GPT-2 Model Architecture (Beginner-Friendly)

## What You'll Learn

- What GPT-2 is and how it works (at a high level)
- What "hidden states" and "pooling" mean
- How we turn GPT-2 into a sentiment classifier
- The difference between frozen and fine-tuning modes
- What "parameters" are and why 124 million matters

---

## Key Terms (Defined Simply)

| Term | Simple definition | Analogy |
|------|------------------|---------|
| **Model** | A mathematical function that maps input to output | A vending machine — put in coins (input), get a snack (output) |
| **Neural network** | A type of model inspired by the brain's network of neurons | A team of workers passing information from person to person |
| **Transformer** | A specific type of neural network architecture | An assembly line where every worker can see what ALL previous workers did |
| **Hidden state** | A vector (list of numbers) that represents what the model understands about a token | A sticky note summarizing everything the model has read so far |
| **Vector** | A list of numbers (e.g., 768 numbers in a row) | GPS coordinates — a point in space (but with 768 dimensions instead of 3) |
| **Pooling** | Combining multiple values into one summary | A movie critic summarizing a 2-hour film in one sentence |
| **Parameters** | The "knobs" of the model that get adjusted during training | The dials on a combination lock — turn them to the right values to unlock the answer |
| **Embedding** | Converting a token ID into a vector the model can work with | Looking up a word in a dictionary to find its meaning |

---

## 1. What Is GPT-2?

**GPT-2** (Generative Pretrained Transformer 2) is a language model created by OpenAI in 2019.

### What was it trained to do?

GPT-2 was trained on **millions of web pages** to do one simple thing: **predict the next word in a sentence**.

> "The cat sat on the ___" → GPT-2 predicts "mat" (or "chair" or "floor")

By doing this billions of times across billions of sentences, GPT-2 learned:
- **Grammar**: How sentences are structured
- **Vocabulary**: What words mean and how they relate
- **Context**: How the meaning of a word changes based on surrounding words
- **Facts**: Basic knowledge about the world (actors, movies, science, etc.)

### Why are we using it?

Instead of training a model from scratch (which would take weeks and require millions of examples), we **reuse** GPT-2's knowledge. This is called **transfer learning** — taking a model that knows one thing (language) and adapting it to a related task (sentiment analysis).

> **Analogy:** Imagine you need a chef who can cook Italian food. Instead of teaching someone from scratch (how to hold a knife, how to boil water, etc.), you hire an experienced chef who already knows all the basics. You just teach them a few new recipes. That's what we're doing with GPT-2.

---

## 2. How GPT-2 Processes Text

### Step by step:

```
Input:  "The movie was great"
           ↓
    [Tokenizer: converts words to numbers]
           ↓
Tokens:  [464, 3181, 373, 3734]
           ↓
    [GPT-2 Model: processes through 12 layers]
           ↓
Hidden states for each token:
  h₀ = vector for "The"     (768 numbers)
  h₁ = vector for "movie"   (768 numbers)
  h₂ = vector for "was"     (768 numbers)
  h₃ = vector for "great"   (768 numbers)
```

### What's a "hidden state"?

A **hidden state** is a vector of 768 numbers that represents what GPT-2 understands about a token **in its current context**.

> **Analogy:** Imagine each token gets a "sticky note." As GPT-2 reads from left to right, it updates the sticky note on each token to include information from all the tokens before it. By the time it reaches the word "great," the sticky note for "great" also contains information about "The movie was" — so it knows "great" is being used to describe a movie.

### Important: Causal (Left-to-Right) Processing

GPT-2 reads from left to right. Each token can only "see" tokens that came before it. This is called **causal attention** or **autoregressive** processing.

This matters because:
- "The movie was great" — the word "great" can see "The movie was" and understands the full context
- The word "The" can only see itself (no context)
- The last token has the **most information**

---

## 3. The GPT-2 Architecture (Simplified)

GPT-2 small has:
- **12 transformer layers** (stacked on top of each other)
- **12 attention heads** per layer (each paying attention to different things)
- **768 hidden size** (each hidden state is a vector of 768 numbers)
- **124 million parameters** (the "knobs" that can be adjusted)

### The layers do different things:

| Layer depth | What it learns | Analogy |
|-------------|---------------|---------|
| **Early layers** (1-4) | Basic grammar, word types (nouns, verbs) | Recognizing letters and words |
| **Middle layers** (5-8) | Phrase meanings, simple relationships | Understanding short phrases |
| **Late layers** (9-12) | Complex meaning, overall sentence intent | Understanding the big picture |

This is important for one of the innovations (multi-layer pooling) — different layers capture different levels of understanding.

---

## 4. Last-Token Pooling (The Standard Approach)

### What is pooling?

**Pooling** is the process of taking multiple values and combining them into one summary.

For sentiment classification, we need to go from "12 tokens, each with 768 numbers" to "1 vector of 768 numbers" that represents the **whole sentence's sentiment**.

### Why the last token?

In a causal (left-to-right) model like GPT-2, the last token's hidden state has "seen" all the previous tokens. So it contains information about the entire sentence.

```
Tokens:  [The] [movie] [was] [great]
            ↓      ↓      ↓      ↓
States:   h₀    h₁     h₂     h₃
                                 └── Use this one for classification
                                 (it "knows" about all previous words)
```

> **Analogy:** Imagine reading a sentence one word at a time, and each time you read a new word you write a summary. Your summary for the last word includes everything you've read. That's what last-token pooling does.

### How we implement it:

```python
# After getting hidden states from GPT-2
last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)

# Find the last NON-PADDING token for each example
last_token_idx = attention_mask.sum(dim=1) - 1
batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)

# Grab the last token's hidden state
sentence_repr = last_hidden[batch_indices, last_token_idx]  # (batch, 768)
```

**Why the attention mask trick?** We padded all sentences to the same length, so the last token might be padding (0). We use the attention mask to find the last REAL token.

---

## 5. The Full Classifier Model

Now we add a **classification head** on top of GPT-2:

```python
class GPT2Classifier(nn.Module):
    def __init__(self, model_name="gpt2", num_classes=5, dropout=0.1, freeze=True):
        super().__init__()
        
        # Load pretrained GPT-2
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        hidden_size = self.gpt2.config.hidden_size  # 768 for gpt2
        
        # Classification head (the part we add)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Freeze GPT-2 if requested
        if freeze:
            for param in self.gpt2.parameters():
                param.requires_grad = False
            print("GPT-2 frozen: Only classifier head will be trained")
        else:
            print("GPT-2 unfrozen: Full fine-tuning")
    
    def forward(self, input_ids, attention_mask):
        # Step 1: Get GPT-2's hidden states
        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # Step 2: Last-token pooling
        last_token_idx = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        sentence_repr = last_hidden[batch_indices, last_token_idx]  # (batch, 768)
        
        # Step 3: Classify
        sentence_repr = self.dropout(sentence_repr)
        logits = self.classifier(sentence_repr)  # (batch, num_classes)
        
        return logits
```

### The full pipeline:

```
Input tokens → GPT-2 Model → Hidden states → Last-token pooling → Dropout → Linear → Class scores
                  ↑                                               ↑              ↑
        (Frozen or Fine-tuned?)                       (Prevents        (Learns which 768-number
                                                        overfitting)     patterns map to each sentiment)
```

### What is the "Dropout" layer?

**Dropout** randomly turns off some neurons during training. This might sound counterproductive, but it helps prevent **overfitting** — when the model memorizes the training data instead of learning general patterns.

> **Analogy:** If a student only ever practices with a calculator, they'll fail when the calculator is taken away. Dropout is like occasionally taking away the calculator so the student learns to do math in their head.

---

## 6. Architecture Diagram (Visual)

```
Input IDs:        [464, 3181, 373, 3734, 50256, 50256, ...]
                    │      │     │     │      │       │
Attention Mask:    [ 1,     1,    1,    1,     0,      0, ...]
                    │      │     │     │      │       │
                    └──────┴─────┴─────┴──────┴───────┘
                                      │
                              GPT-2 Model
                              (12 layers)
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
                                Dropout (0.1)
                                      │
                                Linear Layer
                            (768 inputs → 5 outputs)
                                      │
                              logits (batch, 5)
```

The **logits** are raw scores for each class. The largest score = the model's prediction. For example:
```
[2.1, 1.3, -0.5, 0.8, -1.2]  →  Class 0 (very negative) has highest score → prediction = "very negative"
```

We convert logits to probabilities using **softmax**, but that happens inside the loss function during training.

---

## 7. Frozen vs Fine-Tuning

### What does "frozen" mean?

When we **freeze** GPT-2, we lock all 124 million parameters. They don't change during training. Only the classifier head (3,845 parameters) is trained.

### What does "fine-tuning" mean?

When we **fine-tune**, ALL 124 million parameters are updated during training. This gives better accuracy but requires more memory and takes longer.

### Why would you ever freeze?

| Reason | Explanation |
|--------|-------------|
| **Speed** | Training 3,845 params is much faster than 124 million |
| **Less memory** | Frozen model uses ~2 GB GPU memory vs ~6-10 GB for fine-tuning |
| **Small data** | If you have very little data, fine-tuning can overfit (memorize) |

### Parameter comparison:

```python
# Count parameters in frozen mode
model_frozen = GPT2Classifier(model_name="gpt2", num_classes=5, freeze=True)
trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_frozen.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
```

**Output:**
```
GPT-2 frozen: Only classifier head will be trained
Trainable params: 3,845 / 124,439,808 (0.003%)
```

Only **3,845** parameters trainable out of **124,439,808** — that's 0.003%!

### What are "parameters" exactly?

Parameters are the **numbers that define the model's behavior**. Think of them as the **dials and knobs** on a complex machine.

- **GPT-2 starts** with the dials set to values that make it good at predicting the next word
- **When we freeze**, we keep those dials exactly as they are and only add a few new dials (the classifier)
- **When we fine-tune**, we adjust ALL the dials to make the model good at sentiment analysis instead

> **Analogy:** A frozen GPT-2 is like hiring a talented musician and just telling them what song to play (they already know how to play). Fine-tuning is like sending the musician to a special workshop to learn a new instrument — they'll be even better at that specific instrument, but it takes more time and effort.

---

## 8. Alternative Pooling Strategies (for Your Innovation)

Last-token pooling is the simplest approach. But there are other ways to combine the hidden states:

### Mean Pooling
Average ALL token hidden states instead of using just the last one. This gives equal weight to every word.

```python
def mean_pooling(last_hidden, attention_mask):
    """Average all non-padding token representations."""
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
```

**When it helps:** If important sentiment words appear at the beginning of a sentence, mean pooling captures them better than last-token pooling.

### Max Pooling
Take the maximum value across all tokens for each dimension. This captures the "strongest signal."

```python
def max_pooling(last_hidden, attention_mask):
    """Max over all non-padding token representations."""
    mask = attention_mask.unsqueeze(-1).float()
    mask[mask == 0] = -1e9  # Set padding to very negative
    return (last_hidden + mask).max(dim=1).values
```

### Attention Pooling (Advanced)
Learn which tokens are most important for sentiment and weight them accordingly. This is one of the innovations you might implement later.

---

## Summary

- **GPT-2** is a pre-trained language model (124M parameters) that knows English
- **Hidden states** are vectors (768 numbers) representing each token in context
- **Last-token pooling** uses the last token's hidden state as the sentence representation
- **Frozen mode** trains only 3,845 parameters (fast, lower accuracy)
- **Fine-tuning mode** trains all 124M parameters (slow, higher accuracy)
- **Classification head** = Dropout + Linear layer that turns 768 numbers into class scores
