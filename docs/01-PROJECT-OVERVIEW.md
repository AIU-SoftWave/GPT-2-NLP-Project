# 01 — Project Overview (Beginner-Friendly)

## What Is This Project? — In Plain English

You are building a **sentiment classifier**: a program that reads a movie review and decides if it's positive or negative (and how strongly).

**Example:**
> "This movie was absolutely fantastic!" → **Positive (very positive)**
>
> "What a waste of time, terrible acting." → **Negative (very negative)**

The computer learns to do this by looking at thousands of examples — just like how you'd learn by reading many reviews.

---

## Why Is This Hard (For a Computer)?

A computer doesn't understand words the way humans do. It sees text as raw characters/bytes. We need to:

1. **Convert words to numbers** (tokenization) — so the computer can process them
2. **Give those numbers meaning** — using a pre-trained language model (GPT-2) that already "knows" English
3. **Train a classifier** on top — teach it to recognize sentiment patterns

---

## The Two Datasets

| Dataset | What it contains | How many classes | What the model predicts |
|---------|-----------------|------------------|------------------------|
| **SST** (Stanford Sentiment Treebank) | Short movie review sentences (~19 words each) | 5 | 0=very negative, 1=negative, 2=neutral, 3=positive, 4=very positive |
| **CFIMDB** (IMDB movie reviews) | Full movie reviews (~234 words each) | 2 | positive or negative |

**Which is harder?** SST is harder because there are 5 possible answers and the sentences are very short (less context). CFIMDB is easier because there are only 2 choices and more text to work with.

---

## The Model: GPT-2

**GPT-2** is a language model created by OpenAI. It was trained to predict the next word in a sentence (like autocomplete). By doing this on billions of sentences, it learned:

- Grammar and sentence structure
- Word meanings and relationships
- A general "understanding" of language

We're **reusing** GPT-2 for our sentiment task. This is called **transfer learning** — taking a model that knows one thing (language) and teaching it something related (sentiment).

> **Analogy:** Imagine hiring a chef who's an expert at Italian cooking. You don't need to teach them how to chop vegetables or use a stove. You just need to show them a few new recipes. That's what transfer learning is.

---

## Two Approaches You Must Implement

### 1. Frozen Feature Extractor (Baseline) — "The lazy approach"

- GPT-2's weights are **frozen** (locked, not updated during training)
- Only a tiny **linear classifier** on top is trained (just 3,845 parameters out of 124 million!)
- **Fast to train** (~5 minutes on GPU)
- **Lower accuracy** (~45% on SST, ~83% on CFIMDB)

> **Analogy:** You ask a concert pianist to play a song they've never heard. They can sight-read it okay, but it won't be their best performance.

### 2. Full Fine-Tuning — "The dedicated approach"

- ALL of GPT-2's weights are **updated** during training (124 million parameters)
- **Slower to train** (~30-60 minutes on GPU)
- **Higher accuracy** (~52% on SST, ~97% on CFIMDB)

> **Analogy:** The pianist practices the exact song for a week. Now they play it much better because they've adjusted to the specific piece.

---

## The Innovation Requirement (Your Original Idea)

You must come up with **your own improvement** beyond these two baselines. This is the creative part of the project.

**Some ideas (simple explanations):**

| Idea | How it works | Why it might help |
|------|-------------|-------------------|
| **Attention pooling** | Instead of only using the last word's hidden state, let the model learn which words are most important | A review might have "terrible" early on and "amazing" at the end — attention can weigh both |
| **Mean pooling** | Average ALL word hidden states instead of just the last one | Captures the "whole sentence" feeling instead of just the last word |
| **Adapter modules** | Insert small trainable "plug-in" layers inside GPT-2 | Like fine-tuning but with fewer parameters (faster, less memory) |
| **Ensemble** | Train 3 models and let them vote on the answer | Reduces mistakes — if 2 out of 3 agree, it's probably right |
| **Data augmentation** | Create fake variations of reviews (swap words with synonyms, back-translate) | More training data = better learning |

---

## Expected Results (from reference report)

| Model | SST Dev Accuracy | CFIMDB Dev Accuracy |
|-------|-----------------|---------------------|
| Frozen (last linear layer) | ~45.1% | ~82.9% |
| Fine-tuned | ~51.8% | ~97.6% |

Your innovation should try to beat these numbers.

> **What's "good enough"?** Getting ~52% on SST might not sound impressive (a random guess would get 20%), but SST is genuinely hard — even humans don't always agree on whether a review is "positive" or "very positive". For CFIMDB, 97% is already very high, so improvements there are harder to get.

---

## Grading Rubric (15 marks total)

| Section                          | Marks | What they're looking for                                                |
| -------------------------------- | ----- | ----------------------------------------------------------------------- |
| Innovation & Originality         | 4.5   | Is your idea new? Does it make sense? Does it improve results?          |
| Implementation & Experimentation | 4.5   | Does your code work? Did you compare fairly against baselines?          |
| Analysis & Insight               | 3.0   | Can you explain WHY your idea worked (or didn't)? Show graphs, examples |
| Report Quality                   | 1.5   | Is your report well-written and professional-looking?                   |
| Team Contributions               | 1.5   | Did you clearly describe who did what?                                  |

---

## Deliverables

1. **Report** — 6-8 page PDF using the provided LaTeX template
2. **Code** — ZIP file (<1MB, no datasets or model checkpoints)

---

## Notebook Structure (main.ipynb)

The notebook is organized into these sections:

```
Section 1:  Setup & Environment      → Install packages, check GPU, verify data
Section 2:  Load & Explore SST       → Read the sentence-level dataset
Section 3:  Load & Explore CFIMDB    → Read the review-level dataset
Section 4:  Tokenization             → Convert words to numbers GPT-2 can process
Section 5:  Build the Model          → Create GPT-2 + classifier head
Section 6:  Training Functions       → Write the training loop utilities
Section 7:  Frozen Baseline          → Run frozen GPT-2 on both datasets
Section 8:  Fine-Tuning              → Run full fine-tuning on both datasets
Section 9:  Results                  → Compare everything in a table
```

---

## File Locations

```
Project/
├── Datasets/
│   ├── SST2Data/                # Stanford Sentiment Treebank
│   └── CFIMDB/
│       └── IMDB Dataset.csv     # 50K IMDB reviews
├── docs/                        # These guides (step-by-step)
├── main.ipynb                   # Your main notebook (run this!)
├── project.md                   # Course assignment description
└── Sahaj_Saini_CS224N_Final_Report.md  # Reference report (example)
```
