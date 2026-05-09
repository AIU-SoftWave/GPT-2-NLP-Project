# 01 — Project Overview

## What Is This Project?

You are building a **sentiment classification system** using **GPT-2** (a pretrained language model). The goal is to classify movie review text into sentiment categories.

This is based on the **Stanford CS224N Default Project** (adapted for your AIE241 NLP course).

---

## The Problem

Given a sentence from a movie review, predict its sentiment:

| Dataset | Classes | Labels |
|---------|---------|--------|
| **SST** (Stanford Sentiment Treebank) | 5 classes | 0=very negative, 1=negative, 2=neutral, 3=positive, 4=very positive |
| **CFIMDB** / IMDB | 2 classes | positive, negative |

---

## Two Approaches You Must Implement

### 1. Frozen Feature Extractor (Baseline)
- GPT-2 weights are **frozen** (not updated during training)
- Only a small **linear classifier** on top is trained
- Lower accuracy, faster to train, less GPU memory

### 2. Full Fine-Tuning
- All GPT-2 weights are **updated** during training
- Higher accuracy, slower, needs more GPU memory

---

## The Innovation Requirement

You must propose and implement **your own improvement** beyond these two baselines. Ideas include:

- Different pooling strategies (attention pooling, multi-layer aggregation)
- Adapter modules (lightweight trainable layers inserted into GPT-2)
- Gradual unfreezing (slowly unfreeze layers during training)
- Multi-task learning (SST + CFIMDB jointly)
- Data augmentation (back-translation, mixup)
- Ensemble methods

---

## Expected Results (from reference report)

| Model | SST Dev Accuracy | CFIMDB Dev Accuracy |
|-------|-----------------|---------------------|
| Frozen (last linear layer) | ~45.1% | ~82.9% |
| Fine-tuned | ~51.8% | ~97.6% |

Your innovation should improve on these numbers.

---

## Grading Rubric (15 marks total)

| Section | Marks | What They Look For |
|---------|-------|--------------------|
| Innovation & Originality | 4.5 | Novelty, justification, meaningful improvement |
| Implementation & Experimentation | 4.5 | Correct code, fair comparison, reproducible results |
| Analysis & Insight | 3.0 | Error analysis, visualizations, qualitative examples |
| Report Quality | 1.5 | Clear writing, professional format, proper figures |
| Team Contributions | 1.5 | Each member's role clearly described |

---

## Deliverables

1. **Report** — 6-8 page PDF (use the provided LaTeX template)
2. **Code** — ZIP file (<1MB, no datasets or model checkpoints)

---

## Project Structure (After Setup)

```
Project/
├── Datasets/
│   ├── SST2Data/                # Stanford Sentiment Treebank
│   └── CFIMDB/
│       └── IMDB Dataset.csv     # 50K IMDB reviews
├── docs/                        # These guides
├── main.ipynb                   # Your notebook
├── project.md                   # Course assignment description
└── Sahaj_Saini_CS224N_Final_Report.md  # Reference report
```
