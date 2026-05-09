# 14 — Report Writing (Beginner-Friendly)

## What You'll Learn

- How to write your 6-8 page NLP project report
- What to put in each section
- Key writing tips for beginners
- How to use LaTeX (briefly)

---

## Key Terms (Defined Simply)

| Term | Simple definition | Analogy |
|------|------------------|---------|
| **Abstract** | A short summary of the entire paper (what you did, what you found) | A movie trailer — gives you the highlights without spoiling everything |
| **BibTeX** | A system for managing references/citations | A digital recipe box where you store all your recipe sources |
| **LaTeX** | A document preparation system (not Word) | Like coding your document instead of typing it |
| **Ablation** | Testing what happens when you remove a component | Baking without sugar to see if it's still good |

---

## Template

Use the provided LaTeX template. If you don't have it, ask your instructor.

**Key formatting rules:**
- **6-8 pages** (excluding references)
- PDF format
- Named: `TeamName_ProjectReport.pdf`
- Use BibTeX for references
- Write as an NLP research paper

---

## Section-by-Section Guide

### 1. Key Information

```
Title: Sentiment Classification using GPT-2 with [Your Innovation]

Team Members:
- Name 1 (email1)
- Name 2 (email2)
- Name 3 (email3)

Original CS224N Project: Default Final Project — Sentiment Classification
```

---

### 2. Abstract (<300 words)

**The abstract is the most important paragraph of your paper.** Many people will only read this.

**Structure (4 sentences):**
1. **Problem**: One sentence on sentiment classification
2. **Approach**: What you did (GPT-2 + your innovation)
3. **Results**: Key numbers (SST: X%, CFIMDB: Y%)
4. **Conclusion**: One sentence on significance

**Example:**
> This paper presents a sentiment classification system using pretrained GPT-2 representations. We evaluate two baseline strategies: frozen feature extraction and full fine-tuning. We then propose [YOUR INNOVATION], which improves performance by X% on the Stanford Sentiment Treebank (SST) and Y% on the CFIMDB dataset. Our best model achieves Z% accuracy on SST and W% on CFIMDB, demonstrating the effectiveness of [YOUR INNOVATION] for sentiment analysis.

---

### 3. Introduction (~1 page)

**What to include:**

1. **Hook** (3-4 sentences): Why does sentiment analysis matter?
   - Product reviews, social media monitoring, customer feedback
   - Billions of text items generated daily — automating analysis is valuable

2. **Background** (3-4 sentences): What are pretrained language models?
   - GPT-2, BERT — trained on massive text data
   - Transfer learning: adapt them to new tasks

3. **Gap** (1-2 sentences): What's missing?
   - Standard approaches have limitations
   - You propose to address one of those

4. **Your contribution** (2-3 sentences): What you did differently
   - Briefly describe your innovation (save details for Approach section)

5. **Results preview** (1 sentence): Best numbers

6. **Roadmap** (1 sentence): "Section 2 covers related work..."

**Tip:** Cite 3-5 papers here (GPT-2 paper, BERT paper, sentiment analysis surveys)

---

### 4. Related Work (~0.5 page)

**What to include:**

1. **Traditional sentiment analysis** (2-3 sentences):
   - Bag of Words, SVM, LSTMs
   - These required careful feature engineering

2. **Pretrained language models** (2-3 sentences):
   - GPT-2, BERT, RoBERTa
   - These learn general language representations

3. **Fine-tuning vs. feature extraction** (1-2 sentences):
   - The debate in the literature
   - Each has trade-offs

4. **Your innovation's related work** (2-3 sentences):
   - Papers that inspired your approach
   - How your work builds on or differs from theirs

**Key papers to cite:**
- Radford et al., 2019 — GPT-2
- Devlin et al., 2019 — BERT
- Vaswani et al., 2017 — Transformers (attention is all you need)
- Howard & Ruder, 2018 — ULMFiT (gradual unfreezing)
- Houlsby et al., 2019 — Adapters

---

### 5. Approach (~1-2 pages)

**What to include:**

1. **Problem formulation** (short paragraph):
   - Input → GPT-2 → classifier → output
   - Define the task mathematically

2. **GPT-2 architecture** (1-2 paragraphs):
   - Brief description
   - Last-token pooling (why we use it)
   - Diagram recommended

3. **Baseline 1: Frozen features**:
   - Diagram showing what's frozen vs trained
   - Training details

4. **Baseline 2: Fine-tuning**:
   - Differences from frozen
   - Why different hyperparameters needed

5. **Your innovation** (most detailed):
   - **Motivation**: Why this should work (2-3 sentences)
   - **Architecture diagram**: A picture is worth 1000 words
   - **Key equations or pseudocode**: The core idea
   - **Implementation details**: Any tricky parts

**Example diagram:**
```
Input Sentence
     ↓
GPT-2 Encoder (Frozen or Fine-tuned)
     ↓
Token Hidden States [h₁, h₂, ..., hₙ]
     ↓
[Your Innovation] ← e.g., Attention Pooling
     ↓
Sentence Representation
     ↓
Dropout → Linear → Softmax
     ↓
Predicted Sentiment
```

---

### 6. Experiments (~1-2 pages)

#### 6.1 Data

| Dataset | Classes | Train | Dev | Test |
|---------|---------|-------|-----|------|
| SST | 5 (0-4) | 8,544 | 1,101 | 2,210 |
| CFIMDB | 2 (pos/neg) | 40,000 | 5,000 | 5,000 |

Describe preprocessing:
- Tokenization: GPT-2 BPE tokenizer
- Max sequence length: 128 (SST), 512 (CFIMDB)
- Padding/truncation strategy

#### 6.2 Experimental Details

| Hyperparameter | Frozen | Fine-tuned |
|---------------|--------|------------|
| Optimizer | AdamW | AdamW |
| Learning rate | 3e-3 | 5e-5 |
| Batch size | 16 | 8 |
| Epochs | 10 | 5 |
| Dropout | 0.1 | 0.1 |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping | None | 1.0 |
| LR scheduler | None | Linear warmup |
| GPU | [Your GPU] | [Your GPU] |
| Training time | ~5-10 min | ~30-60 min |

#### 6.3 Results

| Model | SST Dev | SST Test | CFIMDB Dev | CFIMDB Test |
|-------|---------|----------|------------|-------------|
| Frozen (baseline) | 0.451 | 0.448 | 0.829 | 0.825 |
| Fine-tuned (baseline) | 0.518 | 0.512 | 0.976 | 0.971 |
| **Ours** | **0.542** | **0.535** | **0.982** | **0.979** |
| Improvement | +2.4% | +2.3% | +0.6% | +0.8% |

Include a **bar chart** comparing results.

---

### 7. Analysis (~1-2 pages)

#### 7.1 Confusion Matrix Analysis
- Which classes are confused? (e.g., SST: very negative ↔ negative)
- Include confusion matrix figure
- **Why it matters:** Shows the model isn't making random mistakes — it's confused between similar classes

#### 7.2 Error Analysis
- Show 3-5 example errors with analysis
- **Why it matters:** Numbers don't tell the whole story. Examples help readers understand the model's limitations

**Example error table:**
| Text | True | Pred | Analysis |
|------|------|------|----------|
| "This movie is not bad" | 3 (positive) | 2 (neutral) | Negation confuses model — "not bad" is positive, but model sees "bad" as negative |
| "A truly disappointing film" | 1 (negative) | 0 (very neg) | Intensity mismatch — model overestimates the negativity |

#### 7.3 Effect of Sentence Length
- Plot accuracy vs. length
- Are longer sentences harder?

#### 7.4 Innovation Analysis
- Ablation study table (what happens when you remove parts of your innovation?)
- When does your innovation help most?
- Any failure cases?

---

### 8. Conclusion (~0.5 page)

**Structure:**
1. **Summary** (2-3 sentences): What you did
2. **Key findings** (2-3 sentences): What worked, what didn't
3. **Limitations** (1-2 sentences): Be honest about weaknesses
   - Only tested on 2 datasets
   - Only used GPT-2 small
4. **Future work** (2-3 sentences): What would you try next?
   - Larger models (GPT-2 XL, GPT-3)
   - Other pooling strategies
   - More datasets

**Example:**
> We presented a sentiment classification system using GPT-2 with [INNOVATION]. Our approach achieves X% on SST and Y% on CFIMDB, improving over baselines by Z%. Analysis shows that [INNOVATION] particularly helps with [specific case]. Future work could explore applying our method to larger pretrained models and additional classification tasks.

---

### 9. Team Contributions

| Member | Contribution |
|--------|-------------|
| Name 1 | Implemented baselines, data preprocessing, ran SST experiments |
| Name 2 | Designed and implemented [innovation], hyperparameter tuning |
| Name 3 | Error analysis, visualizations, report writing |

---

### 10. References (BibTeX)

```
@article{radford2019gpt2,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}

@article{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={NAACL},
  year={2019}
}

@inproceedings{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={NeurIPS},
  year={2017}
}
```

---

## Writing Tips for Beginners

1. **Write first, edit later** — Don't try to make it perfect as you write. Just get ideas down, then refine.

2. **One idea per paragraph** — If a paragraph has more than one main idea, split it.

3. **Use active voice** — "We trained the model..." not "The model was trained..." Active voice is clearer and more direct.

4. **Be precise** — Use exact numbers ("2.4% improvement"), not vague terms ("significantly better").

5. **Label all figures/tables** — "Figure 1: Training curves" with a caption underneath. Reference them in text: "As shown in Figure 1..."

6. **Keep it simple** — If you can explain something in plain English, do that. Don't use jargon to sound smart.

7. **Proofread** — Read your paper aloud at least once. You'll catch awkward phrasing.

8. **Ask a friend to read it** — If they can understand it, you're doing it right.

---

## Submission Checklist

- [ ] PDF is 6-8 pages (excluding references)
- [ ] All sections present (1-10)
- [ ] At least 3 figures (training curves, confusion matrix, results comparison)
- [ ] At least 2 tables (hyperparameters, results)
- [ ] All citations in BibTeX format
- [ ] File named `TeamName_ProjectReport.pdf`
- [ ] Code ZIP named `TeamName_Code.zip`
- [ ] ZIP does NOT include datasets or model checkpoints
- [ ] ZIP < 1MB
- [ ] All team members tagged in submission
