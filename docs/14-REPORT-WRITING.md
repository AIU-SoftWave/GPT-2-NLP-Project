# 14 — Report Writing

How to write your 6-8 page NLP project report.

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

**Structure:**
1. **Problem**: One sentence on sentiment classification
2. **Approach**: What you did (GPT-2 + your innovation)
3. **Results**: Key numbers (SST: X%, CFIMDB: Y%)
4. **Conclusion**: One sentence on significance

**Example:**
> This paper presents a sentiment classification system using pretrained GPT-2 representations. We evaluate two baseline strategies: frozen feature extraction and full fine-tuning. We then propose [YOUR INNOVATION], which improves performance by X% on the Stanford Sentiment Treebank (SST) and Y% on the CFIMDB dataset. Our best model achieves Z% accuracy on SST and W% on CFIMDB, demonstrating the effectiveness of [YOUR INNOVATION] for sentiment analysis.

---

### 3. Introduction (~1 page)

**Structure:**
1. **Hook**: Why sentiment analysis matters (product reviews, social media)
2. **Background**: Pretrained language models (GPT-2, BERT)
3. **Gap**: What's missing / what this paper addresses
4. **Your contribution**: What you did differently
5. **Results preview**: Best numbers
6. **Roadmap**: "Section 2 covers related work..."

**Tip**: Cite 3-5 papers here (GPT-2 paper, BERT paper, sentiment analysis surveys)

---

### 4. Related Work (~0.5 page)

**Structure:**
1. **Traditional sentiment analysis**: BoW, SVM, LSTMs
2. **Pretrained language models**: GPT-2, BERT, RoBERTa
3. **Fine-tuning vs. feature extraction**: The debate
4. **Your innovation's related work**: Papers that inspired your approach

**Key papers to cite:**
- Radford et al., 2019 — GPT-2
- Devlin et al., 2019 — BERT
- Vaswani et al., 2017 — Transformers
- Howard & Ruder, 2018 — ULMFiT (gradual unfreezing)
- Houlsby et al., 2019 — Adapters

---

### 5. Approach (~1-2 pages)

**Structure:**
1. **Problem formulation**: Input → GPT-2 → classifier → output
2. **GPT-2 architecture**: Brief description, last-token pooling
3. **Baseline 1: Frozen features**: Diagram, training details
4. **Baseline 2: Fine-tuning**: Differences from frozen
5. **Your innovation**: Full description with:
   - Motivation (why this should work)
   - Architecture diagram
   - Key equations or pseudocode
   - Implementation details

**Include a model architecture diagram!**

Example diagram description (use tikz or similar):
```
Input Sentence
     ↓
GPT-2 Encoder (Frozen or Fine-tuned)
     ↓
Token Hidden States [h1, h2, ..., hn]
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

**6.1 Data**

| Dataset | Classes | Train | Dev | Test |
|---------|---------|-------|-----|------|
| SST | 5 (0-4) | 8,544 | 1,101 | 2,210 |
| CFIMDB | 2 (pos/neg) | [your count] | [your count] | [your count] |

Describe preprocessing:
- Tokenization: GPT-2 BPE tokenizer
- Max sequence length: 128 (SST), 512 (CFIMDB)
- Padding/truncation strategy

**6.2 Experimental Details**

| Hyperparameter | Frozen | Fine-tuned |
|---------------|--------|------------|
| Optimizer | AdamW | AdamW |
| Learning rate | 1e-3 | 5e-5 |
| Batch size | 16 | 8 |
| Epochs | 10 | 5 |
| Dropout | 0.1 | 0.1 |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping | None | 1.0 |
| LR scheduler | None | Linear warmup |

**6.3 Results**

| Model | SST Dev | SST Test | CFIMDB Dev | CFIMDB Test |
|-------|---------|----------|------------|-------------|
| Frozen (baseline) | 0.451 | 0.448 | 0.829 | 0.825 |
| Fine-tuned (baseline) | 0.518 | 0.512 | 0.976 | 0.971 |
| **Ours** | **0.542** | **0.535** | **0.982** | **0.979** |
| Improvement | +2.4% | +2.3% | +0.6% | +0.8% |

Include a **bar chart** comparing results.

---

### 7. Analysis (~1-2 pages)

**Structure:**

**7.1 Confusion Matrix Analysis**
- Which classes are confused? (e.g., SST: very negative ↔ negative)
- Include confusion matrix figure

**7.2 Error Analysis**
- Show 3-5 example errors
- Why did the model get them wrong?
  - Ambiguous sentences
  - Sarcasm
  - Length effects

**Example error table:**
| Text | True | Pred | Analysis |
|------|------|------|----------|
| "This movie is not bad" | 3 (positive) | 2 (neutral) | Negation confuses model |
| "A truly disappointing film" | 1 (negative) | 0 (very neg) | Intensity mismatch |

**7.3 Effect of Sentence Length**
- Plot accuracy vs. length
- Are longer sentences harder?

**7.4 Innovation Analysis**
- Ablation study table
- When does your innovation help most?
- Any failure cases?

---

### 8. Conclusion (~0.5 page)

**Structure:**
1. **Summary**: 2-3 sentences on what you did
2. **Key findings**: What worked, what didn't
3. **Limitations**: (e.g., only tested on 2 datasets, GPT-2 small only)
4. **Future work**: What would you try next?
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

## Writing Tips

1. **Be concise**: Every sentence should add value
2. **Be precise**: Use exact numbers, not vague terms ("significantly better" → "2.4% improvement")
3. **Use active voice**: "We trained the model..." not "The model was trained..."
4. **Label all figures/tables**: "Figure 1: Training curves" with caption
5. **Reference figures in text**: "As shown in Figure 1..."
6. **Use LaTeX math**: `$y = W h_T + b$`
7. **Proofread**: Read aloud at least once

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
