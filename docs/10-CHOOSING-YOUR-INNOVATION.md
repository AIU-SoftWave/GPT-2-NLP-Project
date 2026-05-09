# 10 — Choosing Your Innovation (Beginner-Friendly)

## What You'll Learn

- How to pick the right innovation for your team and situation
- Which innovations are best for different team sizes
- How to plan your time and GPU budget
- What the graders are looking for

---

## Step 1: Assess Your Situation

### Team Size

| Team Size | What to focus on |
|-----------|-----------------|
| **Solo** | Pick 1-2 easy/medium innovations (A1, B2, C3) |
| **2 people** | Pick 1 medium + 1 medium/hard (A1 + B1, or A2 + D2) |
| **3 people** | Pick 2-3 innovations across categories (A1 + B1 + D1) |

### GPU Budget

| GPU Time Available | Strategy |
|-------------------|----------|
| **Limited** (<2 hours total) | Frozen baseline + 1 easy innovation (A1, A3) |
| **Moderate** (2-5 hours) | Full baselines + 1-2 innovations (A1, B2) |
| **Generous** (5+ hours) | Full experiments + multi-task + ensemble |

### Target Grade

| Target Grade | What you need |
|-------------|---------------|
| **Pass** (60%) | Both baselines + basic innovation |
| **Good** (75%) | Both baselines + solid innovation + analysis |
| **Excellent** (90%+) | Strong innovation + ablation studies + thorough analysis |

---

## Step 2: Quick Decision Flowchart

```
Do you understand attention? (from doc 06)
    ├── YES → Try Attention Pooling (A1) — biggest bang for your buck
    │
    └── NO → Do you want to change the training or the model?
                │
                ├── TRAINING → Layer-wise LR Decay (B2) or Gradual Unfreezing (B1)
                │
                └── MODEL → CLS Token Pooling (A3) or BiLSTM on Top (E1)

After your first innovation, pick a second one from a DIFFERENT category
(e.g., A1 + B1, or A2 + B2)
```

---

## Step 3: Recommended Combinations

### Path A: "Safe and Solid" (Good grade, moderate effort)

| Innovation | Why | Effort |
|-----------|-----|--------|
| **A1 — Attention Pooling** | Easy, intuitive, clear improvement | 1 hour coding |
| **B1 — Gradual Unfreezing** | Standard technique, well-motivated | 1 hour coding |
| **B2 — Layer-wise LR Decay** | Easy addition to B1 | 30 min coding |

**Total effort:** ~3-4 hours coding + ~3-4 hours training

**Expected improvement:** +3-7% (combined)

### Path B: "Ambitious" (Excellent grade, high effort)

| Innovation | Why | Effort |
|-----------|-----|--------|
| **A2 — Multi-Layer Pooling** | Novel, shows deep understanding | 2 hours coding |
| **B3 — Adapters** | Current research direction | 3 hours coding |
| **D1 — Multi-Task Learning** | Impressive, leverages both datasets | 3 hours coding |

**Total effort:** ~6-8 hours coding + ~6-8 hours training

**Expected improvement:** +3-8% (combined)

### Path C: "Analysis-Focused" (Good grade, less coding)

| Innovation | Why | Effort |
|-----------|-----|--------|
| **C3 — CFIMDB Subsampling** | Compare with original paper | 15 min coding |
| **D2 — Model Ensemble** | Simple but effective | 1 hour coding |
| **Heavy analysis** | Confusion matrices, error analysis, attention viz | 3 hours analysis |

**Total effort:** ~2-3 hours coding + ~5 hours training + ~3 hours analysis

---

## Step 4: Understanding the Grading Rubric

| Criteria | Marks | How to get top marks |
|----------|-------|---------------------|
| **Innovation & Originality** | 4.5 | Novel idea, well-justified (cite papers!), clear why it should work |
| **Implementation & Experimentation** | 4.5 | Code works, reproducible, fair comparison to baselines |
| **Analysis & Insight** | 3.0 | Explain WHY your innovation helped or didn't. Show examples. |
| **Report Quality** | 1.5 | Clear writing, good figures, professional format |
| **Team Contributions** | 1.5 | Each member's role is clear |

### For maximum marks:

1. **Justify your innovation** — Cite 2-3 papers that inspired you
2. **Run ablation studies** — Test your innovation alone, then combined with others
3. **Show failure cases too** — When does your innovation NOT help?
4. **Visualize** — Attention maps, confusion matrices, accuracy curves

---

## Step 5: Innovation Checklist

Before finalizing your innovation, check:

- [ ] Is it different from both baselines?
- [ ] Can you explain WHY it should work (in one sentence)?
- [ ] Can you implement it within 1-2 days?
- [ ] Can you train it within your available GPU budget?
- [ ] Can you measure its impact clearly?
- [ ] Does it have a clear motivation (from papers)?

If you answered YES to all 6, you have a solid innovation plan.

---

## What to Include in the Report

For each innovation, discuss:

1. **Motivation**: What problem does it solve? (1 paragraph)
2. **Implementation**: How did you implement it? (key code, not everything)
3. **Results**: How much did it improve accuracy? (table with numbers)
4. **Analysis**: Why did it help? When did it fail? (examples)
5. **Ablation**: What happens if you remove part of it? (comparison)
