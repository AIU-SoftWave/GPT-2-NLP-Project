# 10 — Choosing Your Innovation

A guide to selecting the right innovation for your project.

---

## Decision Framework

Ask yourself these questions:

### 1. What's the team size and skill distribution?

| Team Size | Recommended Strategy |
|-----------|---------------------|
| Solo | Pick 1-2 easy/medium innovations (A1, B2, C3) |
| 2 people | Pick 1 medium + 1 medium/hard (A1 + B1, or A2 + D2) |
| 3 people | Pick 2-3 innovations across categories (A1 + B1 + D1) |

### 2. How much GPU time do you have?

| GPU Budget | Strategy |
|-----------|----------|
| Limited (<2 hours total) | Frozen baseline + 1 easy innovation (A1, A3) |
| Moderate (2-5 hours) | Full baselines + 1-2 innovations (A1, B2) |
| Generous (5+ hours) | Full experiments + multi-task + ensemble |

### 3. What grade do you aim for?

| Target Grade | Required |
|-------------|----------|
| Pass (60%) | Both baselines + basic innovation |
| Good (75%) | Both baselines + solid innovation + analysis |
| Excellent (90%+) | Strong innovation + ablation studies + thorough analysis |

---

## Recommended Combinations

### Path A: "Safe and Solid" (Good grade, moderate effort)

1. **Attention Pooling (A1)** — Easy, clear improvement
2. **Gradual Unfreezing (B1)** — Standard technique, well-motivated
3. **Layer-wise LR Decay (B2)** — Easy addition to B1

**Total effort**: ~3-4 hours coding + ~3-4 hours training

### Path B: "Ambitious" (Excellent grade, high effort)

1. **Multi-Layer Pooling (A2)** — Novel, strong motivator
2. **Adapters (B3)** — Current research direction
3. **Multi-Task Learning (D1)** — Impressive, leverages both datasets

**Total effort**: ~6-8 hours coding + ~6-8 hours training

### Path C: "Analysis-Focused" (Good grade, less coding)

1. **CFIMDB Subsampling (C3)** — Compare with original paper
2. **Model Ensemble (D2)** — Simple but effective
3. **Heavy analysis** — Confusion matrices, error analysis, attention visualization

**Total effort**: ~2-3 hours coding + ~5 hours training + ~3 hours analysis

---

## Grading Impact

Remember the rubric:

| Criteria | Marks | How Your Innovation Affects It |
|----------|-------|-------------------------------|
| Innovation & Originality | 4.5 | The idea itself must be novel and well-justified |
| Implementation & Experimentation | 4.5 | Code must work, experiments must be reproducible |
| Analysis & Insight | 3.0 | You need to analyze WHY your innovation helped or didn't |
| Report Quality | 1.5 | Clear writing, good figures, professional formatting |
| Team Contributions | 1.5 | Each member has a clear role |

### For maximum marks:

1. **Justify your innovation** with citations (papers that inspired you)
2. **Run ablation studies** — try your innovation alone, then combined
3. **Show failure cases too** — when does your innovation NOT help?
4. **Visualize** — attention maps, confusion matrices, accuracy curves

---

## Innovation Checklist

Before finalizing your innovation, verify:

- [ ] Is it different from both baselines?
- [ ] Can you explain WHY it should work?
- [ ] Can you implement it within 1-2 days?
- [ ] Can you train it within available GPU budget?
- [ ] Can you measure its impact clearly?
- [ ] Does it have a clear motivation (from literature)?

---

## What to Include in the Report

For each innovation, you should discuss:

1. **Motivation**: What problem does it solve?
2. **Implementation**: How did you implement it? (with key code)
3. **Results**: How much did it improve accuracy?
4. **Analysis**: Why did it help? When did it fail?
5. **Ablation**: What happens if you remove part of it?
