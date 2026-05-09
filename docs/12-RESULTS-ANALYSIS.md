# 12 — Results Analysis (Beginner-Friendly)

## What You'll Learn

- How to analyze your model's performance beyond just "accuracy"
- What a confusion matrix tells you
- How to find and analyze specific errors
- How to check if your improvement is statistically significant

---

## Key Terms (Defined Simply)

| Term | Simple definition | Analogy |
|------|------------------|---------|
| **Confusion matrix** | A table showing how often the model predicts each class vs the true class | A teacher's gradebook showing which questions students got right and wrong, and what wrong answers they gave |
| **Precision** | Of all the times the model said "positive," how many were actually positive? | "When I predict it'll rain, how often am I right?" |
| **Recall** | Of all the actually positive examples, how many did the model catch? | "Of all the rainy days, how many did I predict?" |
| **F1 score** | The harmonic mean of precision and recall | A balanced score that penalizes being too conservative or too aggressive |
| **p-value** | The probability that your improvement happened by chance | If p < 0.05, the improvement is "statistically significant" (real, not luck) |

---

## 1. Compare Your Results to the Reference

First, see how your numbers stack up against the expected results:

```python
# Your results
my_results = {
    'sst_frozen': {'dev': 0.451, 'test': 0.448},
    'sst_finetune': {'dev': 0.518, 'test': 0.512},
    'cfimdb_frozen': {'dev': 0.829, 'test': 0.825},
    'cfimdb_finetune': {'dev': 0.976, 'test': 0.971},
}

# Reference paper results
ref_results = {
    'sst_frozen': {'dev': 0.451},
    'sst_finetune': {'dev': 0.518},
    'cfimdb_frozen': {'dev': 0.829},
    'cfimdb_finetune': {'dev': 0.976},
}

print(f"{'Experiment':<20} {'You (dev)':<12} {'Paper (dev)':<12} {'Diff':<10}")
print("-"*54)
for exp in my_results:
    my = my_results[exp]['dev']
    ref = ref_results[exp]['dev']
    diff = my - ref
    print(f"{exp:<20} {my:<12.4f} {ref:<12.4f} {diff:<+10.4f}")
```

---

## 2. Confusion Matrix (SST — 5 Classes)

### What is a confusion matrix?

It's a grid that shows:
- **Rows**: True labels (what the correct answer actually was)
- **Columns**: Predicted labels (what the model guessed)
- **Diagonal**: Correct predictions (the model got it right)
- **Off-diagonal**: Errors (the model got it wrong)

> **Analogy:** Imagine a test with 5 possible grades (A, B, C, D, F). A confusion matrix shows: "Of the students who deserved an A, how many got A, how many got B, how many got C, etc."

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(model, dataloader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    if class_names is None:
        class_names = ['Very Neg', 'Neg', 'Neutral', 'Pos', 'Very Pos']
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        correct = cm[i, i]
        total = cm[i, :].sum()
        print(f"  {name:15s}: {correct}/{total} = {correct/total*100:.1f}%")
    
    return cm
```

### How to read the output:

```
Per-class accuracy:
  Very Neg      : 85/200 = 42.5%    ← Hardest (most confused with "Neg")
  Neg           : 120/250 = 48.0%   
  Neutral       : 200/350 = 57.1%   ← Easiest
  Pos           : 180/320 = 56.2%   
  Very Pos      : 90/200 = 45.0%    
```

**What this tells you:**
- Neutral and Positive are the easiest (the model has most data for these)
- Very Negative and Very Positive are hardest (the extremes are harder to distinguish)
- Most errors are between **adjacent classes** (predicting 3 instead of 4, or 1 instead of 0)
- This is expected! Even humans sometimes disagree on whether a review is "positive" vs "very positive"

---

## 3. Per-Class Analysis for CFIMDB (Binary)

For binary classification, we look at precision, recall, and F1:

```python
from sklearn.metrics import classification_report

def analyze_binary_results(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(classification_report(all_labels, all_preds, 
          target_names=['Negative', 'Positive']))
```

**Expected output (fine-tuned CFIMDB):**
```
              precision    recall  f1-score   support
    Negative       0.98      0.97      0.97      5000
    Positive       0.97      0.98      0.97      5000
```

Both classes are near-perfect for the fine-tuned model.

---

## 4. Error Analysis

### Why do error analysis?

Numbers don't tell the full story. Looking at actual examples where the model was wrong helps you understand:
- **Patterns**: Does the model fail on sarcastic sentences? Very short sentences? Sentences with negation?
- **Data issues**: Are some labels wrong? (It happens!)
- **Model limitations**: What kinds of texts does the model struggle with?

```python
def show_errors(model, dataloader, device, tokenizer, num_examples=10):
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            # Find errors
            mask = preds != labels
            for i in mask.nonzero(as_tuple=True)[0]:
                idx = i.item()
                
                # Decode input (remove padding)
                tokens = input_ids[idx][attention_mask[idx].bool()]
                text = tokenizer.decode(tokens)
                
                errors.append({
                    'text': text,
                    'true': labels[idx].item(),
                    'pred': preds[idx].item()
                })
                
                if len(errors) >= num_examples:
                    break
            
            if len(errors) >= num_examples:
                break
    
    print(f"\n=== {len(errors)} Error Examples ===\n")
    for i, err in enumerate(errors, 1):
        print(f"{i}. True: {err['true']} | Pred: {err['pred']}")
        print(f"   Text: {err['text'][:150]}...")
        print()
```

**Look for patterns in the errors:**
- Are errors mostly between adjacent classes? (Predicting 3 instead of 4)
- Are longer or shorter sentences harder?
- Does the model struggle with negation? ("not bad" means positive, but model might predict negative)
- Does the model handle sarcasm? ("Great, another boring movie")

---

## 5. Accuracy vs. Sentence Length

Check if the model performs worse on very short or very long sentences:

```python
def accuracy_by_length(model, df, dataloader, tokenizer, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_lengths = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            lengths = attention_mask.sum(dim=1).cpu()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_lengths.extend(lengths.numpy())
    
    df_result = pd.DataFrame({
        'length': all_lengths,
        'correct': np.array(all_preds) == np.array(all_labels)
    })
    
    df_result['bucket'] = pd.cut(df_result['length'], 
                                  bins=[0, 10, 20, 30, 50, 100, 200, 1000],
                                  labels=['0-10', '10-20', '20-30', '30-50', '50-100', '100-200', '200+'])
    
    accuracy_by_len = df_result.groupby('bucket')['correct'].mean()
    
    print("\nAccuracy by sentence length:")
    print(accuracy_by_len)
    
    accuracy_by_len.plot(kind='bar')
    plt.title('Accuracy by Sentence Length')
    plt.xlabel('Length (tokens)')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
```

**What to look for:** If accuracy drops on very long sentences, your model might be truncating important information.

---

## 6. Innovation Impact Analysis

Compare your innovation to the baseline:

```python
def compare_innovations(results_dict):
    """
    results_dict: {
        'baseline': {'dev': 0.518, 'test': 0.512},
        'attention_pool': {'dev': 0.534, 'test': 0.528},
        'gradual_unfreeze': {'dev': 0.526, 'test': 0.520},
        'combined': {'dev': 0.542, 'test': 0.535},
    }
    """
    baseline_dev = results_dict['baseline']['dev']
    baseline_test = results_dict['baseline']['test']
    
    print(f"{'Method':<20} {'Dev Acc':<10} {'Test Acc':<10} {'Δ Dev':<10} {'Δ Test':<10}")
    print("-"*60)
    
    for method, scores in results_dict.items():
        delta_dev = scores['dev'] - baseline_dev
        delta_test = scores['test'] - baseline_test
        arrow = "▲" if delta_dev > 0 else "▼" if delta_dev < 0 else " "
        print(f"{method:<20} {scores['dev']:<10.4f} {scores['test']:<10.4f} "
              f"{arrow}{delta_dev:<+9.4f} {arrow}{delta_test:<+9.4f}")
```

---

## 7. Statistical Significance (Optional)

### What is statistical significance?

If your innovation gives +1%, how do you know it's a REAL improvement and not just luck? Statistical tests help answer this.

**McNemar's test** is designed for comparing two classifiers on the same test set:

```python
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(model1_preds, model2_preds, true_labels):
    """
    Test if model2 is significantly better than model1.
    """
    # Build contingency table
    a = sum((m1 == t) & (m2 == t) for m1, m2, t in zip(model1_preds, model2_preds, true_labels))
    b = sum((m1 == t) & (m2 != t) for m1, m2, t in zip(model1_preds, model2_preds, true_labels))
    c = sum((m1 != t) & (m2 == t) for m1, m2, t in zip(model1_preds, model2_preds, true_labels))
    d = sum((m1 != t) & (m2 != t) for m1, m2, t in zip(model1_preds, model2_preds, true_labels))
    
    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=True)
    
    print(f"Contingency table:")
    print(f"  [{a:5d} {b:5d}]")
    print(f"  [{c:5d} {d:5d}]")
    print(f"McNemar's test p-value: {result.pvalue:.4f}")
    print(f"Model2 improves on {c} cases, regresses on {b} cases")
    
    if result.pvalue < 0.05:
        print("→ Statistically significant (p < 0.05)")
    else:
        print("→ Not statistically significant")
```

**How to interpret:**
- **p < 0.05**: Only 5% chance the improvement happened randomly. It's REAL.
- **p >= 0.05**: The improvement could be due to chance. Be careful claiming it works.

---

## Summary Checklist

After completing your analysis:

- [ ] Results table with all experiments (frozen, finetune, innovation)
- [ ] Confusion matrices for SST (5 classes — show which classes are confused)
- [ ] Error analysis with text examples (show 5-10 actual errors)
- [ ] Accuracy vs. sentence length analysis (does length matter?)
- [ ] Innovation impact analysis (how much did your idea help?)
- [ ] Key insights written up for the report (explain WHY)
