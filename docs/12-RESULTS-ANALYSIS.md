# 12 — Results Analysis

After running experiments, analyze what happened and why.

---

## 1. Compare Your Results to the Reference

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

## 2. Confusion Matrix

Visualize which classes are being confused:

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
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    if class_names is None:
        class_names = ['Very Neg', 'Neg', 'Neutral', 'Pos', 'Very Pos']
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        correct = cm[i, i]
        total = cm[i, :].sum()
        print(f"  {name:15s}: {correct}/{total} = {correct/total*100:.1f}%")
    
    return cm

# Usage
# plot_confusion_matrix(model, dataloaders['test'], device)
```

**Example for SST (5 classes):**
```
Per-class accuracy:
  Very Neg      : 85/200 = 42.5%
  Neg           : 120/250 = 48.0%
  Neutral       : 200/350 = 57.1%
  Pos           : 180/320 = 56.2%
  Very Pos      : 90/200 = 45.0%
```

This tells you: neutral and positive are easiest, very negative/very positive are hardest (consistent with the reference paper).

---

## 3. Per-Class Analysis for CFIMDB

```python
def analyze_binary_results(model, dataloader, device):
    """For binary classification: precision, recall, F1 per class."""
    from sklearn.metrics import classification_report
    
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

---

## 4. Error Analysis

Look at specific examples the model got wrong:

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

Analyze patterns in the errors:
- Are errors mostly between adjacent classes? (e.g., predicting 3 instead of 4)
- Are longer sentences harder?
- Are neutral sentences harder?

---

## 5. Accuracy vs. Sentence Length

```python
def accuracy_by_length(model, df, dataloader, tokenizer, device):
    """Check if accuracy varies by sentence length."""
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
            
            # Get actual lengths (non-padding tokens)
            lengths = attention_mask.sum(dim=1).cpu()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_lengths.extend(lengths.numpy())
    
    # Group by length buckets
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
    
    # Plot
    accuracy_by_len.plot(kind='bar')
    plt.title('Accuracy by Sentence Length')
    plt.xlabel('Length (tokens)')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('accuracy_by_length.png')
    plt.show()
```

---

## 6. Innovation Impact Analysis

If you implemented an innovation, measure its specific impact:

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

## 7. Statistical Significance

To check if improvements are significant:

```python
from sklearn.model_selection import cross_val_score
# For simplicity, use McNemar's test for paired comparison
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(model1_preds, model2_preds, true_labels):
    """
    Test if model2 is significantly better than model1.
    """
    # Contingency table
    #   | Model2 correct | Model2 wrong
    # ---+----------------+-------------
    # M1 correct | a | b
    # M1 wrong   | c | d
    
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

---

## Summary Checklist

After completing your analysis:

- [ ] Results table with all experiments
- [ ] Confusion matrices for SST (5 classes)
- [ ] Error analysis with text examples
- [ ] Accuracy vs. sentence length analysis
- [ ] Innovation impact analysis (with significance)
- [ ] Key insights written up for the report
