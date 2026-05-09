# 11 — Experiment Tracking

Track your experiments so you can compare results systematically.

---

## Simple Results Table

Create a DataFrame to track results:

```python
import pandas as pd
from datetime import datetime

results = []

def log_experiment(dataset, approach, dev_acc, test_acc, 
                   lr, batch_size, epochs, notes=""):
    results.append({
        'dataset': dataset,
        'approach': approach,
        'dev_acc': dev_acc,
        'test_acc': test_acc,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'notes': notes
    })
    return pd.DataFrame(results)

# Log your experiments as you run them
log_experiment('sst', 'frozen', 0.451, 0.448, 1e-3, 16, 10, 'baseline')
log_experiment('cfimdb', 'frozen', 0.829, 0.825, 1e-3, 8, 5, 'baseline')
log_experiment('sst', 'finetune', 0.518, 0.512, 5e-5, 8, 5, 'full fine-tuning')
log_experiment('cfimdb', 'finetune', 0.976, 0.971, 3e-5, 4, 3, 'full fine-tuning')

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

---

## CSV Logging

Save results to a CSV file for later analysis:

```python
import csv
import os

EXPERIMENT_LOG = "experiment_results.csv"

def save_result(dataset, approach, dev_acc, test_acc, 
                lr, batch_size, epochs, notes=""):
    file_exists = os.path.exists(EXPERIMENT_LOG)
    
    with open(EXPERIMENT_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['dataset', 'approach', 'dev_acc', 'test_acc',
                           'lr', 'batch_size', 'epochs', 'timestamp', 'notes'])
        writer.writerow([
            dataset, approach, dev_acc, test_acc,
            lr, batch_size, epochs,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            notes
        ])

# Usage
save_result('sst', 'frozen', 0.451, 0.448, 1e-3, 16, 10, 'baseline')
```

---

## Organizing Model Checkpoints

Use a consistent naming scheme:

```python
checkpoint_name = f"{dataset}_{approach}_lr{lr}_bs{batch_size}_epoch{epoch}.pt"
# Example: sst_finetune_lr5e-5_bs8_epoch3.pt
```

---

## Hyperparameter Search

Try multiple hyperparameters systematically:

```python
def hyperparameter_search(dataset_name, approach, param_grid):
    """Run multiple experiments with different hyperparameters."""
    best_result = None
    best_dev = 0
    
    for lr in param_grid['lr']:
        for batch_size in param_grid['batch_size']:
            print(f"\n{'='*40}")
            print(f"Trying: lr={lr}, batch_size={batch_size}")
            
            dev_acc, test_acc = run_experiment(
                dataset_name=dataset_name,
                freeze=(approach == 'frozen'),
                batch_size=batch_size,
                lr=lr,
                epochs=param_grid.get('epochs', 5)
            )
            
            save_result(dataset_name, approach, dev_acc, test_acc, 
                       lr, batch_size, param_grid.get('epochs', 5))
            
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_result = {
                    'dev': dev_acc, 'test': test_acc,
                    'lr': lr, 'batch_size': batch_size
                }
    
    print(f"\n=== Best for {dataset_name}_{approach} ===")
    print(best_result)
    return best_result

# Example grid
sst_grid = {
    'lr': [1e-5, 3e-5, 5e-5],
    'batch_size': [8, 16],
    'epochs': 5
}

# Run (warning: this takes a while!)
# best = hyperparameter_search('sst', 'finetune', sst_grid)
```

---

## Ablation Studies

If your innovation has multiple components, test each one separately:

```python
def run_ablation(components, dataset='sst'):
    """
    Test each component individually and together.
    
    components: dict of {name: (model_class, kwargs)}
    """
    results = {}
    
    for name, (model_class, kwargs) in components.items():
        print(f"\n--- Testing: {name} ---")
        dev_acc, test_acc = run_with_model(
            dataset=dataset,
            model_class=model_class,
            **kwargs
        )
        results[name] = {'dev': dev_acc, 'test': test_acc}
    
    return results

# Example:
# components = {
#     'baseline': (GPT2Classifier, {'freeze': False}),
#     'attention_pool': (GPT2AttentionPooling, {}),
#     'multi_layer': (GPT2MultiLayer, {}),
#     'combined': (GPT2AttentionMultiLayer, {}),
# }
# ablation_results = run_ablation(components)
```

---

## Final Results Table Template

```markdown
| Dataset | Approach | Dev Acc | Test Acc | vs. Baseline |
|---------|----------|---------|----------|-------------|
| SST | Frozen (baseline) | 0.451 | 0.448 | — |
| SST | Fine-tuned (baseline) | 0.518 | 0.512 | — |
| SST | Fine-tuned + Attention Pool | 0.534 | 0.528 | +1.6% |
| SST | Fine-tuned + Gradual Unfreeze | 0.526 | 0.520 | +0.8% |
| SST | Full Innovation | 0.542 | 0.535 | +2.4% |
| CFIMDB | Frozen (baseline) | 0.829 | 0.825 | — |
| CFIMDB | Fine-tuned (baseline) | 0.976 | 0.971 | — |
| CFIMDB | Fine-tuned + Innovation | 0.982 | 0.979 | +0.6% |
```
