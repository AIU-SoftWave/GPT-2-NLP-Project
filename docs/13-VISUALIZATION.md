# 13 — Visualization (Beginner-Friendly)

## What You'll Learn

- How to create the key plots for your report
- What each plot tells you about your model
- How to save plots as high-quality images for the report

---

## Key Terms (Defined Simply)

| Term | Simple definition | Analogy |
|------|------------------|---------|
| **Training curve** | A graph showing how loss/accuracy changes over epochs | A fitness tracker showing your weight loss over weeks |
| **Confusion matrix** | A grid showing correct vs incorrect predictions | A gradebook showing which problems students got right/wrong |
| **Ablation study** | A bar chart showing what happens when you remove components | A recipe comparison showing the effect of each ingredient |
| **Attention weights** | Values showing which words the model focused on | A heat map of where a person was looking |

---

## 1. Training Curves

### What this tells you:

- **Is the model learning?** (Loss should decrease over time)
- **Is the model overfitting?** (Dev loss starts increasing while train loss keeps decreasing)
- **When does the model converge?** (When the curve flattens — more epochs won't help)

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(train_losses, dev_losses, train_accs, dev_accs, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax1.plot(epochs, dev_losses, 'r-o', label='Dev Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Dev Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy', markersize=4)
    ax2.plot(epochs, dev_accs, 'r-o', label='Dev Accuracy', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Dev Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

**How to read these plots:**

| Pattern | What it means |
|---------|---------------|
| Both losses decreasing | ✅ Model is learning correctly |
| Train loss decreasing, dev loss INCREASING | ⚠️ **Overfitting** — model is memorizing, not generalizing |
| Both losses flattening | ✅ Model converged — training is done |
| Loss not decreasing | ❌ Something wrong — check code, learning rate, or data |

---

## 2. Confusion Matrix Heatmap

### What this tells you:

- **Which classes are confused with each other** — diagonal = correct, off-diagonal = mistakes
- **Which class is hardest** — the row with the lowest diagonal value
- **Direction of confusion** — e.g., is "very negative" usually predicted as "negative"?

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# SST class names
sst_classes = ['Very Neg', 'Neg', 'Neutral', 'Pos', 'Very Pos']
# cm = confusion_matrix(y_true, y_pred)
# plot_confusion_matrix(cm, sst_classes, 'SST Confusion Matrix', 'sst_confusion.png')
```

---

## 3. Results Comparison Bar Chart

### What this tells you:

- How your methods compare at a glance
- Which method is best on each dataset
- The gap between frozen, fine-tuned, and your innovation

```python
def plot_results_comparison(results_dict, save_path=None):
    """
    results_dict: {
        'Frozen': {'SST': 0.451, 'CFIMDB': 0.829},
        'Fine-tuned': {'SST': 0.518, 'CFIMDB': 0.976},
        'Ours': {'SST': 0.542, 'CFIMDB': 0.982},
    }
    """
    methods = list(results_dict.keys())
    datasets = list(results_dict[methods[0]].keys())
    
    x = np.arange(len(datasets))
    width = 0.25
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method in methods:
        values = [results_dict[method][ds] for ds in datasets]
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=method)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        multiplier += 1
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Sentiment Classification Results', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 4. Per-Class Accuracy Bar Chart

### What this tells you:

- Which sentiment classes are easiest/hardest for your model
- If your innovation helps certain classes more than others

```python
def plot_per_class_accuracy(cm, class_names, save_path=None):
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#ff6b6b', '#ffa07a', '#ffd700', '#90ee90', '#2ecc71']
    
    bars = ax.bar(class_names, per_class_acc, color=colors[:len(class_names)])
    
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.1%}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 5. Ablation Study Bar Chart

### What this tells you:

- Which component of your innovation contributes the most
- Whether combining components helps more than individual ones

```python
def plot_ablation_results(ablation_dict, save_path=None):
    """
    ablation_dict: {
        'Baseline': 0.518,
        '+ Attention Pool': 0.534,
        '+ Gradual Unfreeze': 0.526,
        '+ Both': 0.542,
    }
    """
    methods = list(ablation_dict.keys())
    values = list(ablation_dict.values())
    
    colors = ['#95a5a6'] + ['#3498db'] * (len(methods) - 2) + ['#2ecc71']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, values, color=colors, width=0.6)
    
    baseline = values[0]
    for bar, val in zip(bars, values):
        diff = val - baseline
        color = '#27ae60' if diff > 0 else '#e74c3c'
        sign = '+' if diff > 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{val:.3f}\n({sign}{diff:.3f})', ha='center', va='bottom', 
               fontsize=10, color=color, fontweight='bold')
    
    ax.set_ylabel('Dev Accuracy', fontsize=12)
    ax.set_title('Ablation Study', fontsize=14)
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'Baseline: {baseline:.3f}')
    ax.legend(fontsize=10)
    ax.set_ylim(min(values) - 0.02, max(values) + 0.03)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

---

## Summary: Which Plots for Your Report?

| Plot | Report Section | Why Include |
|------|---------------|-------------|
| Training curves | Experiments | Show the model is learning and not overfitting |
| Confusion matrix | Analysis | Show which classes are confused (for SST) |
| Results comparison | Results | Show ALL methods side by side — makes the improvement clear |
| Per-class accuracy | Analysis | Show which sentiment levels are hardest |
| Ablation study | Approach | Show each component's contribution to your innovation |

**Tip:** Save each figure with `dpi=150` or `dpi=300` for publication-quality images in your report. Use `save_path` to save them as PNG files.
