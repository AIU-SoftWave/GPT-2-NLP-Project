# 13 — Visualization

Good visualizations make your report stronger. Here are key plots to create.

---

## 1. Training Curves

Plot loss and accuracy over epochs:

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

# Usage (collect these during training):
# train_losses = [1.2, 0.9, 0.7, 0.6, 0.5]
# dev_losses = [1.1, 0.85, 0.75, 0.7, 0.72]
# train_accs = [0.35, 0.42, 0.46, 0.48, 0.50]
# dev_accs = [0.38, 0.44, 0.47, 0.49, 0.48]
# plot_training_curves(train_losses, dev_losses, train_accs, dev_accs, 'training_curves.png')
```

---

## 2. Confusion Matrix Heatmap

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, colorbar=True, values_format='d')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Rotate labels
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

Compare your methods side by side:

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
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, rotation=0)
        
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

# Example
# results = {
#     'Frozen': {'SST': 0.451, 'CFIMDB': 0.829},
#     'Fine-tuned': {'SST': 0.518, 'CFIMDB': 0.976},
#     'Ours': {'SST': 0.542, 'CFIMDB': 0.982},
# }
# plot_results_comparison(results, 'results_comparison.png')
```

---

## 4. Per-Class Accuracy Bar Chart

```python
def plot_per_class_accuracy(cm, class_names, save_path=None):
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#ff6b6b', '#ffa07a', '#ffd700', '#90ee90', '#2ecc71']
    
    bars = ax.bar(class_names, per_class_acc, color=colors[:len(class_names)])
    
    # Add value labels
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

# sst_cm = confusion_matrix(y_true, y_pred)
# plot_per_class_accuracy(sst_cm, ['Very Neg', 'Neg', 'Neutral', 'Pos', 'Very Pos'], 'per_class_accuracy.png')
```

---

## 5. Accuracy vs. Length Scatter Plot

```python
def plot_accuracy_vs_length(df, save_path=None):
    """
    df should have columns: 'correct' (bool), 'length' (int)
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Scatter with jitter
    jitter = np.random.normal(0, 0.02, len(df))
    correct = df[df['correct']]
    incorrect = df[~df['correct']]
    
    ax.scatter(correct['length'], np.ones(len(correct)) + jitter[:len(correct)], 
              alpha=0.3, s=10, c='green', label='Correct')
    ax.scatter(incorrect['length'], np.zeros(len(incorrect)) + jitter[:len(incorrect)], 
              alpha=0.3, s=10, c='red', label='Incorrect')
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Incorrect', 'Correct'])
    ax.set_xlabel('Sentence Length (tokens)', fontsize=12)
    ax.set_title('Prediction Correctness vs. Sentence Length', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 6. Attention Visualization (for Attention Pooling)

If you implemented attention pooling, visualize the attention weights:

```python
def visualize_attention(text, attention_weights, tokenizer, save_path=None):
    """
    attention_weights: list of floats, one per token
    """
    tokens = tokenizer.tokenize(text)
    
    # Truncate if too long for visualization
    if len(tokens) > 30:
        tokens = tokens[:30]
        attention_weights = attention_weights[:30]
    
    # Normalize weights for color intensity
    weights = np.array(attention_weights)
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(max(8, len(tokens)*0.4), 3))
    
    # Create colored text
    for i, (token, weight) in enumerate(zip(tokens, weights)):
        color = plt.cm.RdYlGn(weight)  # Red (low) to Green (high)
        ax.text(i, 0, token, fontsize=12,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    ax.set_xlim(-1, len(tokens))
    ax.set_ylim(-1, 1)
    ax.axis('off')
    ax.set_title('Token Attention Weights', fontsize=14)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Example:
# text = "The movie was absolutely fantastic and thrilling"
# dummy_weights = [0.05, 0.02, 0.03, 0.15, 0.20, 0.18, 0.12, 0.10, 0.15]
# visualize_attention(text, dummy_weights, tokenizer, 'attention_weights.png')
```

---

## 7. Ablation Study Bar Chart

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
    
    # Add improvement arrows
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

| Plot | Section | Why Include |
|------|---------|-------------|
| Training curves | Experiments | Show convergence, no overfitting |
| Confusion matrix | Analysis | Show which classes are confused |
| Results comparison | Results | Clear comparison of all methods |
| Per-class accuracy | Analysis | Show which classes are hardest |
| Ablation study | Approach | Show each component's contribution |
| Attention weights | Analysis | Visualize what model focuses on |
