"""
Analyze document length distributions for CFIMDB and SST datasets.
Generates Figure 5: length-vs-gain analysis.
"""

import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import load_sst_data, load_cfimdb_data

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def compute_lengths(texts, max_length=512):
    lengths = []
    for t in texts:
        tokens = tokenizer.encode(t, truncation=True, max_length=max_length)
        lengths.append(len(tokens))
    return np.array(lengths)

datasets = {}

# CFIMDB
df_cfi = load_cfimdb_data()
lens_cfi = {}
for split in ['train', 'dev', 'test']:
    subset = df_cfi[df_cfi['split'] == split]
    lens_cfi[split] = compute_lengths(subset['text'].tolist(), max_length=512)
datasets['CFIMDB'] = lens_cfi

# SST
df_sst = load_sst_data()
lens_sst = {}
for split in ['train', 'dev', 'test']:
    subset = df_sst[df_sst['split'] == split]
    lens_sst[split] = compute_lengths(subset['text'].tolist(), max_length=128)
datasets['SST'] = lens_sst

# Results from experiments
results = {
    'CFIMDB': {
        'frozen': {'Last-token': 0.886, 'Gated Attn': 0.951, 'Gain': 0.0733},
        'finetune': {'Last-token': 0.971, 'Gated Attn': 0.955, 'Gain': -0.016},
    },
    'SST': {
        'frozen': {'Last-token': 0.481, 'Gated Attn': 0.459, 'Gain': -0.022},
        'finetune': {'Last-token': 0.513, 'Gated Attn': 0.515, 'Gain': 0.002},
    },
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [3, 1]})

for row, dataset_name in enumerate(['CFIMDB', 'SST']):
    ax_hist = axes[row, 0]
    ax_bar = axes[row, 1]
    lens = datasets[dataset_name]

    all_lens = np.concatenate([lens['train'], lens['dev'], lens['test']])
    ax_hist.hist(all_lens, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    median_l = np.median(all_lens)
    mean_l = np.mean(all_lens)
    ax_hist.axvline(median_l, color='red', linestyle='--', linewidth=1.5, label=f'Median={median_l:.0f}')
    ax_hist.axvline(mean_l, color='darkorange', linestyle=':', linewidth=1.5, label=f'Mean={mean_l:.0f}')
    ax_hist.set_xlabel('Token Length')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title(f'{dataset_name} — Document Length Distribution')
    ax_hist.legend(fontsize=8)

    # Bar chart: gain from gated attention
    modes = results[dataset_name]
    x = np.arange(len(modes))
    gains = [modes[m]['Gain'] * 100 for m in modes]
    colors = ['#2ecc71' if g >= 0 else '#e74c3c' for g in gains]
    ax_bar.barh(x, gains, color=colors, height=0.5)
    ax_bar.set_yticks(x)
    ax_bar.set_yticklabels(list(modes.keys()))
    ax_bar.axvline(0, color='black', linewidth=0.8)
    ax_bar.set_xlabel('Accuracy Gain (%)')
    ax_bar.set_title('Gated Attention Gain')
    for i, g in enumerate(gains):
        ax_bar.text(g + (0.3 if g >= 0 else -0.3), i, f'{g:+.2f}%',
                    va='center', ha='left' if g >= 0 else 'right', fontsize=9)

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), '..', 'LatexReport', 'figures')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'length_analysis.pdf'), bbox_inches='tight')
print(f'Saved to {out_dir}/length_analysis.pdf')

# Print statistics
for dataset_name in ['CFIMDB', 'SST']:
    lens = datasets[dataset_name]
    all_lens = np.concatenate([lens['train'], lens['dev'], lens['test']])
    print(f'\n{dataset_name}:')
    print(f'  Count: {len(all_lens)}')
    print(f'  Mean: {np.mean(all_lens):.1f}, Median: {np.median(all_lens):.0f}')
    print(f'  Min: {np.min(all_lens)}, Max: {np.max(all_lens)}')
    print(f'  Std: {np.std(all_lens):.1f}')
    for split in ['train', 'dev', 'test']:
        print(f'  {split}: {len(lens[split])} docs, mean {np.mean(lens[split]):.1f}')
