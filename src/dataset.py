"""
Data loading and pre-tokenized Dataset for multiprocessing-safe DataLoader workers.

All datasets are tokenized once at construction time (not per-batch),
so DataLoader workers never need access to the tokenizer object.
This avoids pickling issues with HuggingFace tokenizers.
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    """
    PyTorch Dataset that pre-tokenizes all text at init.

    This design means DataLoader workers (multiprocessing) only need
    to index into pre-computed tensors — no tokenizer access required.
    """

    def __init__(self, df, tokenizer, max_length=128):
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)
        encodings = tokenizer(
            df['text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
        )
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'label': self.labels[idx],
        }


def load_sst_data():
    """
    Load SST (Stanford Sentiment Treebank) data.

    Returns a DataFrame with columns: text, label (0-4), split (train/dev/test).
    Parses Penn Treebank-style tree files using the pytreebank library.
    """
    import pytreebank
    data = []
    tree_path = './data/Datasets/SST2Data/trainDevTestTrees_PTB/trees/'
    for split in ['train', 'dev', 'test']:
        full_path = os.path.join(tree_path, f'{split}.txt')
        if not os.path.exists(full_path):
            print(f'Warning: {full_path} not found. Skipping...')
            continue
        trees = pytreebank.import_tree_corpus(full_path)
        for tree in trees:
            data.append({'text': tree.to_lines()[0], 'label': tree.label, 'split': split})
    return pd.DataFrame(data)


def load_cfimdb_data():
    """
    Load CFIMDB (CS224N Default Project subset).

    Returns a DataFrame with columns: text, label (0/1), split (train/dev/test).
    Test labels are set to -1 (unlabeled by design — held out for evaluation).

    Splits: 1,701 train / 245 dev / 488 test.
    """
    base = './data/Datasets/CFIMDB_CS224N/'
    parts = []

    for split, fname in [('train', 'ids-cfimdb-train.csv'), ('dev', 'ids-cfimdb-dev.csv')]:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            print(f'Warning: {path} not found. Skipping...')
            continue
        df = pd.read_csv(path, sep='\t')
        df = df.rename(columns={'sentence': 'text', 'sentiment': 'label'})
        df['split'] = split
        parts.append(df[['text', 'label', 'split']])

    path = os.path.join(base, 'ids-cfimdb-test-student.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t', header=None, skiprows=1, usecols=[3], names=['text'])
        df['label'] = -1
        df['split'] = 'test'
        parts.append(df[['text', 'label', 'split']])

    return pd.concat(parts, ignore_index=True)
