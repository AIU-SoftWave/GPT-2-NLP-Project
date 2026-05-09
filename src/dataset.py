"""
Data loading and pre-tokenized Dataset for multiprocessing-safe DataLoader workers.
"""

import os
import re
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SentimentDataset(Dataset):
    """Pre-tokenizes all text once at init — no per-item tokenizer calls."""

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
    """Load SST data from tree files into a pandas DataFrame."""
    import pytreebank
    data = []
    tree_path = './data/SST2Data/trainDevTestTrees_PTB/trees/'
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
    """Load IMDB CSV, clean HTML, and create train/dev/test splits."""
    df = pd.read_csv('./data/CFIMDB/IMDB Dataset.csv')
    def clean_text(text):
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    df['text'] = df['review'].apply(clean_text)
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    dev_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )
    train_df['split'] = 'train'
    dev_df['split'] = 'dev'
    test_df['split'] = 'test'
    return pd.concat([train_df, dev_df, test_df])[['text', 'label', 'split']]
