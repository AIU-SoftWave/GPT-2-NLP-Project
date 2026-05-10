"""
GPT-2 Sentiment Analysis — single-experiment runner.
Call from notebook:  !python -m src.train --dataset sst --frozen --epochs 10

Saves results to results/<name>.json which the notebook loads.
"""

import json, os, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.dataset import SentimentDataset, load_sst_data, load_cfimdb_data
from src.model import get_model

os.makedirs('results', exist_ok=True)

# ─── Performance tweaks ──────────────────────────────────────────────────
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

def make_loaders(df, tokenizer, bs, max_len, workers=6):
    loaders = {}
    for split in ['train', 'dev', 'test']:
        sd = df[df['split'] == split].reset_index(drop=True)
        ds = SentimentDataset(sd, tokenizer, max_len)
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=(split == 'train'),
            num_workers=workers, pin_memory=True,
            persistent_workers=True if workers > 0 else False,
            prefetch_factor=2 if workers > 0 else None,
        )
    return loaders

# ─── Train / Eval ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, opt, crit, device, accum=1):
    model.train()
    total_loss, preds, labels = 0, [], []
    opt.zero_grad()
    pbar = tqdm(loader, desc='Train')
    for i, b in enumerate(pbar):
        ids = b['input_ids'].to(device, non_blocking=True)
        mask = b['attention_mask'].to(device, non_blocking=True)
        labs = b['label'].to(device, non_blocking=True)
        logits = model(ids, mask)
        loss = crit(logits, labs) / accum
        loss.backward()
        if (i + 1) % accum == 0:
            opt.step()
            opt.zero_grad()
        total_loss += loss.item() * accum
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels.extend(labs.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item() * accum:.4f}'})
    return total_loss / len(loader), accuracy_score(labels, preds)

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    total_loss, preds, labels = 0, [], []
    for b in tqdm(loader, desc='Eval'):
        ids = b['input_ids'].to(device, non_blocking=True)
        mask = b['attention_mask'].to(device, non_blocking=True)
        labs = b['label'].to(device, non_blocking=True)
        logits = model(ids, mask)
        total_loss += crit(logits, labs).item()
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels.extend(labs.cpu().numpy())
    return total_loss / len(loader), accuracy_score(labels, preds)

# ─── Main ────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['sst', 'cfimdb'], required=True)
    p.add_argument('--frozen', action='store_true')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--model', type=str, default='baseline', help='Model from model.py registry')
    p.add_argument('--name', type=str, default=None)
    p.add_argument('--sample', type=int, default=None, help='Use N train samples for fast testing')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == 'sst':
        df = load_sst_data(); num_classes = 5; max_len = 128
    elif args.dataset == 'cfimdb':
        df = load_cfimdb_data(); num_classes = 2; max_len = 512

    if args.sample and args.sample < len(df[df['split'] == 'train']):
        train_idx = df[df['split'] == 'train'].sample(args.sample, random_state=42).index
        other_idx = df[df['split'] != 'train'].index
        df = df.loc[train_idx.union(other_idx)]

    loaders = make_loaders(df, tokenizer, args.batch_size, max_len, args.workers)

    model = get_model(args.model, num_classes, freeze=args.frozen).to(device)
    #try:
      #  model = torch.compile(model)
     #   print('torch.compile: ON')
    #except Exception:
   #     pass

    mode = 'frozen' if args.frozen else 'finetune'
    name = args.name or f'{args.dataset}_{mode}'

    crit = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    best_dev, history = 0, {'train_loss': [], 'train_acc': [], 'dev_loss': [], 'dev_acc': []}
    for epoch in range(args.epochs):
        print(f'\n--- Epoch {epoch+1}/{args.epochs} ---')
        tl, ta = train_one_epoch(model, loaders['train'], opt, crit, device, args.accum)
        dl, da = evaluate(model, loaders['dev'], crit, device)
        print(f'Train Loss: {tl:.4f} | Acc: {ta:.4f}')
        print(f'Dev   Loss: {dl:.4f} | Acc: {da:.4f}')
        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['dev_loss'].append(dl)
        history['dev_acc'].append(da)
        if da > best_dev:
            best_dev = da
            torch.save(model.state_dict(), f'checkpoints/best_model_{name}.pt')
            print(f'>> New best dev: {da:.4f}')

    model.load_state_dict(torch.load(f'checkpoints/best_model_{name}.pt'))
    test_labels = loaders['test'].dataset.labels
    if (test_labels == -1).any():
        dl_final, ta_final = evaluate(model, loaders['dev'], crit, device)
        print(f'\nDev Loss: {dl_final:.4f} | Dev Acc: {ta_final:.4f} (test labels unavailable, using dev)')
    else:
        tl_final, ta_final = evaluate(model, loaders['test'], crit, device)
        print(f'\nTest Loss: {tl_final:.4f} | Test Acc: {ta_final:.4f}')

    results = {'name': name, 'best_dev': best_dev, 'test_acc': ta_final, 'history': history}
    with open(f'results/{name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to results/{name}.json')

if __name__ == '__main__':
    main()
