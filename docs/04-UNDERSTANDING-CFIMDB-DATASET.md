# 04 — Understanding the CFIMDB / IMDB Dataset

## What Is CFIMDB?

In the original CS224N project, CFIMDB is a curated subset of the IMDB movie review dataset with **binary sentiment labels** (positive/negative).

Your dataset is the **full IMDB dataset** (50,000 reviews), which is actually better — more data for training.

**File location:**
```
./Datasets/CFIMDB/IMDB Dataset.csv
```

---

## Data Format

The file is a CSV with two columns:

```
review,sentiment
"One of the other reviewers has mentioned...",positive
"A wonderful little production...",positive
"Basically there's a family where...",negative
```

| Column | Type | Description |
|--------|------|-------------|
| `review` | string | Full movie review text (may contain HTML `<br />` tags) |
| `sentiment` | string | `positive` or `negative` |

---

## Loading the Data

```python
import pandas as pd

df = pd.read_csv("./Datasets/CFIMDB/IMDB Dataset.csv")
print(f"Total reviews: {len(df)}")
print(df.head())
print(f"\nLabel distribution:\n{df['sentiment'].value_counts()}")
```

**Expected output:**
```
Total reviews: 50000
                                              review sentiment
0  "One of the other reviewers has mentioned...   positive
1  "A wonderful little production...              positive
...
Label distribution:
sentiment
positive    25000
negative    25000
```

Perfectly balanced: 25,000 positive + 25,000 negative.

---

## Creating Train/Dev/Test Splits

The IMDB CSV doesn't come pre-split. You need to create splits yourself.

Recommended split: **80% train, 10% dev, 10% test**

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./Datasets/CFIMDB/IMDB Dataset.csv")

# Convert sentiment to binary labels
df['label'] = (df['sentiment'] == 'positive').astype(int)

# First split: separate test
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

# Second split: separate dev from test
dev_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
)

print(f"Train: {len(train_df)} ({train_df['label'].mean():.2%} positive)")
print(f"Dev:   {len(dev_df)} ({dev_df['label'].mean():.2%} positive)")
print(f"Test:  {len(test_df)} ({test_df['label'].mean():.2%} positive)")
```

**Expected output:**
```
Train: 40000 (50.00% positive)
Dev:   5000 (50.00% positive)
Test:  5000 (50.00% positive)
```

---

## Text Cleaning

The reviews contain HTML `<br />` tags. Clean them:

```python
import re

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)  # Remove <br> tags
    text = re.sub(r'\s+', ' ', text)         # Collapse whitespace
    return text.strip()

df['review'] = df['review'].apply(clean_text)

# Check
print(df['review'].iloc[0][:200])
```

---

## Dataset Statistics

```python
df['word_count'] = df['review'].str.split().str.len()
print(f"Avg review length: {df['word_count'].mean():.0f} words")
print(f"Min: {df['word_count'].min()}, Max: {df['word_count'].max()}")
print(f"Median: {df['word_count'].median():.0f}")
```

**Expected output:**
```
Avg review length: 234 words
Min: 10, Max: 2470
Median: 174 words
```

Reviews are much longer than SST sentences (avg ~19 words). This matters for GPT-2's maximum context length (1024 tokens).

---

## Important Note vs. Original CFIMDB

The original CS224N CFIMDB dataset has:
- Train: 1,701 examples
- Dev: 245 examples
- Test: 488 examples

Your IMDB dataset is **~23x larger** (40K train). This will likely give you **better results** than the reference report. Keep this in mind when comparing.
