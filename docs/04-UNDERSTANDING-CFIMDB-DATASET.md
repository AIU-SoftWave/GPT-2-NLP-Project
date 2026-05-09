# 04 — Understanding the CFIMDB (IMDB) Dataset (Beginner-Friendly)

## What You'll Learn

- What the CFIMDB/IMDB dataset is
- How the data is stored (CSV format)
- How to load and clean the data
- How to create training/validation/test splits
- Key statistics about the dataset

---

## Key Terms (Defined Simply)

| Term | Simple definition | Analogy |
|------|------------------|---------|
| **Binary classification** | A problem with only 2 possible answers | True/False, Yes/No, Positive/Negative |
| **CSV** | Comma-Separated Values — a simple table format | Like an Excel spreadsheet saved as plain text |
| **HTML tags** | Code used to format web pages (like `<br>` for line breaks) | Invisible formatting marks that need to be cleaned up |
| **Stratify** | Keep the same proportion of each class when splitting | If a class has 50% of data, keep it at 50% in every split |

---

## What Is CFIMDB?

**CFIMDB** is a dataset of **movie reviews** from IMDB. Each review is labeled as either **positive** or **negative**.

- **Source**: IMDB (Internet Movie Database)
- **Total reviews**: 50,000
- **Classes**: 2 (positive, negative)
- **Balance**: Perfectly balanced — 25,000 positive, 25,000 negative

### Wait, what's the difference between SST and CFIMDB?

| Aspect | SST | CFIMDB |
|--------|-----|--------|
| **Type of text** | Short sentences (~19 words) | Full reviews (~234 words) |
| **Number of classes** | 5 (very negative → very positive) | 2 (positive or negative) |
| **Difficulty** | Harder (more choices, less context) | Easier (only 2 choices, more text) |
| **Size** | 8,544 train examples | 40,000 train examples |

> **Analogy:** SST is like asking "How good is this movie on a scale of 1-5?" while CFIMDB is like asking "Did you like this movie — yes or no?" The second is much easier.

---

## Data Format (CSV)

The data comes in a file called `IMDB Dataset.csv`. Here's what it looks like:

```
review,sentiment
"One of the other reviewers has mentioned...",positive
"A wonderful little production...",positive
"Basically there's a family where...",negative
```

It has two columns:
- **`review`**: The full text of the movie review (may contain HTML like `<br />` tags)
- **`sentiment`**: Either `positive` or `negative`

### What is a CSV file?

CSV stands for **Comma-Separated Values**. It's the simplest way to store a table as text:
- Each row is one example
- Column values are separated by commas
- The first row usually has the column names

---

## Creating Train/Dev/Test Splits

Unlike SST (which came pre-split), the IMDB CSV doesn't tell us which reviews are for training, validation, or testing. We need to **create our own splits**.

### Why split at all?

Remember the exam analogy:
- **Training set** = Textbook (what you study)
- **Dev set** = Practice quiz (check your progress)
- **Test set** = Final exam (measure true performance)

If we test on data we've already trained on, we're cheating — we won't know if the model actually learned general patterns or just memorized the answers.

### Our split: 80% / 10% / 10%

| Split | Number of reviews | Purpose |
|-------|------------------|---------|
| Train | 40,000 | Teach the model |
| Dev | 5,000 | Tune hyperparameters during development |
| Test | 5,000 | Measure final performance |

### What does "stratify" mean?

When splitting, we use `stratify` to keep the same proportions in each split. Since the full dataset is 50% positive / 50% negative, we want each split to also be 50/50.

> **Analogy:** If your class has 10 boys and 10 girls, and you want to split into 2 groups of 10, you'd put 5 boys and 5 girls in each group. That's stratifying — keeping the balance.

---

## Text Cleaning

### Why clean the data?

The reviews contain HTML tags like `<br />` which are used for line breaks on web pages. These add noise to our data — the model doesn't need to learn that `<br />` means "line break" for sentiment analysis.

### What we do:

1. Remove `<br />` tags (replace them with spaces)
2. Collapse multiple spaces into one
3. Remove leading/trailing whitespace

```python
import re

def clean_text(text):
    # Step 1: Remove <br> tags
    text = re.sub(r'<br\s*/?>', ' ', text)
    # Step 2: Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Step 3: Remove leading/trailing spaces
    return text.strip()

# Example
dirty = "Great movie!<br /><br />Loved every minute.  "
clean = clean_text(dirty)
print(repr(clean))  # 'Great movie! Loved every minute.'
```

---

## Dataset Statistics

Reviews are much longer than SST sentences:

| Statistic | SST | CFIMDB |
|-----------|-----|--------|
| Average length | ~19 words | ~234 words |
| Shortest | 1 word | 10 words |
| Longest | 56 words | 2,470 words |

### Why does length matter?

GPT-2 has a **maximum context length** of 1024 tokens (roughly 750-800 words). If a review is longer than 1024 tokens, we need to **truncate** (cut off) the extra part.

For SST, no sentences are even close to 1024 tokens. But for CFIMDB, some reviews are much longer and will need to be truncated.

> **Analogy:** GPT-2 is like a person with a short attention span. It can only read about 1024 "words" (tokens) before it forgets the beginning. If a review is longer, we have to summarize (truncate) it.

---

## Important Note vs. Original CS224N Project

The original CS224N project used a **smaller** CFIMDB dataset:
- Train: 1,701 examples
- Dev: 245 examples
- Test: 488 examples

Your dataset (50K IMDB reviews — 40K train) is about **23x larger**. This is actually an advantage! More data = better learning.

**Keep this in mind:** When comparing your results to the reference report, your scores will likely be higher simply because you have more training data.

---

## What to Remember

- **CFIMDB** = Binary sentiment classification (positive/negative)
- **Format**: CSV with two columns (review text, sentiment label)
- **Size**: 50,000 reviews (perfectly balanced)
- **Splits**: Create your own 80/10/10 split using `train_test_split` with `stratify`
- **Cleaning**: Remove HTML `<br />` tags
- **Challenge**: Some reviews exceed GPT-2's 1024-token limit
