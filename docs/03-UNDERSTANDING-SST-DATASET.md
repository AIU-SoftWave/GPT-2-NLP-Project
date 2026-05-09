# 03 — Understanding the SST Dataset

## What Is SST?

The **Stanford Sentiment Treebank (SST)** is a dataset of movie review sentences with fine-grained sentiment labels. Every phrase in every sentence is labeled, not just the whole sentence.

- **Source**: Rotten Tomatoes movie reviews
- **Total sentences**: 11,855
- **Label granularity**: 5 classes (0-4)
- **Unique feature**: Full parse trees with sentiment at every node

---

## Tree Format (Penn Treebank Style)

The SST trees come in PTB (Penn Treebank) format. Each line is one sentence as a nested tree.

**File location:**
```
./Datasets/SST2Data/trainDevTestTrees_PTB/trees/train.txt
./Datasets/SST2Data/trainDevTestTrees_PTB/trees/dev.txt
./Datasets/SST2Data/trainDevTestTrees_PTB/trees/test.txt
```

**Example tree:**
```
(3 (2 (2 The) (2 Rock)) (4 (3 (2 is) (4 (2 destined) (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 Century) (2 (2 's) (2 (2 new) (2 (2 `` (2 Conan) '') (2 (2 and) (2 (2 that) (2 (2 he) (2 (2 's) (2 (2 going) (2 (2 to) (2 (2 make) (2 (2 a) (2 (2 splash) (2 (2 even) (2 (2 greater) (2 (2 than) (2 (2 Arnold) (2 (2 Schwarzenegger) (2 ,) (2 (2 Jean-Claud) (2 (2 Van) (2 (2 Damme) (2 (2 or) (2 (2 Steven) (2 Segal))))))))))))))))))))))))))))))))))))))))) (2 .)))
```

### Structure:
- **Format**: `(label subtree_or_word)`
- The first integer is the **sentiment label** (0-4)
- If followed by a word → leaf node (word)
- If followed by `(...)` → internal node with children

---

## Label Mapping

| Label | Sentiment | Range (from README) |
|-------|-----------|---------------------|
| 0 | Very negative | [0.0, 0.2] |
| 1 | Negative | (0.2, 0.4] |
| 2 | Neutral | (0.4, 0.6] |
| 3 | Positive | (0.6, 0.8] |
| 4 | Very positive | (0.8, 1.0] |

---

## Dataset Splits

| Split | # Sentences | File |
|-------|-------------|------|
| Train | 8,544 | `train.txt` |
| Dev (validation) | 1,101 | `dev.txt` |
| Test | 2,210 | `test.txt` |

**Important**: You only use the **root label** (the outermost one) for sentence-level sentiment classification.

---

## Using pytreebank to Parse Trees

The `pytreebank` library handles all the parsing for you.

### Installation
```bash
pip install pytreebank
```

### Loading a single tree
```python
import pytreebank

tree_string = "(3 (2 The) (4 movie))"
tree = pytreebank.create_tree_from_string(tree_string)

# Get the sentence text
sentence = tree.to_lines()[0]
print(f"Sentence: {sentence}")

# Get the root sentiment label
label = tree.label
print(f"Label: {label}")
```

**Expected output:**
```
Sentence: The movie
Label: 3
```

### Loading all trees from a file
```python
import pytreebank
import os

path = "./Datasets/SST2Data/trainDevTestTrees_PTB/trees/train.txt"
trees = pytreebank.load_tree_file(path, format="custom")

print(f"Loaded {len(trees)} trees")
print(f"Type: {type(trees[0])}")

# First tree
print(f"Sentence: {trees[0].to_lines()[0]}")
print(f"Label: {trees[0].label}")
```

---

## Exploring Dataset Statistics

```python
import pytreebank
import os
from collections import Counter

def analyze_sst_split(split_name):
    path = f"./Datasets/SST2Data/trainDevTestTrees_PTB/trees/{split_name}.txt"
    trees = pytreebank.load_tree_file(path, format="custom")
    
    labels = Counter()
    lengths = []
    
    for tree in trees:
        sentence = tree.to_lines()[0]
        labels[tree.label] += 1
        lengths.append(len(sentence.split()))
    
    print(f"\n=== {split_name.upper()} ===")
    print(f"Count: {len(trees)}")
    print(f"Label distribution: {dict(sorted(labels.items()))}")
    print(f"Avg sentence length: {sum(lengths)/len(lengths):.1f} words")
    print(f"Min length: {min(lengths)}, Max length: {max(lengths)}")

for split in ["train", "dev", "test"]:
    analyze_sst_split(split)
```

**Expected output (approximate):**
```
=== TRAIN ===
Count: 8544
Label distribution: {0: 1092, 1: 1406, 2: 1680, 3: 2064, 4: 1288}
Avg sentence length: 19.2 words
Min length: 1, Max length: 56

=== DEV ===
Count: 1101
...
```

---

## Tabular Format Files (Alternative)

The SST dataset also comes in a tabular format:

| File | Contents |
|------|----------|
| `datasetSentences.txt` | sentence_index → sentence text |
| `datasetSplit.txt` | sentence_index → split (1=train, 2=test, 3=dev) |
| `SOStr.txt` | Tokenized words (pipe-separated) |
| `STree.txt` | Parent pointers (pipe-separated) |
| `dictionary.txt` | phrase → phrase_id |
| `sentiment_labels.txt` | phrase_id → sentiment float (0-1) |

We'll use the **tree format** (`pytreebank`) since it's simpler.
