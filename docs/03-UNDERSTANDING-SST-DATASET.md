# 03 — Understanding the SST Dataset (Beginner-Friendly)

## What You'll Learn

- What the SST dataset is and where it comes from
- How the data is stored (the "tree" format)
- How to load it using code
- What the data looks like (label distribution, sentence lengths)

---

## Key Terms (Defined Simply)

| Term | Simple definition | Analogy |
|------|------------------|---------|
| **Dataset** | A collection of examples used to train and test a model | A stack of flashcards for studying |
| **Label** | The correct answer for an example | The answer on the back of a flashcard |
| **Tree format** | A way of organizing data where things are nested inside each other | Russian nesting dolls — each doll contains a smaller one inside |
| **Parse** | Convert data from one format to another | Translating French to English |
| **Distribution** | How spread out the data is across different categories | How many students got A, B, C, D, F on a test |

---

## What Is SST?

**SST** stands for **Stanford Sentiment Treebank**.

It's a collection of **movie review sentences** from Rotten Tomatoes, each labeled with a sentiment (how positive or negative it is).

**Source:** Rotten Tomatoes (movie review website)
**Total sentences:** 11,855
**What's special:** Every phrase within every sentence is labeled, not just the whole sentence

### Why is this dataset famous?

SST is special because each sentence comes with a **parse tree** — a diagram showing how the words group together. And **every group** (not just the whole sentence) has its own sentiment label.

Example: In "The movie was fantastic," the model can learn that "fantastic" alone is very positive, and "The movie" alone is neutral, and "The movie was fantastic" as a whole is positive. This helps the model understand how word combinations create meaning.

---

## The Labels (Sentiment Scores)

Each sentence gets a score from 0 to 4:

| Label | Sentiment | What it means | Example phrase |
|-------|-----------|---------------|----------------|
| 0 | Very negative | Awful, terrible | "A complete waste of time" |
| 1 | Negative | Bad, disappointing | "Boring and predictable" |
| 2 | Neutral | Neither good nor bad | "It's an okay movie" |
| 3 | Positive | Good, enjoyable | "A fun ride from start to finish" |
| 4 | Very positive | Amazing, excellent | "An absolute masterpiece" |

---

## How the Data Is Stored (Tree Format)

### The problem

Computers don't naturally understand nested structures. We need to store sentences in a way that preserves the grouping of words.

### The solution: Penn Treebank (PTB) format

Each sentence is stored as a **nested tree** of parentheses. Think of it as Russian nesting dolls:

```
(3 (2 The) (4 movie))
```

Breaking this down:
- The **outermost** `(3 ... )` means the whole sentence has label **3** (positive)
- Inside, we have two parts:
  - `(2 The)` — The word "The" has label **2** (neutral on its own)
  - `(4 movie)` — The word "movie" has label **4** (very positive on its own — wait, "movie" alone isn't positive? That's right — the labels are assigned by human annotators based on how the word contributes to the sentence)

### A real example

Here's a real sentence from the dataset (simplified):

```
(3 (2 The) (2 Rock)) (4 (3 (2 is) (4 (2 destined) ...
```

This represents: "The Rock is destined to be the 21st Century's new Conan..."

The structure shows how words combine into phrases, and each phrase has a sentiment label.

---

## Dataset Splits

The data is divided into three groups:

| Split | Purpose | Number of sentences |
|-------|---------|-------------------|
| **Train** | Used to teach (train) the model | 8,544 |
| **Dev** (validation) | Used to tune the model during development | 1,101 |
| **Test** | Used only at the end to measure final performance | 2,210 |

### Why three splits?

Think of it like studying for an exam:

- **Training set** = Your textbook and practice problems. You study these to learn.
- **Dev set** = A practice quiz. You take it to see how you're doing and adjust your studying.
- **Test set** = The final exam. You only take it once at the end to see your true score.

**Crucial rule:** Never use the test set during development. If you practice on the final exam questions, you won't know your real score.

---

## How to Load the Data (Using pytreebank)

### What is pytreebank?

`pytreebank` is a library that reads the tree format and converts it into something we can work with in Python. It's like having a translator who reads Russian nesting dolls and writes down the important information in a simple list.

### Loading a single tree

```python
import pytreebank

tree_string = "(3 (2 The) (4 movie))"
tree = pytreebank.create_tree_from_string(tree_string)

# Get the sentence text
sentence = tree.to_lines()[0]
print(f"Sentence: {sentence}")

# Get the sentiment label
label = tree.label
print(f"Label: {label}")
```

**Expected output:**
```
Sentence: The movie
Label: 3
```

**What's happening here?**
1. We give `pytreebank` the tree string
2. It parses the tree and creates a `LabeledTree` object
3. `.to_lines()[0]` extracts the full sentence text
4. `.label` extracts the outermost (root) sentiment label

### Loading all trees from a file

```python
import pytreebank
import os

path = "./Datasets/SST2Data/trainDevTestTrees_PTB/trees/train.txt"
trees = pytreebank.import_tree_corpus(path)

print(f"Loaded {len(trees)} trees")
print(f"Type: {type(trees[0])}")

# Look at the first tree
print(f"Sentence: {trees[0].to_lines()[0]}")
print(f"Label: {trees[0].label}")
```

**Expected output:**
```
Loaded 8544 trees
Type: <class 'pytreebank.LabeledTree'>
Sentence: The Rock is destined to be the 21st Century's new `` Conan '' ...
Label: 3
```

---

## Exploring Dataset Statistics

Let's look at what's actually in the data:

```python
import pytreebank
import os
from collections import Counter

def analyze_sst_split(split_name):
    path = f"./Datasets/SST2Data/trainDevTestTrees_PTB/trees/{split_name}.txt"
    trees = pytreebank.import_tree_corpus(path)
    
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

### What to notice in the output:

**Label distribution** — Shows how many sentences of each sentiment exist. SST is fairly balanced (no class has dramatically more or fewer examples).

**Sentence lengths** — Most sentences are around 19 words long. Some are just 1 word ("Great!"). Some are up to 56 words. This tells us we don't need to process very long sequences for SST.

---

## Important: Root Label Only

For this project, we only use the **root label** — the outermost label of the tree that represents the whole sentence's sentiment.

We don't use the internal phrase-level labels. Those exist in the dataset, but we're doing **sentence-level** sentiment classification, not phrase-level.

> Think of the tree as a family tree. The root label is like the grandparent at the top — it represents the whole family's reputation. The internal nodes are like individual family members — interesting, but not what we need for this project.

---

## Quick Summary

- **SST** = Stanford Sentiment Treebank (movie review sentences with sentiment labels)
- **Labels**: 0 (very negative) to 4 (very positive)
- **Format**: Tree format (nested parentheses) — parsed by `pytreebank`
- **Splits**: 8,544 train / 1,101 dev / 2,210 test
- **Avg length**: ~19 words per sentence
- **What we use**: Only the root label (whole sentence sentiment)
