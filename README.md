# 🐦 ML Assignment 2 — Health Tweet Dataset

A labeled tweet dataset used for **Machine Learning Assignment 2** coursework, focused on natural language processing tasks on health and wellness content from Twitter/X.

## Dataset

The repository contains `goodhealth.txt` — a pipe-delimited file of health-related tweets collected from Twitter in 2015, sourced from wellness and health-focused accounts.

### Format

Each line follows the format:
```
tweet_id|timestamp|tweet_text
```

**Example:**
```
577117109701967872|Sun Mar 15 14:40:15 +0000 2015|It's not hard to get a little cardio in at home! @KristinMcGee has an awesome plyometric move...
577083140977651712|Sun Mar 15 12:25:17 +0000 2015|Beauty really does come from within. Discover wellness tips...
```

### Dataset Stats

| Property | Value |
|---|---|
| File size | ~1.2 MB |
| Domain | Health, fitness, nutrition, and wellness |
| Language | English |
| Source | Health-focused Twitter accounts (2015) |

### Content Topics

Tweets cover a range of health and wellness topics including:
- Fitness tips and workout moves
- Nutrition and diet advice
- Sleep and recovery
- Weight management
- Mental wellness
- Beauty and skincare

## Potential ML Tasks

This dataset is suitable for:

- **Text classification** — categorize tweets by health topic (fitness, nutrition, sleep, etc.)
- **Sentiment analysis** — positive, neutral, or negative health sentiment
- **Named Entity Recognition (NER)** — identify health conditions, food items, exercises
- **Topic modeling** — LDA or NMF to discover latent health topic clusters
- **Feature extraction** — TF-IDF or word embeddings for downstream ML models

## Loading the Data

```python
import pandas as pd

df = pd.read_csv(
    "goodhealth.txt",
    sep="|",
    header=None,
    names=["tweet_id", "timestamp", "text"],
    engine="python"
)

print(df.shape)
print(df["text"].head())
```

## Course Context

**Assignment:** ML Assignment 2  
**Focus:** NLP/text classification using tweet data  
**Techniques:** Likely involves text preprocessing, feature extraction (TF-IDF / word embeddings), and classification models (Naive Bayes, SVM, Logistic Regression, or neural networks)
