
# Item-Based Recommendation System

This folder contains a modular implementation of an **item-based KNN collaborative filtering** recommendation system. The system recommends movies to users by identifying and suggesting items similar. The design is lightweight, interpretable, and well-suited for sparse rating datasets like MovieLens and Netflix.

---

## Project Structure

```
item-cf/
├── item_based.py              # Core item-item collaborative filtering model
├── tune_k_item_based.py       # Hyperparameter tuning for top-k recommendation
├── movie-lens.ipynb           # MovieLens dataset processing and analysis
├── netflix.ipynb              # Netflix dataset preparation and experimentation
└── README.md                  # Project documentation (this file)
```

---

## Overview

This project implements:
- An **item-item collaborative filtering algorithm**
- Memory-based recommendation using **cosine similarity**
- **Top-k filtering** for selecting most similar items
- Compatibility with large-scale sparse datasets
- Efficient matrix handling using `scipy.sparse`

It does not require user profile data or metadata, only interaction history (ratings).

---

## Features

- Sparse matrix format (CSR) for efficient computation
- Train/validation/test split for robust evaluation
- Fine-tunable hyperparameter `k` (number of similar items)
- Cleanly separated scripts for modeling and tuning
- Dataset-agnostic implementation

---

## Implementation Details

### 1. **Core Recommender (`item_based.py`)**
- Computes **cosine similarity** between item vectors
- For a given user, recommends top-k similar items not yet rated
- Optimized with sparse matrix ops (`scipy.sparse`)

### 2. **Hyperparameter Tuning (`tune_k_item_based.py`)**
- Tests various values of `k` (number of top similar items)
- Uses held-out validation data to evaluate performance
- Check RMSE on validation

### 3. **Notebook Analysis**

#### `movie-lens.ipynb`
- Loads and preprocesses MovieLens dataset
- Converts timestamps, builds rating matrix
- Splits into train / validation / test
- Checks data integrity

#### `netflix.ipynb`
- Loads and preprocesses Netflix dataset
- Converts timestamps, builds rating matrix
- Splits into train / validation / test
- Checks data integrity

---

## Sample Workflow

1. Load dataset and convert to rating matrix
2. Split into train / validation / test
3. Fit model using training data
4. Recommend top-k items per user using cosine similarity
5. Evaluate using notebook cells or custom metrics

---

## Performance Metrics

These evaluations assess the **RMSE, Relevance, Coverage, diversity, novelty**, and **serendipity** of the recommendations produced by the item-based collaborative filtering model.

### MovieLens Dataset

| Metric                          | Area               | Value     | Value Range  | Meaning                                                                 |
|---------------------------------|--------------------|-----------|--------------|-------------------------------------------------------------------------|
| Recovery                        | Relevance          | None      | [0, 0.9]     | How early relevant items appear in top-N recommendations                |
| Normalized AggDiv (diversity)   | Inter-user diversity | 0.134171 | [0, 1]       | Proportion of unique items recommended across all users                |
| Normalized AggDiv (coverage)    | Coverage           | 0.355368  | [0, 1]       | Proportion of unique items recommended across all users vs catalog     |
| Item Space Coverage             | Coverage           | 25.031    | [0, ∞]       | Total number of unique items recommended and their frequency            |
| Normalized ItemDeg              | Novelty            | 0.883     | [0, 1]       | Novelty based on inverse item popularity                               |
| Unexpectedness (no relevance)   | Serendipity        | 0.873     | [0, 1]       | Proportion of unexpected (less popular) items                          |
| Serendipity (with relevance)    | Serendipity        | 0.0       | [0, 1]       | Proportion of unexpected and relevant items                            |
| RMSE                            | Relevance          | 1.072     | [0, 6]       | Root Mean Square Error of predicted vs actual ratings                  |

---

### Netflix Dataset

| Metric                          | Area               | Value     | Value Range  | Meaning                                                                 |
|---------------------------------|--------------------|-----------|--------------|-------------------------------------------------------------------------|
| Recovery                        | Relevance          | None      | [0, 0.9]     | How early relevant items appear in top-N recommendations                |
| Normalized AggDiv (diversity)   | Inter-user diversity | 0.073533 | [0, 1]       | Proportion of unique items recommended across all users                |
| Normalized AggDiv (coverage)    | Coverage           | 0.573308  | [0, 1]       | Proportion of unique items recommended across all users vs catalog     |
| Item Space Coverage             | Coverage           | 21.595    | [0, ∞]       | Total number of unique items recommended and their frequency            |
| Normalized ItemDeg              | Novelty            | 0.891     | [0, 1]       | Novelty based on inverse item popularity                               |
| Unexpectedness (no relevance)   | Serendipity        | 0.882     | [0, 1]       | Proportion of unexpected (less popular) items                          |
| Serendipity (with relevance)    | Serendipity        | 0.0       | [0, 1]       | Proportion of unexpected and relevant items                            |
| RMSE                            | Relevance          | 0.922     | [0, 6]       | Root Mean Square Error of predicted vs actual ratings                  |

---

### Final Comparison Table

| Metric                        | MovieLens | Netflix  | Conclusion                                                                 |
|------------------------------|-----------|----------|----------------------------------------------------------------------------|
| Recovery                     | None      | None     | Recovery is not available for either (no relevant item in top-N)           |
| Normalized AggDiv (diversity)| 0.134     | 0.074    | MovieLens delivers more diverse recommendations                           |
| Normalized AggDiv (coverage) | 0.355     | 0.573    | Netflix covers more of the item catalog                                   |
| Item Space Coverage          | 25.031    | 21.595   | MovieLens includes slightly more variety in unique item recommendations   |
| Normalized ItemDeg (novelty) | 0.883     | 0.891    | Netflix provides slightly more novel recommendations                      |
| Unexpectedness               | 0.873     | 0.882    | Netflix yields marginally more unexpected items                           |
| Serendipity                  | 0.0       | 0.0      | Neither model surfaces unexpected yet relevant items                      |
| RMSE                         | 1.072     | 0.922    | Netflix produces more accurate predictions                                |

---

**Summary:**  
Netflix recommendations are **more accurate and novel**, with **broader coverage**. MovieLens performs slightly better in **inter-user diversity** but less in accuracy. Both systems lack meaningful **serendipity** due to the absence of relevant unexpected items in top-N lists.
