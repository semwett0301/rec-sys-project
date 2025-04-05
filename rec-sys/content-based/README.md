# Content-Based Recommendation System

This folder contains an implementation of a **Content-Based Filtering (CBF)** recommendation system. Unlike collaborative methods, this system recommends items based on user profiles and item features (e.g., genres, categories, keywords). It includes both **user-based** and **item-based** content KNN approaches.

---

## 📁 Project Structure

```
content-based/
├── library/
│   ├── item_knn_content.py     # Item-based content filtering model
│   ├── user_knn_content.py     # User-based content filtering model
│   ├── k_tuner.py              # Hyperparameter tuning using k-NN variants
│   └── __init__.py
├── movie-lens.ipynb            # Evaluation notebook on MovieLens dataset
├── yelp.ipynb                  # Evaluation notebook on Yelp dataset
└── README.md                   # Project description and usage
```

---

## 🧠 Overview

This recommender system uses **content-based filtering** to recommend items similar to those the user has liked before, based on item attributes and user preferences. Key characteristics:

- Uses vector representations for item/user features
- Dot product to calculate similarity scores
- Top-N recommendation generation
- Tested in the **cold start**

---

## ✨ Features

- Supports both **item-based** and **user-based** filtering
- Efficient vectorized similarity computation
- Flexible k-NN configuration and tuning
- Easy integration with rating or binary interaction matrices
- Support for evaluation across relevance, diversity, novelty, and serendipity

---



## 📈 Final Model Comparison

### 🔎 Relevance Metrics

| Metric   | Yelp (ItemKNN) | Yelp (UserKNN) | MovieLens (ItemKNN) | MovieLens (UserKNN) | Conclusion |
|----------|----------------|----------------|----------------------|----------------------|------------|
| Recovery | 0.5439         | 0.3733         | 0.4857               | 0.5214               | ItemKNN performs better on Yelp, UserKNN better on MovieLens |

---

### 🌐 Diversity & Coverage Metrics

| Metric                           | Yelp (ItemKNN) | Yelp (UserKNN) | MovieLens (ItemKNN) | MovieLens (UserKNN) | Conclusion |
|----------------------------------|----------------|----------------|----------------------|----------------------|------------|
| Normalized AggDiv (diversity)    | 0.3780         | 0.1126         | 0.1109               | 0.0251               | Highest diversity with ItemKNN on Yelp |
| Normalized AggDiv (coverage)     | 0.6317         | 0.1882         | 0.2937               | 0.0665               | ItemKNN on Yelp also shows best catalog coverage |
| Item Space Coverage              | 56.5810        | 49.3110        | 24.6850              | 22.4520              | ItemKNN on Yelp leads in entropy-based item usage |

---

### 🌟 Novelty & Serendipity Metrics

| Metric                          | Yelp (ItemKNN) | Yelp (UserKNN) | MovieLens (ItemKNN) | MovieLens (UserKNN) | Conclusion |
|----------------------------------|----------------|----------------|----------------------|----------------------|------------|
| Normalized ItemDeg              | 0.7090         | 0.6280         | 0.6220               | 0.4740               | Yelp-ItemKNN provides the most novel items |
| Unexpectedness (no relevance)   | 0.6710         | 0.5870         | 0.5880               | 0.4370               | Consistently stronger exploration on Yelp |
| Serendipity (with relevance)    | 0.0            | 0.0            | 0.0                  | 0.0                  | All models fail to recommend both unexpected and relevant items |

---

## ✅ Final Summary

- **Yelp + ItemKNN** delivers the strongest results across most metrics, especially in **coverage**, **diversity**, and **novelty**.
- **MovieLens + UserKNN** slightly edges out others in **relevance**, but falls short in exploration and coverage.
- All models currently fail to generate **serendipitous recommendations**, indicating room for improvement in combining novelty with relevance.
- The content-based approach is particularly effective on datasets with **rich item features** like Yelp.