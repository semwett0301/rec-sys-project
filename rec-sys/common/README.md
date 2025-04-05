# Common Module for Recommendation Systems

This module provides reusable utilities for recommendation system evaluation and rating matrix preparation. It is shared across different algorithms such as matrix factorization and content-based filtering.

---

## 📁 Structure

```
common/
├── metrics.py          # Unified evaluation suite (diversity, novelty, serendipity, etc.)
├── rating.py           # Utilities for rating matrix handling and preparation
├── __init__.py         # Module initializer
```

---

## 🔧 Features

### `metrics.py`
- Implements key metrics:
  - **Relevance** (Recovery)
  - **Diversity** (Normalized AggDiv)
  - **Coverage** (Item Space Coverage)
  - **Novelty** (Normalized ItemDeg)
  - **Serendipity** (with/without relevance)
- Works with recommendation dictionaries and sparse matrices
- Fully compatible with top-N recommendation outputs
- Contain the classes for RMSE and classification-based metrics calculations

### `rating.py`
- Functions to:
  - Convert rating DataFrames to sparse matrices
  - Map original user/item IDs to internal matrix indices
  - Handle both explicit and implicit feedback formats



## ✅ Designed For

- Matrix Factorization (e.g., SVD, SVD++)
- Content-Based Filtering
- Any recommender system using top-N recommendation format and sparse matrix ratings