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

### `rating.py`
- Functions to:
  - Convert rating DataFrames to sparse matrices
  - Map original user/item IDs to internal matrix indices
  - Handle both explicit and implicit feedback formats

---

## 🧪 Usage Example

```python
from common.metrics import TestMetricsCalculator
from common.rating import prepare_rating_matrix

# Prepare test matrix from DataFrame
test_matrix, user_map, item_map = prepare_rating_matrix(rating_df)

# Evaluate recommendation model
metrics = TestMetricsCalculator(test_matrix, model, user_map, item_map, n=10)
print(metrics.calculate_item_space_coverage())
print(metrics.calculate_normalized_item_deg())
```

---

## ✅ Designed For

- Matrix Factorization (e.g., SVD, SVD++)
- Content-Based Filtering
- Any recommender system using top-N recommendation format and sparse matrix ratings