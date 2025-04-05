# SVD++ Recommendation System

This folder contains an implementation of the SVD++ recommendation algorithm, which is an extension of the classic SVD (Singular Value Decomposition) approach that incorporates implicit feedback. The implementation is optimized for sparse matrices and includes support for both explicit and implicit user-item interactions.

## Project Structure

```
svd++/
├── library/                 # Core implementation files
│   ├── svdpp.py            # Main SVD++ algorithm implementation
│   ├── tuning.py           # Grid search and hyperparameter tuning
│   └── __init__.py
├── movie-lens.ipynb        # MovieLens dataset implementation
├── yelp.ipynb              # Yelp dataset implementation
└── README.MD               # This file
```

## Overview

SVD++ is a collaborative filtering algorithm that combines:
- Explicit feedback (user ratings)
- Implicit feedback (user interactions) - in our case, the amount of positive reviews
- User and item biases
- Latent factor modeling

The implementation uses scipy's CSR (Compressed Sparse Row) matrix format for efficient storage and computation of sparse rating matrices.

## Features

- Efficient handling of sparse rating matrices
- Support for both explicit and implicit feedback
- Fine-grained control over learning rates and regularization
- Configurable number of latent factors
- Customizable initialization parameters
- Comprehensive error handling and parameter validation
- Grid search for hyperparameter tuning
- Support for multiple datasets (MovieLens and Yelp)

## Implementation Details

### Core Components

1. **SVD++ Algorithm (`library/svdpp.py`)**
   - Matrix factorization with implicit feedback
   - Stochastic gradient descent optimization
   - Configurable learning rates and regularization
   - Efficient sparse matrix operations

2. **Hyperparameter Tuning (`library/tuning.py`)**
   - Grid search implementation
   - Cross-validation support
   - Performance metrics tracking

### Dataset Implementations

1. **MovieLens (`movie-lens.ipynb`)**
   - Dataset size: 6040 users × 3706 items
   - Rating scale: 1-5
   - Implicit feedback: Based on positive reviews
   - Sampled in `eda`

2. **Yelp (`yelp.ipynb`)**
   - Sampled dataset (10% of original)
   - Rating scale: 1-5
   - Implicit feedback: Based on positive reviews
   - Sampled in `eda`

## Usage

### Basic Usage

```python
from library.svdpp import SVDpp
from scipy.sparse import csr_matrix

# Initialize the model
model = SVDpp(
    n_factors=20,          # Number of latent factors
    reg_all=0.02,          # Global regularization term
    lr_all=0.007,          # Global learning rate
    n_epochs=20            # Number of training iterations
)

# Prepare your data
ratings = csr_matrix(...)  # Your sparse rating matrix
implicit_rating = {...}    # Dictionary of implicit feedback
user_id_to_idx = {...}     # User ID to matrix index mapping
item_id_to_idx = {...}     # Item ID to matrix index mapping

# Train the model
model.fit(ratings, implicit_rating, user_id_to_idx, item_id_to_idx)

# Make predictions
prediction = model.predict(user_id="user1", item_id="item1")
```

### Advanced Configuration

The model supports fine-grained control over various parameters:

```python
model = SVDpp(
    n_factors=20,
    reg_all=0.02,
    lr_all=0.007,
    # Individual learning rates
    lr_bu=0.005,  # User bias
    lr_bi=0.005,  # Item bias
    lr_pu=0.007,  # User factors
    lr_qi=0.007,  # Item factors
    lr_yj=0.007,  # Implicit feedback factors
    # Individual regularization terms
    reg_bu=0.02,
    reg_bi=0.02,
    reg_pu=0.02,
    reg_qi=0.02,
    reg_yj=0.02,
    n_epochs=20,
    init_mean=0,
    init_std=0.1
)
```

## Performance Metrics

### MovieLens Dataset
| Metric | Area | Value | Value Range | Meaning |
|--------|------|-------|-------------|----------|
| Recovery | Relevance | None | [0, 0.9] | How early relevant items appear in top-N recommendations |
| Normalized AggDiv (diversity) | Inter-user diversity | 0.01541 | [0, 1] | Proportion of unique items recommended across all users divided by amount of recommendations |
| Normalized AggDiv (coverage) | Coverage | 0.040816 | [0, 1] | Proportion of unique items recommended across all users divided by size of catalog |
| Item Space Coverage | Coverage | 10.087 | [0, Not defined] | Shows how many unique items and how often appears in the RLs |
| Normalized ItemDeg | Novelty | 0.29 | [0, 1] | Novelty of recommended items based on inverse (log) item popularity |
| Unexpectedness (no relevance) | Serendipity | 0.548 | [0, 1] | Proportion of items that are unexpected (less popular than average) |
| Serendipity (with relevance) | Serendipity | 0.0 | [0, 1] | Proportion of unexpected and relevant items in top-N recommendations |
| RMSE | Relevance | 0.988 | [0, 6] | Root Mean Square Error between predicted and actual ratings |

### Yelp Dataset
| Metric | Area | Value | Value Range | Meaning |
|--------|------|-------|-------------|----------|
| Recovery | Relevance | None | [0, 0.9] | How early relevant items appear in top-N recommendations |
| Normalized AggDiv (diversity) | Inter-user diversity | 0.043927 | [0, 1] | Proportion of unique items recommended across all users divided by amount of recommendations |
| Normalized AggDiv (coverage) | Coverage | 0.073406 | [0, 1] | Proportion of unique items recommended across all users divided by size of catalog |
| Item Space Coverage | Coverage | 21.383 | [0, Not defined] | Shows how many unique items and how often appears in the RLs |
| Normalized ItemDeg | Novelty | 0.654 | [0, 1] | Novelty of recommended items based on inverse (log) item popularity |
| Unexpectedness (no relevance) | Serendipity | 0.615 | [0, 1] | Proportion of items that are unexpected (less popular than average) |
| Serendipity (with relevance) | Serendipity | 0.0 | [0, 1] | Proportion of unexpected and relevant items in top-N recommendations |
| RMSE | Relevance | 1.179 | [0, 6] | Root Mean Square Error between predicted and actual ratings |

### Final comparison

#### Relevance Metrics
| Metric | MovieLens | Yelp | Conclusion |
|--------|-----------|------|------------|
| Recovery | None | None | Neither system effectively surfaces relevant items early in recommendations |
| RMSE | 0.988 | 1.179 | MovieLens provides more accurate rating predictions with 16.2% lower error |

#### Diversity & Coverage Metrics
| Metric | MovieLens | Yelp | Conclusion                                                                        |
|--------|-----------|------|-----------------------------------------------------------------------------------|
| Normalized AggDiv (diversity) | 0.01541 | 0.043927 | Yelp delivers 2.85x more diverse recommendations across users                     |
| Normalized AggDiv (coverage) | 0.040816 | 0.073406 | Yelp utilizes 79.6% more of its available catalog                                 |
| Item Space Coverage | 10.087 | 21.383 | Yelp uses more unique items with more uniform distribution in its recommendations |

#### Novelty & Serendipity Metrics
| Metric | MovieLens | Yelp | Conclusion                                                                         |
|--------|-----------|------|------------------------------------------------------------------------------------|
| Normalized ItemDeg | 0.29 | 0.654 | Yelp recommends less popular items than MovieLens                                  |
| Unexpectedness | 0.548 | 0.615 | Yelp provides 12.2% more unexpected recommendations                                |
| Serendipity | 0.0 | 0.0 | Both systems fail to deliver recommendations that are both unexpected and relevant |

#### Summary
The Yelp recommendation system significantly outperforms MovieLens in diversity, coverage, and novelty metrics, suggesting a stronger emphasis on exploration and introducing users to varied content beyond mainstream options. MovieLens holds a modest advantage in prediction accuracy. Both systems demonstrate a **cold start** state on the test sets (`Serendipity` and `Recorery` equal to 0)