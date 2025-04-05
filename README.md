# Recommendation Systems Project

This project implements and compares various recommendation system algorithms across multiple datasets. The implementation includes collaborative filtering, content-based filtering, and matrix factorization approaches.

## Project Structure

```
.
├── eda/                    # Exploratory Data Analysis
│   ├── movie-lens.ipynb    # MovieLens dataset analysis
│   ├── netflix.ipynb       # Netflix dataset analysis
│   ├── yelp.ipynb          # Yelp dataset analysis
│   └── dataset_samples/    # Sampled datasets
│
├── rec-sys/                # Recommendation System Implementations
│   ├── common/             # Common utilities and metrics
│   ├── content-based/      # Content-based filtering
│   ├── item-knn/           # Item-based collaborative filtering
│   └── svd++/              # SVD++ matrix factorization
│
├── backend/                # FastAPI backend service
│   ├── app/                # Application code
│   │   ├── api/           # API endpoints
│   │   ├── models/        # Database models
│   │   ├── services/      # Business logic
│   │   └── utils/         # Utility functions
│   └── tests/             # Backend tests
│
├── frontend/              # React frontend application
   ├── src/               # Source code
   │   ├── components/    # React components
   │   ├── types          # Types of TS
   │   ├── App.tsx        # Main component
   │   └── main.ts        # Start point
   └── public/            # Static assets
```

## Datasets

### MovieLens
- **Size**: 6040 users × 3706 items
- **Rating Scale**: 1-5 stars
- **Features**: User ratings, movie metadata
- **Sampling**: Full dataset used (no sampling needed)

### Netflix
- **Original Size**: Large-scale dataset
- **Rating Scale**: 1-5 stars
- **Features**: User ratings, movie metadata
- **Sampling**: Random sampling applied for computational efficiency

### Yelp
- **Original Size**: 900,000+ reviews
- **Rating Scale**: 1-5 stars
- **Features**: 
  - Review text
  - User ratings (stars)
  - Interaction metrics (useful, funny, cool)
  - Business categories
- **Sampling**: 10% random sample of open businesses in health category

## Metrics Description

The following metrics are used to evaluate recommendation systems:

1. **Relevance Metrics**
   - **Recovery**: Measures how early relevant items appear in top-N recommendations
   - **RMSE**: Root Mean Square Error between predicted and actual ratings

2. **Diversity & Coverage Metrics**
   - **Normalized AggDiv (diversity)**: Proportion of unique items recommended across users
   - **Normalized AggDiv (coverage)**: Proportion of catalog items recommended
   - **Item Space Coverage**: Distribution of items in recommendations

3. **Novelty & Serendipity Metrics**
   - **Normalized ItemDeg**: Novelty based on inverse item popularity
   - **Unexpectedness**: Proportion of unexpected (less popular) items
   - **Serendipity**: Proportion of unexpected and relevant items

## Dataset & Algorithm Comparison

### Relevance Metrics

| Metric   | MovieLens<br>(CF) | Yelp<br>(CF) | MovieLens<br>(ItemKNN) | MovieLens<br>(UserKNN) | Yelp<br>(ItemKNN) | Yelp<br>(UserKNN) | Netflix<br>(CF) |
|----------|-------------------|--------------|------------------------|------------------------|-------------------|-------------------|-----------------|
| Recovery | None              | None         | 0.4857                 | 0.5214                 | 0.5439            | 0.3733            | None            |
| RMSE     | 0.988             | 1.179        | Not provided           | Not provided           | Not provided      | Not provided      | 0.922           |


**Key Findings:**
- KNN-based approaches show measurable Recovery metrics unlike base CF models
- ItemKNN performs better on Yelp dataset for relevant item surfacing
- UserKNN shows stronger performance on MovieLens for Recovery
- Netflix provides the most accurate rating predictions (lowest RMSE)
- MovieLens demonstrates better prediction accuracy than Yelp with CF models

### Diversity & Coverage Metrics

| Metric                        | MovieLens<br>(CF) | Yelp<br>(CF) | MovieLens<br>(ItemKNN) | MovieLens<br>(UserKNN) | Yelp<br>(ItemKNN) | Yelp<br>(UserKNN) | Netflix<br>(CF) |
|-------------------------------|-------------------|--------------|------------------------|------------------------|-------------------|-------------------|-----------------|
| Normalized AggDiv (diversity) | 0.01541           | 0.043927     | 0.1109                 | 0.0251                 | 0.3780            | 0.1126            | 0.073533        |
| Normalized AggDiv (coverage)  | 0.040816          | 0.073406     | 0.2937                 | 0.0665                 | 0.6317            | 0.1882            | 0.573308        |
| Item Space Coverage           | 10.087            | 21.383       | 24.6850                | 22.4520                | 56.5810           | 49.3110           | 21.595          |

**Key Findings:**
- ItemKNN on Yelp significantly outperforms all other systems in diversity and coverage
- KNN approaches generally deliver better diversity than base CF models
- Netflix CF model shows impressive catalog coverage despite having lower diversity
- Content-rich datasets (Yelp) benefit more from content-based approaches for diversity
- There's a general trade-off between accuracy and diversity across most models

### Novelty & Serendipity Metrics

| Metric                        | MovieLens<br>(CF) | Yelp<br>(CF) | MovieLens<br>(ItemKNN) | MovieLens<br>(UserKNN) | Yelp<br>(ItemKNN) | Yelp<br>(UserKNN) | Netflix<br>(CF) |
|-------------------------------|-------------------|--------------|------------------------|------------------------|-------------------|-------------------|-----------------|
| Normalized ItemDeg (novelty)  | 0.29              | 0.654        | 0.6220                 | 0.4740                 | 0.7090            | 0.6280            | 0.891           |
| Unexpectedness                | 0.548             | 0.615        | 0.5880                 | 0.4370                 | 0.6710            | 0.5870            | 0.882           |
| Serendipity                   | 0.0               | 0.0          | 0.0                    | 0.0                    | 0.0               | 0.0               | 0.0             |

**Key Findings:**
- Netflix CF model achieves the highest novelty scores overall
- Yelp-ItemKNN provides the strongest performance among KNN approaches
- All systems consistently fail to deliver serendipitous recommendations
- ItemKNN generally outperforms UserKNN for novelty metrics
- CF approaches on Netflix show particularly strong novelty despite good accuracy

## Conclusions 

1. **Best Overall Performance:**
   - **Yelp + ItemKNN** delivers the strongest combined results across exploration metrics
   - **Netflix CF** shows an impressive balance between accuracy and novelty
   - **MovieLens + UserKNN** performs slightly better for relevance but underperforms in exploration

2. **Dataset Characteristics Matter:**
   - Datasets with rich item features (Yelp) benefit significantly from content-based approaches
   - Movie recommendation datasets show stronger accuracy but generally lower diversity
   
3. **Algorithm Selection:**
   - ItemKNN approaches consistently outperform UserKNN for exploration metrics
   - Traditional CF models show stronger accuracy but weaker diversity
   - The ideal algorithm depends on whether the recommendation goal prioritizes accuracy or discovery

4. **Areas for Improvement:**
   - All models fail to generate serendipitous recommendations (0.0 across the board)
   - Balancing relevance with diversity remains challenging
   - Consider hybrid approaches that can maintain accuracy while improving exploration metrics

5. **Trade-offs:**
   - Clear inverse relationship between accuracy (RMSE/Recovery) and diversity metrics
   - Systems optimized for prediction accuracy generally underperform in catalog coverage
   - The "cold start" problem remains evident in many of the evaluation scenarios

## Usage

Each recommendation system implementation includes:
- Jupyter notebooks for experimentation
- Library code for production use
- Detailed README with setup instructions
- Example usage and configuration options

See individual implementation folders for specific usage instructions.

## Work splitting (in approaches)
- `SVD++` - common
- `content-based UserKNN` | `content-based ItemKNN` - Simon
- `collaborative-based ItemKNN` - Masoud

## Final service

