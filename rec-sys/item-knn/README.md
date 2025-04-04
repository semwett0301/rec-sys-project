# Item-Based Collaborative Filtering Recommender System

This project implements an item-based collaborative filtering recommender system using the MovieLens 1M dataset. The system uses k-nearest neighbors to find similar items and make recommendations.

## Features

- Item-based collaborative filtering
- K-nearest neighbors similarity
- Mean-centering normalization
- Comprehensive K value tuning with multiple elbow methods
- Detailed visualization of model performance

## Plots Generated

The system generates four plots to analyze model performance and K value selection:

1. **Tuning Progress** (`tuning_progress.png`)
   - Bar chart showing RMSE values for each K
   - Trend line showing overall pattern
   - Vertical line marking the best K value
   - Purpose: Visualize overall tuning progress and identify optimal K

2. **K Selection Analysis** (`k_selection_analysis.png`)
   - 2x2 grid of plots showing different elbow methods:
     - Top Left: RMSE vs K with all elbow points
     - Top Right: Second Derivative Method
     - Bottom Left: Percentage Improvement Method
     - Bottom Right: Moving Average Method
   - Purpose: Compare different methods for finding the elbow point

3. **Learning Curves** (`learning_curves.png`)
   - Training and validation RMSE over different K values
   - Shows model performance on both sets
   - Purpose: Analyze model learning behavior

4. **Overfitting Gap** (`overfitting_gap.png`)
   - Difference between validation and training RMSE
   - Shows how much the model overfits at different K values
   - Purpose: Analyze model generalization

## Usage

1. Ensure you have the required data files:
   - `/Users/masoud/Downloads/MovieLens_1M_Dataset/train.csv`
   - `/Users/masoud/Downloads/MovieLens_1M_Dataset/test.csv`

2. Run the main script:
   ```bash
   python item_based.py
   ```
   This will:
   - Run K tuning analysis
   - Generate all four plots
   - Train the final model with optimal K
   - Show model performance metrics
   - Generate sample recommendations

## Requirements

- Python 3.x
- Required packages:
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - matplotlib
  - tqdm

## Code Structure

- `item_based.py`: Main recommender system implementation
- `tune_k.py`: K value tuning and visualization

## Model Details

The recommender system:
1. Normalizes ratings by subtracting user means
2. Computes item-item similarity using cosine similarity
3. Uses k-nearest neighbors to find similar items
4. Makes predictions based on weighted average of similar items' ratings
5. Evaluates performance using RMSE metric

## K Value Selection

The system uses three methods to find the optimal K value:
1. Second Derivative Method: Finds where the rate of change levels off
2. Percentage Improvement Method: Finds where improvements become minimal
3. Moving Average Method: Smoothes the rate of change to find the elbow

The final K value is selected based on the consensus of these methods.

## Output

The system provides:
1. K tuning results showing RMSE for different K values
2. Best K value found
3. Overall dataset statistics
4. Detailed user analysis including:
   - Total items rated
   - Average rating
   - Rating range
   - Rating distribution
   - Predictions for test set items

## How It Works

1. **Data Loading**:
   - Loads training and test data
   - Creates user and item mappings
   - Normalizes ratings

2. **Training**:
   - Computes item-item similarity matrix using cosine similarity
   - Finds K nearest neighbors for each item
   - Stores similarity scores and neighbor indices

3. **Prediction**:
   - Uses weighted average of similar items' ratings
   - Accounts for user's mean rating
   - Clips predictions to rating range (1-5)

4. **Evaluation**:
   - Uses RMSE to evaluate prediction accuracy
   - Considers only items in the test set

## Example Output

```
K Tuning Results:
==================================================
Best K value: 100
Best RMSE: 0.969527

All Results:
   k    rmse
   5  1.048587
  10  1.023456
  15  1.001234
  ...
100  0.969527
==================================================

Overall Dataset Statistics:
Total users: 6040
Total items: 3706
Total ratings: 1000209
Average rating: 3.58
Rating distribution:
1    6110
2    11370
3    27145
4    34174
5    15110
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 