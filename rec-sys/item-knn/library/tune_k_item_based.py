import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from item_based import ItemBasedRecommender
from metrics import RmseCalculator


def plot_k_vs_rmse(results_df):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k'], results_df['val_rmse'], marker='o')
    plt.title('Validation RMSE vs. K in Item-Based Collaborative Filtering')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Validation RMSE')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def tune_k(train_matrix, val_matrix, test_matrix, user_mapping, item_mapping, k_values=range(5, 101, 5)):
    print("\nStarting K tuning for Item-Based Collaborative Filtering...")
    results = []

    for k in tqdm(k_values, desc="Testing different K values"):
        recommender = ItemBasedRecommender(k=k)
        recommender.fit(train_matrix, user_mapping, item_mapping)

        metric_calculator = RmseCalculator(
            matrix=val_matrix,
            model=recommender,
            idx_to_user_id=user_mapping['idx_to_id'],
            idx_to_item_id=item_mapping['idx_to_id']
        )
        val_rmse = metric_calculator.calculate_rmse()

        results.append({
            'k': k,
            'val_rmse': val_rmse
        })

    results_df = pd.DataFrame(results)

    plot_k_vs_rmse(results_df)

    elbow_k = find_elbow_point(results_df)
    best_k = results_df.loc[results_df['val_rmse'].idxmin(), 'k']

    print("\nTuning Complete")
    print(f"Best K (lowest validation RMSE): {best_k}")
    print(f"Elbow Point K: {elbow_k}")



    return best_k, elbow_k, results_df, train_matrix, val_matrix, test_matrix


def find_elbow_point(results_df):
    results_df['rmse_change'] = results_df['val_rmse'].diff()
    results_df['rmse_change_rate'] = results_df['rmse_change'] / results_df['k'].diff()

    results_df['rate_of_change'] = results_df['rmse_change_rate'].diff()
    elbow1 = results_df.loc[
        results_df['rate_of_change'].abs() < results_df['rate_of_change'].abs().mean() * 0.1, 'k'].min()

    results_df['improvement_percentage'] = (results_df['rmse_change'] / results_df['val_rmse']) * 100
    elbow2 = results_df[results_df['improvement_percentage'] < 0.5]['k'].min()

    window_size = 5
    results_df['ma_rate'] = results_df['rmse_change_rate'].rolling(window=window_size).mean()
    elbow3 = results_df[results_df['ma_rate'] < results_df['ma_rate'].mean() * 0.1]['k'].min()

    elbows = [elbow1, elbow2, elbow3]
    elbows = [k for k in elbows if not pd.isna(k)]

    return int(np.median(elbows)) if elbows else results_df['k'].iloc[0]
