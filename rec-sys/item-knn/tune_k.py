import numpy as np
import pandas as pd
from item_based import ItemBasedRecommender
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def create_data_splits(train_path, test_path, val_ratio=0.15, random_state=42):
    """
    Create validation split from training data, keeping test set separate
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        val_ratio: Ratio of training data to use for validation
        random_state: Random seed for reproducibility
    
    Returns:
        train_data, val_data, test_data: DataFrames containing the splits
    """
    print("Loading and splitting data...")
    
    # Load the data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Rename columns to match expected format
    train_data = train_data.rename(columns={
        'movie_id': 'item_id',
        'Rating': 'rating'
    })
    test_data = test_data.rename(columns={
        'movie_id': 'item_id',
        'Rating': 'rating'
    })
    
    # Split training data into train and validation
    train_data, val_data = train_test_split(
        train_data, 
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_data['user_id']  # Stratify by user to maintain user distribution
    )
    
    # Print split statistics
    print("\nData Split Statistics:")
    print(f"{'='*50}")
    print(f"Training data points: {len(train_data)}")
    print(f"Validation set size: {len(val_data)} ({len(val_data)/(len(train_data)+len(val_data))*100:.1f}% of training data)")
    print(f"Test set size: {len(test_data)}")
    print(f"\nUnique users in each set:")
    print(f"Training users: {train_data['user_id'].nunique()}")
    print(f"Validation users: {val_data['user_id'].nunique()}")
    print(f"Test users: {test_data['user_id'].nunique()}")
    print(f"{'='*50}")
    
    return train_data, val_data, test_data

def plot_progress(results_df):
    """
    Plot the tuning progress with bar chart showing both training and validation RMSE
    """
    plt.figure(figsize=(12, 6))
    
    # Create bar chart for validation RMSE
    val_bars = plt.bar(results_df['k'], results_df['val_rmse'], alpha=0.6, label='Validation RMSE')
    
    # Add line plots for both training and validation RMSE
    plt.plot(results_df['k'], results_df['val_rmse'], 'r-', linewidth=2, label='Validation Trend')
    plt.plot(results_df['k'], results_df['train_rmse'], 'g-', linewidth=2, label='Training Trend')
    
    # Highlight best K
    best_k = results_df.loc[results_df['val_rmse'].idxmin(), 'k']
    best_rmse = results_df.loc[results_df['val_rmse'].idxmin(), 'val_rmse']
    plt.axvline(x=best_k, color='g', linestyle='--', label=f'Best K = {best_k}')
    
    # Add RMSE value labels on top of bars
    for bar in val_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.title('Tuning Progress: Training and Validation RMSE vs K Value')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('tuning_progress.png')
    plt.close()

def plot_k_selection_analysis(results_df, elbow_point):
    """
    Plot detailed analysis of K selection methods
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Calculate metrics for all methods
    results_df['rmse_change'] = results_df['val_rmse'].diff()
    results_df['rmse_change_rate'] = results_df['rmse_change'] / results_df['k'].diff()
    results_df['rate_of_change'] = results_df['rmse_change_rate'].diff()
    results_df['improvement_percentage'] = (results_df['rmse_change'] / results_df['val_rmse']) * 100
    window_size = 5
    results_df['ma_rate'] = results_df['rmse_change_rate'].rolling(window=window_size).mean()
    
    # Find elbow points for each method
    elbow1 = results_df.loc[results_df['rate_of_change'].abs() < results_df['rate_of_change'].abs().mean() * 0.1, 'k'].min()
    elbow2 = results_df[results_df['improvement_percentage'] < 0.5]['k'].min()
    elbow3 = results_df[results_df['ma_rate'] < results_df['ma_rate'].mean() * 0.1]['k'].min()
    
    # Plot 1: RMSE vs K with all elbow points
    ax1.plot(results_df['k'], results_df['val_rmse'], 'b-', marker='o', label='Validation RMSE')
    ax1.plot(results_df['k'], results_df['train_rmse'], 'g-', marker='o', label='Training RMSE')
    ax1.axvline(x=elbow1, color='r', linestyle='--', label=f'Method 1 (K={elbow1})')
    ax1.axvline(x=elbow2, color='m', linestyle='--', label=f'Method 2 (K={elbow2})')
    ax1.axvline(x=elbow3, color='y', linestyle='--', label=f'Method 3 (K={elbow3})')
    best_k = results_df.loc[results_df['val_rmse'].idxmin(), 'k']
    ax1.axvline(x=best_k, color='g', linestyle='--', label=f'Best RMSE (K={best_k})')
    ax1.set_title('RMSE vs K Value (All Methods)')
    ax1.set_xlabel('K (Number of Neighbors)')
    ax1.set_ylabel('RMSE')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Second Derivative Method
    ax2.plot(results_df['k'], results_df['rate_of_change'].abs(), 'r-', marker='o', label='Rate of Change')
    ax2.axhline(y=results_df['rate_of_change'].abs().mean() * 0.1, color='r', linestyle='--', label='Threshold')
    ax2.axvline(x=elbow1, color='r', linestyle='--', label=f'Elbow Point (K={elbow1})')
    ax2.set_title('Method 1: Second Derivative')
    ax2.set_xlabel('K (Number of Neighbors)')
    ax2.set_ylabel('Absolute Rate of Change')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Percentage Improvement Method
    ax3.plot(results_df['k'], results_df['improvement_percentage'], 'g-', marker='o', label='Improvement %')
    ax3.axhline(y=0.5, color='r', linestyle='--', label='0.5% Threshold')
    ax3.axvline(x=elbow2, color='m', linestyle='--', label=f'Elbow Point (K={elbow2})')
    ax3.set_title('Method 2: Percentage Improvement')
    ax3.set_xlabel('K (Number of Neighbors)')
    ax3.set_ylabel('Improvement (%)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Moving Average Method
    ax4.plot(results_df['k'], results_df['ma_rate'], 'm-', marker='o', label='Moving Average')
    ax4.axhline(y=results_df['ma_rate'].mean() * 0.1, color='r', linestyle='--', label='Threshold')
    ax4.axvline(x=elbow3, color='y', linestyle='--', label=f'Elbow Point (K={elbow3})')
    ax4.set_title('Method 3: Moving Average')
    ax4.set_xlabel('K (Number of Neighbors)')
    ax4.set_ylabel('Moving Average')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('k_selection_analysis.png')
    plt.close()
    
    # Print detailed analysis of each method
    print("\nDetailed Elbow Point Analysis:")
    print("=" * 50)
    print(f"Method 1 (Second Derivative):")
    print(f"- Elbow Point: K = {elbow1}")
    print(f"- RMSE at elbow: {results_df[results_df['k'] == elbow1]['val_rmse'].iloc[0]:.4f}")
    print(f"- Rate of change: {results_df[results_df['k'] == elbow1]['rate_of_change'].iloc[0]:.4f}")
    print(f"\nMethod 2 (Percentage Improvement):")
    print(f"- Elbow Point: K = {elbow2}")
    print(f"- RMSE at elbow: {results_df[results_df['k'] == elbow2]['val_rmse'].iloc[0]:.4f}")
    print(f"- Improvement %: {results_df[results_df['k'] == elbow2]['improvement_percentage'].iloc[0]:.4f}")
    print(f"\nMethod 3 (Moving Average):")
    print(f"- Elbow Point: K = {elbow3}")
    print(f"- RMSE at elbow: {results_df[results_df['k'] == elbow3]['val_rmse'].iloc[0]:.4f}")
    print(f"- Moving Average: {results_df[results_df['k'] == elbow3]['ma_rate'].iloc[0]:.4f}")
    print("=" * 50)

def find_elbow_point(results_df):
    """
    Find the elbow point using multiple methods
    """
    print("\nElbow Point Analysis:")
    print("=" * 50)
    
    # Calculate rate of change in RMSE
    results_df['rmse_change'] = results_df['val_rmse'].diff()
    results_df['rmse_change_rate'] = results_df['rmse_change'] / results_df['k'].diff()
    
    # Method 1: Second derivative method
    results_df['rate_of_change'] = results_df['rmse_change_rate'].diff()
    elbow1 = results_df.loc[results_df['rate_of_change'].abs() < results_df['rate_of_change'].abs().mean() * 0.1, 'k'].min()
    print(f"Method 1 (Second Derivative) suggests K = {elbow1}")
    
    # Method 2: Percentage improvement method
    results_df['improvement_percentage'] = (results_df['rmse_change'] / results_df['val_rmse']) * 100
    elbow2 = results_df[results_df['improvement_percentage'] < 0.5]['k'].min()
    print(f"Method 2 (Percentage Improvement) suggests K = {elbow2}")
    
    # Method 3: Moving average method
    window_size = 5
    results_df['ma_rate'] = results_df['rmse_change_rate'].rolling(window=window_size).mean()
    elbow3 = results_df[results_df['ma_rate'] < results_df['ma_rate'].mean() * 0.1]['k'].min()
    print(f"Method 3 (Moving Average) suggests K = {elbow3}")
    
    # Combine methods and use median
    elbow_points = [elbow1, elbow2, elbow3]
    elbow_points = [x for x in elbow_points if not pd.isna(x)]
    final_elbow = int(np.median(elbow_points))
    print(f"\nFinal elbow point (median of all methods): K = {final_elbow}")
    print(f"RMSE at elbow point: {results_df[results_df['k'] == final_elbow]['val_rmse'].iloc[0]:.4f}")
    print("=" * 50)
    
    return final_elbow

def tune_k(train_data_path, test_data_path, k_values=range(5, 101, 5)):
    """
    Tune the K hyperparameter for item-based collaborative filtering
    
    Process:
    1. Create train/val splits from training data
    2. For each K:
       - Train model on training data
       - Evaluate on both training and validation sets
    3. Find best K using multiple methods
    4. Return best K and splits for final evaluation
    """
    print("Starting K hyperparameter tuning...")
    results = []
    
    # Create data splits (validation from training)
    print("\nCreating train/validation splits...")
    train_data, val_data, test_data = create_data_splits(train_data_path, test_data_path)
    
    # Create progress bar
    pbar = tqdm(k_values, desc="Testing K values", unit="K")
    
    # Test different K values
    for k in pbar:
        # Initialize recommender with current K
        recommender = ItemBasedRecommender(n_neighbors=k)
        
        # Process the data
        recommender.load_data(train_data, val_data, test_data)
        
        # Train the model on training data
        recommender.train()
        
        # Evaluate on both training and validation sets
        train_rmse, val_rmse, _ = recommender.evaluate(train_data, val_data)
        
        if val_rmse is not None:
            results.append({
                'k': k,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse
            })
            pbar.set_postfix({
                'train_rmse': f'{train_rmse:.4f}',
                'val_rmse': f'{val_rmse:.4f}'
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot tuning progress
    plot_progress(results_df)
    
    # Find elbow point using multiple methods
    elbow_point = find_elbow_point(results_df)
    
    # Plot detailed analysis
    plot_k_selection_analysis(results_df, elbow_point)
    
    # Find best K based on validation RMSE
    best_k = results_df.loc[results_df['val_rmse'].idxmin(), 'k']
    
    print("\nK Selection Results:")
    print("=" * 50)
    print(f"Best K (lowest validation RMSE): {best_k}")
    print(f"Elbow Point K: {elbow_point}")
    print(f"Training RMSE at best K: {results_df.loc[results_df['k'] == best_k, 'train_rmse'].iloc[0]:.4f}")
    print(f"Validation RMSE at best K: {results_df.loc[results_df['k'] == best_k, 'val_rmse'].iloc[0]:.4f}")
    print(f"Training RMSE at elbow K: {results_df.loc[results_df['k'] == elbow_point, 'train_rmse'].iloc[0]:.4f}")
    print(f"Validation RMSE at elbow K: {results_df.loc[results_df['k'] == elbow_point, 'val_rmse'].iloc[0]:.4f}")
    print("=" * 50)
    
    return best_k, elbow_point, results_df, train_data, val_data, test_data

if __name__ == "__main__":
    # Define data paths
    train_data_path = "/Users/masoud/Downloads/MovieLens_1M_Dataset/train.csv"
    test_data_path = "/Users/masoud/Downloads/MovieLens_1M_Dataset/test.csv"
    
    # Run K tuning
    best_k, elbow_point, results_df, train_data, val_data, test_data = tune_k(train_data_path, test_data_path)
    
    # Train final model with elbow point K
    print("\nTraining final model with elbow point K...")
    final_recommender = ItemBasedRecommender(n_neighbors=elbow_point)
    final_recommender.load_data(train_data, val_data, test_data)
    final_recommender.train()
    
    # Final evaluation
    print("\nFinal Model Evaluation:")
    train_rmse, val_rmse, test_rmse = final_recommender.evaluate(train_data, val_data, test_data)
    print(f"Final Training RMSE: {train_rmse:.4f}")
    print(f"Final Validation RMSE: {val_rmse:.4f}")
    print(f"Final Test RMSE: {test_rmse:.4f}") 