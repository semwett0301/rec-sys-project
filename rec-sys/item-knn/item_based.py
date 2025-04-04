import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class ItemBasedRecommender:
    def __init__(self, n_neighbors):
        """
        Initialize the recommender system
        
        Args:
            n_neighbors: Number of similar items to consider for predictions
        """
        self.n_neighbors = n_neighbors
        self.ratings_matrix = None
        self.item_similarity_matrix = None
        self.item_neighbors = {}
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.mean_ratings = None
        self.normalized_ratings = None
        self.user_means = None
        
    def normalize(self, ratings):
        """Normalize ratings by subtracting user mean"""
        # Compute mean rating for each user
        mean = ratings.groupby(by='user_id', as_index=False)['rating'].mean()
        norm_ratings = pd.merge(ratings, mean, suffixes=('','_mean'), on='user_id')
        
        # Normalize each rating by subtracting the mean rating of the corresponding user
        norm_ratings['norm_rating'] = norm_ratings['rating'] - norm_ratings['rating_mean']
        return mean.to_numpy()[:, 1], norm_ratings
        
    def load_data(self, train_data, val_data, test_data):
        """
        Process training, validation, and test data
        
        Args:
            train_data: Training data DataFrame
            val_data: Validation data DataFrame
            test_data: Test data DataFrame
        """
        print("Processing datasets...")
        
        # Create user and item mappings
        unique_users = sorted(train_data['user_id'].unique())
        unique_items = sorted(train_data['item_id'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Convert IDs to indices
        train_data['user_idx'] = train_data['user_id'].map(self.user_mapping)
        train_data['item_idx'] = train_data['item_id'].map(self.item_mapping)
        
        # Create sparse matrix for normalized ratings
        self.mean_ratings, self.normalized_ratings = self.normalize(train_data)
        self.ratings_matrix = csr_matrix((
            self.normalized_ratings['norm_rating'],
            (self.normalized_ratings['user_idx'],
             self.normalized_ratings['item_idx'])
        ), shape=(len(unique_users), len(unique_items)))
        
        return train_data, val_data, test_data
    
    def train(self):
        """Train the recommender system"""
        if self.ratings_matrix is None:
            raise ValueError("Data not loaded. Please call load_data() before training.")
            
        print("\nStarting training process...")
        
        # Create and fit the kNN model
        model = NearestNeighbors(metric='cosine', n_neighbors=self.n_neighbors+1, algorithm='brute')
        model.fit(self.ratings_matrix.T) # Transpose to get item-item similarities
        
        # Get similarities and neighbors
        similarities, neighbors = model.kneighbors(self.ratings_matrix.T)
        
        # Store similarities and neighbors (excluding self)
        self.item_similarity_matrix = similarities[:, 1:]
        self.item_neighbors = neighbors[:, 1:]
        
        print("\nTraining completed:")
        print(f"Similarity matrix shape: {self.item_similarity_matrix.shape}")
        print(f"Neighbors matrix shape: {self.item_neighbors.shape}")
    
    def predict(self, user_id, item_id):
        """Make rating prediction for user on item"""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return None
            
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        # Get items similar to item_id with their similarities
        item_neighbors = self.item_neighbors[item_idx]
        item_similarities = self.item_similarity_matrix[item_idx]
        
        # Get user's ratings
        user_ratings = self.ratings_matrix[user_idx].toarray().flatten()
        
        # Get similar items rated by the user
        similar_ratings = user_ratings[item_neighbors]
        valid_mask = similar_ratings != 0
        
        if np.sum(valid_mask) == 0:
            return self.mean_ratings[user_idx]
            
        valid_ratings = similar_ratings[valid_mask]
        valid_similarities = item_similarities[valid_mask]
        
        # Calculate weighted average
        weighted_sum = np.sum(valid_ratings * valid_similarities)
        similarity_sum = np.sum(np.abs(valid_similarities))
        
        if similarity_sum == 0:
            return self.mean_ratings[user_idx]
            
        predicted_rating = weighted_sum / similarity_sum
        # Add back the user's mean rating
        predicted_rating += self.mean_ratings[user_idx]
        
        # Clip to rating range (1-5)
        return np.clip(predicted_rating, 1, 5)
    
    def recommend(self, user_id, n_recommendations=5):
        """Generate recommendations for a user"""
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        user_ratings = self.ratings_matrix[user_idx].toarray().flatten()
        
        # Get items the user has rated
        rated_items = np.where(user_ratings != 0)[0]
        
        # Calculate predicted ratings for unrated items
        predictions = {}
        for item_id in range(self.ratings_matrix.shape[1]):
            if item_id not in rated_items:
                predicted_rating = self.predict(user_id, self.reverse_item_mapping[item_id])
                if predicted_rating is not None:
                    predictions[self.reverse_item_mapping[item_id]] = predicted_rating
        
        # Get random recommendations
        if len(predictions) > n_recommendations:
            random_items = np.random.choice(list(predictions.keys()), size=n_recommendations, replace=False)
            return [(item_id, predictions[item_id]) for item_id in random_items]
        else:
            return list(predictions.items())
    
    def evaluate(self, train_data, val_data, test_data=None):
        """
        Evaluate the recommender system using train and validation sets.
        Test set evaluation is handled separately to prevent data leakage.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Optional test data for final evaluation only
        """
        print("\nStarting evaluation...")
        
        # Evaluate on training set
        train_predictions = []
        train_actuals = []
        train_total_pairs = len(train_data)
        train_predicted_pairs = 0
        
        print("\nTraining Set Evaluation:")
        print(f"{'='*50}")
        
        for _, row in train_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            predicted_rating = self.predict(user_id, item_id)
            if predicted_rating is not None:
                train_predictions.append(predicted_rating)
                train_actuals.append(actual_rating)
                train_predicted_pairs += 1
        
        if train_predictions:
            train_rmse = np.sqrt(np.mean((np.array(train_actuals) - np.array(train_predictions)) ** 2))
            print(f"Total training pairs: {train_total_pairs}")
            print(f"Successfully predicted pairs: {train_predicted_pairs}")
            print(f"Training prediction coverage: {(train_predicted_pairs/train_total_pairs)*100:.2f}%")
            print(f"Training RMSE: {train_rmse:.4f}")
        
        # Evaluate on validation set
        val_predictions = []
        val_actuals = []
        val_total_pairs = len(val_data)
        val_predicted_pairs = 0
        
        print("\nValidation Set Evaluation:")
        print(f"{'='*50}")
        
        for _, row in val_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            predicted_rating = self.predict(user_id, item_id)
            if predicted_rating is not None:
                val_predictions.append(predicted_rating)
                val_actuals.append(actual_rating)
                val_predicted_pairs += 1
        
        if val_predictions:
            val_rmse = np.sqrt(np.mean((np.array(val_actuals) - np.array(val_predictions)) ** 2))
            print(f"Total validation pairs: {val_total_pairs}")
            print(f"Successfully predicted pairs: {val_predicted_pairs}")
            print(f"Validation prediction coverage: {(val_predicted_pairs/val_total_pairs)*100:.2f}%")
            print(f"Validation RMSE: {val_rmse:.4f}")
            
            # Analyze overfitting
            print("\nOverfitting Analysis:")
            print(f"{'='*50}")
            if train_rmse < val_rmse:
                print("Potential overfitting detected:")
                print(f"Training RMSE ({train_rmse:.4f}) is lower than Validation RMSE ({val_rmse:.4f})")
                print(f"Overfitting gap: {val_rmse - train_rmse:.4f}")
            else:
                print("No significant overfitting detected")
            
            # Only evaluate test set if provided (for final evaluation)
            if test_data is not None:
                print("\nFinal Test Set Evaluation:")
                print(f"{'='*50}")
                test_predictions = []
                test_actuals = []
                test_total_pairs = len(test_data)
                test_predicted_pairs = 0
                
                for _, row in test_data.iterrows():
                    user_id = row['user_id']
                    item_id = row['item_id']
                    actual_rating = row['rating']
                    
                    predicted_rating = self.predict(user_id, item_id)
                    if predicted_rating is not None:
                        test_predictions.append(predicted_rating)
                        test_actuals.append(actual_rating)
                        test_predicted_pairs += 1
                
                if test_predictions:
                    test_rmse = np.sqrt(np.mean((np.array(test_actuals) - np.array(test_predictions)) ** 2))
                    print(f"Total test pairs: {test_total_pairs}")
                    print(f"Successfully predicted pairs: {test_predicted_pairs}")
                    print(f"Test prediction coverage: {(test_predicted_pairs/test_total_pairs)*100:.2f}%")
                    print(f"Test RMSE: {test_rmse:.4f}")
                    return train_rmse, val_rmse, test_rmse
            
            return train_rmse, val_rmse, None
        return None, None, None

def main():
    # Define data paths
    train_data_path = "/Users/masoud/Downloads/MovieLens_1M_Dataset/train.csv"
    test_data_path = "/Users/masoud/Downloads/MovieLens_1M_Dataset/test.csv"
    
    # First run K tuning to get optimal K
    print("\nRunning K tuning analysis...")
    from tune_k import tune_k, create_data_splits
    
    # Create data splits
    print("\nCreating data splits...")
    train_data, val_data, test_data = create_data_splits(train_data_path, test_data_path)
    
    # Run K tuning with the splits
    best_k, elbow_point, results_df, train_data, val_data, test_data = tune_k(train_data_path, test_data_path)
    
    # Initialize recommender with elbow point K
    print("\nInitializing recommender with optimal K...")
    recommender = ItemBasedRecommender(n_neighbors=elbow_point)
    
    # Process the data
    print("\nProcessing data...")
    recommender.load_data(train_data, val_data, test_data)
    
    # Train the model
    print("\nTraining the model...")
    recommender.train()
    
    # Evaluate the model
    print("\nEvaluating the model...")
    train_rmse, val_rmse, test_rmse = recommender.evaluate(train_data, val_data, test_data)
    
    # Print results
    print("\nModel Performance:")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Generate recommendations for a sample user
    print("\nGenerating sample recommendations...")
    sample_user = train_data['user_id'].iloc[0]
    recommendations = recommender.recommend(sample_user, n_recommendations=5)
    print(f"\nTop 5 recommendations for user {sample_user}:")
    print(recommendations)

if __name__ == "__main__":
    main() 