import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from metrics import RankingPredictable


class UserKnnContent(RankingPredictable):
    def __init__(
        self,
        rating_matrix: csr_matrix,
        similarity_features: pd.DataFrame,
        user_mapping: dict[str, dict],
        item_mapping: dict[str, dict],
        k: int = 20
    ):
        """
        Initializes the UserKnnContent-based recommender using user content features.

        :param rating_matrix: Sparse user-item interaction matrix.
        :param similarity_features: DataFrame of user content features (e.g., demographics, preferences).
        :param user_mapping: Dictionary with 'id_to_idx' and 'idx_to_id' for users.
        :param item_mapping: Dictionary with 'id_to_idx' and 'idx_to_id' for items.
        :param k: Number of most similar users (neighbors) to use for recommendations.
        """
        self._rating_matrix = rating_matrix
        self._similarity_features = similarity_features.values
        self._k = k

        self._user_mapping = user_mapping
        self._item_mapping = item_mapping

        # Precompute the user-user similarity matrix using content features
        self._similarity_matrix = self._calculate_similarity()

    def _calculate_similarity(self):
        """
        Computes cosine similarity between all user pairs based on their feature vectors.

        :return: A 2D numpy array representing the user-user similarity matrix.
        """
        # Dot product between user feature vectors
        dot_product = self._similarity_features @ self._similarity_features.T

        # Outer product of norms
        norms = np.linalg.norm(self._similarity_features, axis=1, keepdims=True)
        norm_product = norms @ norms.T

        # Element-wise cosine similarity (safe division)
        similarity_matrix = np.divide(dot_product, norm_product, where=norm_product != 0)

        # Zero out self-similarity
        np.fill_diagonal(similarity_matrix, 0)

        return similarity_matrix

    def predict(self, user_id: str, top_n: int = 10) -> list[str]:
        """
        Recommends top-N items for a given user based on K-nearest neighbors
        found using user content similarity.

        :param user_id: External user ID to generate recommendations for.
        :param top_n: Number of top items to return.
        :return: List of recommended item IDs.
        :raises ValueError: If the user ID is not found in the mapping.
        """
        if user_id not in self._user_mapping['id_to_idx']:
            raise ValueError(f"User {user_id} not found.")

        user_idx = self._user_mapping['id_to_idx'][user_id]

        # Retrieve similarity scores for the target user
        sim_vector = self._similarity_matrix[user_idx]

        # Get indices of top-K most similar users
        top_k_user_indices = np.argsort(sim_vector)[-self._k:]

        # Initialize scores for each item
        scores = np.zeros(self._rating_matrix.shape[1])

        # Aggregate weighted item ratings from similar users
        for sim_user_idx in top_k_user_indices:
            sim_score = sim_vector[sim_user_idx]
            user_ratings = self._rating_matrix[sim_user_idx].toarray().flatten()
            scores += sim_score * user_ratings

        # Sort scores in descending order
        recommended_indices = np.argsort(scores)[::-1]

        # Select top-N items and convert to external item IDs
        top_indices = recommended_indices[:top_n]
        top_ids = [self._item_mapping['idx_to_id'][i] for i in top_indices]

        return top_ids
