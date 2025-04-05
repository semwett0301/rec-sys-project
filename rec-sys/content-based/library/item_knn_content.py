import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from metrics import RankingPredictable


class ItemKnnContent(RankingPredictable):
    def __init__(
            self,
            rating_matrix: csr_matrix,
            similarity_features: pd.DataFrame,
            user_mapping: dict[str, dict],
            item_mapping: dict[str, dict],
            k: int = 20
    ):
        """
        Initializes the ItemKnnContent-based recommender using item content features.

        :param rating_matrix: Sparse user-item interaction matrix.
        :param similarity_features: DataFrame of item content features (e.g., genres, tags).
        :param user_mapping: Dictionary with 'id_to_idx' and 'idx_to_id' for users.
        :param item_mapping: Dictionary with 'id_to_idx' and 'idx_to_id' for items.
        :param k: Number of nearest neighbors (items) to use in scoring.
        """
        self._rating_matrix = rating_matrix
        self._similarity_features = similarity_features.values
        self._k = k

        self._user_mapping = user_mapping
        self._item_mapping = item_mapping

        # Precompute the item-item similarity matrix using content features
        self._similarity_matrix = self._calculate_similarity()

    def _calculate_similarity(self):
        """
        Computes cosine similarity between all item pairs based on their feature vectors.

        :return: A 2D numpy array representing the item-item similarity matrix.
        """
        # Compute dot product between all item vectors
        dot_product = self._similarity_features @ self._similarity_features.T

        # Compute outer product of vector norms
        norms = np.linalg.norm(self._similarity_features, axis=1, keepdims=True)
        norm_product = norms @ norms.T

        # Avoid division by zero by using `where`
        similarity_matrix = np.divide(dot_product, norm_product, where=norm_product != 0)

        # Zero out the diagonal to prevent recommending the same item
        np.fill_diagonal(similarity_matrix, 0)

        return similarity_matrix

    def predict(self, user_id: str, top_n: int = 10) -> list[str]:
        """
        Recommends top-N items for a given user based on item-based collaborative filtering
        using content-based similarity.

        :param user_id: External user ID to generate recommendations for.
        :param top_n: Number of top items to return.
        :return: List of recommended item IDs.
        :raises ValueError: If the user ID is not found in the mapping.
        """
        if user_id not in self._user_mapping['id_to_idx']:
            raise ValueError(f"User {user_id} not found.")

        user_idx = self._user_mapping['id_to_idx'][user_id]

        # Extract the user's interaction vector (sparse row)
        user_vector = self._rating_matrix[user_idx]
        interacted_indices = user_vector.indices  # Indices of items the user has interacted with

        # Initialize an empty score vector for all items
        scores = np.zeros(self._rating_matrix.shape[1])

        # For each item the user has interacted with, find similar items
        for item_idx in interacted_indices:
            sim_vector = self._similarity_matrix[item_idx]

            # Get indices of top-k similar items (highest similarity values)
            top_k_idx = np.argsort(sim_vector)[-self._k:]
            scores[top_k_idx] += sim_vector[top_k_idx]  # Aggregate similarity scores

        # Identify top-N items with highest aggregated scores
        top_indices = np.argsort(scores)[-top_n:][::-1]

        # Convert internal indices to external item IDs
        top_ids = list(map(lambda x: self._item_mapping['idx_to_id'][x], top_indices))

        return top_ids
