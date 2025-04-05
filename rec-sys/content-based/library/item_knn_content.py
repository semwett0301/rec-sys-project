import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

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
        self._rating_matrix = rating_matrix
        self._similarity_features = similarity_features.values
        self._k = k

        self._user_mapping = user_mapping
        self._item_mapping = item_mapping

        self._similarity_matrix = self._calculate_similarity()

    def _calculate_similarity(self):
        dot_product = self._similarity_features @ self._similarity_features.T

        norms = np.linalg.norm(self._similarity_features, axis=1, keepdims=True)
        norm_product = norms @ norms.T

        similarity_matrix = np.divide(dot_product, norm_product, where=norm_product != 0)
        np.fill_diagonal(similarity_matrix, 0)

        return similarity_matrix

    def predict(self, user_id: str, top_n: int = 10) -> list[str]:
        if user_id not in self._user_mapping['id_to_idx']:
            raise ValueError(f"User {user_id} not found.")

        user_idx = self._user_mapping['id_to_idx'][user_id]
        user_vector = self._rating_matrix[user_idx]
        interacted_indices = user_vector.indices

        scores = np.zeros(self._rating_matrix.shape[1])

        for item_idx in interacted_indices:
            # Get top-K similar items to item_idx
            sim_vector = self._similarity_matrix[item_idx]
            top_k_idx = np.argsort(sim_vector)[-self._k:]
            scores[top_k_idx] += sim_vector[top_k_idx]

        # Get top-N scores
        top_indices = np.argsort(scores)[-top_n:][::-1]
        top_ids = list(map(lambda x: self._item_mapping['idx_to_id'][x], top_indices))

        return top_ids
