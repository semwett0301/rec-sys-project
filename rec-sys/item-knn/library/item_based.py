
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from typing import Dict

class ItemBasedRecommender:
    def __init__(self, k: int):
        self.k = k
        self._train_matrix: csr_matrix = None
        self._user_id_to_idx: Dict[str, int] = None
        self._item_id_to_idx: Dict[str, int] = None
        self._idx_to_user_id: Dict[int, str] = None
        self._idx_to_item_id: Dict[int, str] = None

        self._model = None
        self._similarities = None
        self._neighbors = None

    def fit(self, train_matrix: csr_matrix, user_mapping: dict, item_mapping: dict):
        self._train_matrix = train_matrix
        self._user_id_to_idx = user_mapping['id_to_idx']
        self._item_id_to_idx = item_mapping['id_to_idx']
        self._idx_to_user_id = user_mapping['idx_to_id']
        self._idx_to_item_id = item_mapping['idx_to_id']

        self._model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=self.k + 1)
        self._model.fit(train_matrix.T)

        self._similarities, self._neighbors = self._model.kneighbors(train_matrix.T)
        self._similarities = self._similarities[:, 1:]
        self._neighbors = self._neighbors[:, 1:]

    def predict(self, user_id: str, item_id: str) -> float:
        if user_id not in self._user_id_to_idx or item_id not in self._item_id_to_idx:
            return 0.0

        user_idx = self._user_id_to_idx[user_id]
        item_idx = self._item_id_to_idx[item_id]

        user_ratings = self._train_matrix.getrow(user_idx).toarray().flatten()
        sim_items = self._neighbors[item_idx]
        sim_scores = self._similarities[item_idx]

        relevant_ratings = user_ratings[sim_items]
        mask = relevant_ratings != 0

        if not np.any(mask):
            return float(user_ratings[user_ratings > 0].mean() if user_ratings[user_ratings > 0].size else 0)

        weighted_sum = np.dot(relevant_ratings[mask], sim_scores[mask])
        normalization = np.sum(np.abs(sim_scores[mask]))

        if normalization == 0:
            return float(user_ratings[user_ratings > 0].mean() if user_ratings[user_ratings > 0].size else 0)

        return float(np.clip(weighted_sum / normalization, 1, 5))

