import numpy as np
from scipy.sparse import csr_matrix
from svdpp import SVDpp
import heapq

class SvdMetricsCalculator:
    def __init__(self, test_matrix: csr_matrix, model: SVDpp, idx_to_user_id: dict[int, str],
                 idx_to_item_id: dict[int, str]):
        self._test_matrix = test_matrix

        result_lil = test_matrix.tolil()

        coo = test_matrix.tocoo()
        for row, col in zip(coo.row, coo.col):
            result_lil[row, col] = model.predict(idx_to_user_id[row], idx_to_item_id[col])

        self._result_matrix = result_lil.tocsr()

    def calculate_rmse(self) -> float:
        return np.sqrt(np.mean((self._test_matrix.data - self._result_matrix.data) ** 2))


class SvdTestMetricsCalculator:
    def __init__(self, test_matrix: csr_matrix, model: SVDpp,
                 idx_to_user_id: dict[int, str],
                 idx_to_item_id: dict[int, str],
                 n: int = 10):
        self._model = model
        self._test_matrix = test_matrix
        self._idx_to_user_id = idx_to_user_id
        self._idx_to_item_id = idx_to_item_id

        self._item_popularity = self._compute_item_popularity()
        self._avg_popularity = sum(self._item_popularity.values()) / len(self._item_popularity)

        self._top_n_list = self._generate_top_n(n)

    def _compute_item_popularity(self) -> dict[int, int]:
        """
        Computes how many users interacted with each item.
        Uses sparse matrix access (fast and memory-efficient).
        Returns: dict {item_idx: popularity}
        """
        item_popularity = {}
        n_users = self._test_matrix.shape[0]

        for user_idx in range(n_users):
            row = self._test_matrix.getrow(user_idx)
            for item_idx in row.indices:
                if item_idx in item_popularity:
                    item_popularity[item_idx] += 1
                else:
                    item_popularity[item_idx] = 1

        return item_popularity

    def _set_top_n(self, n: int):
        self._top_n_list = self._generate_top_n(n)

    def _generate_top_n(self, top_n: int) -> dict[int, list]:
        """
        Generates top-N recommendations per user using all unseen items.
        Returns: {user_idx: [(item_idx, estimated_rating), ...]}
        """
        top_n_items = {}
        n_users, n_items = self._test_matrix.shape

        for user_idx in range(n_users):
            top_n_items[user_idx] = [(0, -1),] * top_n

            user_id = self._idx_to_user_id[user_idx]

            # Items the user has already interacted with
            seen_items = set(self._test_matrix.getrow(user_idx).indices)

            # All items the user hasn't seen
            unseen_items = set(range(n_items)) - seen_items

            for item_idx in unseen_items:
                item_id = self._idx_to_item_id[item_idx]
                r_est = self._model.predict(user_id, item_id)

                if r_est > top_n_items[user_idx][0][0]:
                    heapq.heappushpop(top_n_items[user_idx], (r_est, item_idx))

        return top_n_items

    # def calculate_rmse(self) -> float:
    #     return np.sqrt(np.mean((self._test_matrix.data - self._result_matrix.data) ** 2))

    def calculate_unexpectedness(self) -> float:
        """
        Calculates the average unexpectedness across all users.
        Returns: single float value.
        """
        total_unexpectedness = 0.0
        user_count = 0

        for user_idx, recs in self._top_n_list.items():
            unexpected_count = 0
            for item_idx, _ in recs:
                popularity = self._item_popularity.get(item_idx, 0)
                if popularity < self._avg_popularity:
                    unexpected_count += 1

            if len(recs) > 0:
                user_score = unexpected_count / len(recs)
                total_unexpectedness += user_score
                user_count += 1

        if user_count == 0:
            return total_unexpectedness

        return total_unexpectedness / user_count






