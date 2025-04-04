import logging
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from pyarrow import union
from scipy.sparse import csr_matrix
from sympy.physics.quantum.density import entropy

from svdpp import SVDpp
import heapq


class RmseCalculator:
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


class TestMetricsCalculator:
    def __init__(self, test_matrix: csr_matrix, model: SVDpp,
                 idx_to_user_id: dict[int, str],
                 idx_to_item_id: dict[int, str],
                 n: int = 10):
        self._model = model
        self._test_matrix = test_matrix
        self._idx_to_user_id = idx_to_user_id
        self._idx_to_item_id = idx_to_item_id

        self._top_n = n
        self._top_n_list = self._generate_top_n(n)

        item_popularity = self._calculate_item_popularity()
        self._item_popularity = item_popularity
        self._mean_popularity = sum(item_popularity.values()) / len(item_popularity.values())

    def _calculate_item_popularity(self):
        recommended_item_list = list(set([item for user_recs in self._top_n_list.values() for item, _ in user_recs]))
        item_popularity = {}

        for item_idx in recommended_item_list:
            current_popularity = self._test_matrix.getcol(item_idx).nnz / self._test_matrix.shape[0]
            item_popularity[item_idx] = current_popularity

            logging.info(f"Calculate item popularity for item {item_idx} - {current_popularity}")

        return item_popularity

    def _is_item_relevant(self, user_idx: int, item_idx: int) -> bool:
        relevant_items = self._test_matrix[user_idx]

        return relevant_items[0, item_idx] != 0

    def _set_top_n(self, n: int):
        self._top_n_list = self._generate_top_n(n)

    def _generate_top_n(self, top_n: int) -> dict[int, list[tuple[int, int]]]:
        """
        Generates top-N recommendations per user using all unseen items.
        Returns: {user_idx: [(item_idx, estimated_rating), ...]}
        """
        logging.info(f"Create top-{top_n} recommendations' list")

        top_n_items = {}
        n_users, n_items = self._test_matrix.shape

        for user_idx in range(n_users):
            top_n_items[user_idx] = [(0, -1), ] * top_n

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

            top_n_items[user_idx] = [(item_idx, r_est) for r_est, item_idx in
                                     sorted(top_n_items[user_idx], reverse=True)]
            logging.info(f"User: {user_idx} -- top {top_n} list -- {top_n_items[user_idx]}")

        return top_n_items

    def get_mean_popularity(self):
        return self._mean_popularity

    def calculate_avg_entropy_novelty(self):
        final_entropy_novelty = 0

        for user_idx in self._top_n_list.keys():
            current_top_n_list = self._top_n_list[user_idx]
            current_items_popularity = [self._item_popularity[item_idx] for item_idx, _ in current_top_n_list]

            final_entropy_novelty += -sum(popularity * math.log(popularity) for popularity in current_items_popularity)

        return final_entropy_novelty / len(self._top_n_list.keys())

    def calculate_item_space_coverage(self) -> float:
        # Flatten all recommended item indices
        all_items = [item for user_recs in self._top_n_list.values() for item, _ in user_recs]

        # Count how often each item was recommended
        item_counts = Counter(all_items)
        total_recs = len(all_items)

        # Calculate entropy
        return -sum((count / total_recs) * math.log(count / total_recs) for count in item_counts.values())

    def calculate_recovery(self):
        total_users = 0
        total_recovery = 0.0

        for user_idx, recs in self._top_n_list.items():
            if self._test_matrix.indptr[user_idx] == self._test_matrix.indptr[user_idx + 1]:
                continue

            user_recovery = 0.0
            found_relevant = 0

            for rank, (item_idx, _) in enumerate(recs):
                if self._is_item_relevant(user_idx, item_idx):
                    user_recovery += (rank + 1) / self._top_n  # normalized rank
                    found_relevant += 1

            if found_relevant > 0:
                user_recovery /= found_relevant  # average over relevant items
                total_recovery += user_recovery
                total_users += 1

        return total_recovery / total_users if total_users > 0 else 0

    def calculate_agg_div(self):
        agg_div_set = set()

        for _, items_list in self._top_n_list.items():
            for item_rating in items_list:
                agg_div_set.add(item_rating[0])

        return len(agg_div_set) / self._test_matrix.shape[1]

    def calculate_serendipity(self, with_relevance=True) -> float:
        """
        Calculates the average unexpectedness across all users.
        Returns: single float value.
        """
        total_serendipity = 0.0
        user_count = 0

        for user_idx, recs in self._top_n_list.items():
            serendipity_count = 0

            for item_idx, _ in recs:
                popularity = self._item_popularity[item_idx]
                if popularity < self._mean_popularity and (
                        self._is_item_relevant(user_idx, item_idx) or not with_relevance):
                    serendipity_count += 1

            if len(recs) > 0:
                user_score = serendipity_count / len(recs)
                total_serendipity += user_score
                user_count += 1

        if user_count == 0:
            return total_serendipity

        return total_serendipity / user_count
