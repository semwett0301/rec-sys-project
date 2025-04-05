import heapq
import logging
import math
from collections import Counter
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# Define a protocol to ensure any model passed has a .predict(user_id, item_id) method
class Predictable(Protocol):
    def predict(self, user_id: str, item_id: str) -> float:
        ...


# RMSE calculator that evaluates a Predictable model against a test matrix
class RmseCalculator:
    def __init__(self, matrix: csr_matrix, model: Predictable, idx_to_user_id: dict[int, str],
                 idx_to_item_id: dict[int, str]):
        # Store the ground truth test matrix (user-item interactions)
        self._test_matrix = matrix

        # Convert the matrix to LIL format to allow efficient row-wise assignments
        result_lil = matrix.tolil()

        # Convert to COO format to efficiently iterate over non-zero elements
        coo = matrix.tocoo()
        for row, col in zip(coo.row, coo.col):
            # Predict the rating using the model for each non-zero (user, item) pair
            result_lil[row, col] = model.predict(idx_to_user_id[row], idx_to_item_id[col])

        # Convert back to CSR format for efficient numerical operations
        self._result_matrix = result_lil.tocsr()

    def calculate_rmse(self) -> float:
        """
        Calculate Root Mean Square Error (RMSE) between predicted and actual ratings.
        Only considers the non-zero entries from the test matrix (i.e., observed ratings).
        """
        return np.sqrt(np.mean((self._test_matrix.data - self._result_matrix.data) ** 2))


# Class for "non-accuracy" metrics calculation
class TestMetricsCalculator:
    def __init__(self, test_matrix: csr_matrix, model: Predictable,
                 idx_to_user_id: dict[int, str],
                 idx_to_item_id: dict[int, str],
                 relevance_threshold = 1,
                 n: int = 10):
        """
        Initializes the calculator for various evaluation metrics.
        - :param test_matrix: user-item interaction test matrix
        - :param model: the recommendation model that implements .predict(user_id, item_id)
        - :param idx_to_user_id / idx_to_item_id: mapping from matrix indices to original IDs
        - :param relevance_threshold: the threshold when we decide that user assessed item relevant for him
        - :param n: top-N recommendation list size
        """
        self._model = model
        self._test_matrix = test_matrix
        self._idx_to_user_id = idx_to_user_id
        self._idx_to_item_id = idx_to_item_id

        self._top_n = n
        self._relevance_threshold = relevance_threshold
        self._top_n_list = self._generate_top_n(n)  # dict of top-N recommendations per user

        self._item_popularity = self._calculate_item_popularity()  # dict <item_idx> - <item_popularity>

    def _calculate_item_popularity(self):
        """
        Computes item popularity as the proportion of users who interacted with each item,
        but only for items that appeared in any top-N recommendation list.

        Popularity is defined as:
            popularity(i) = (# of users who interacted with item i) / total number of users

        This is used in various novelty, serendipity, and diversity metrics to estimate how "expected" or "common" a recommended item is.

        :return: A dictionary {item_idx: popularity_score} for all items in top-N lists.
    """
        # Flatten all top-N lists and get a unique set of all recommended items
        recommended_item_list = list(set([item for user_recs in self._top_n_list.values() for item, _ in user_recs]))
        item_popularity = {}

        # For each recommended item, calculate its popularity from the test set
        for item_idx in recommended_item_list:
            # Get the number of users who interacted with this item (non-zero entries in the item's column)
            current_popularity = self._test_matrix.getcol(item_idx).nnz / self._test_matrix.shape[0]
            item_popularity[
                item_idx] = current_popularity if current_popularity > 0 else 1e-10  # Because otherwise it's possible to get an error

            logging.info(f"Calculate item popularity for item {item_idx} - {current_popularity}")

        return item_popularity

    def _is_item_relevant(self, user_idx: int, item_idx: int) -> bool:
        """
        Checks whether the given item is relevant to the user (i.e., non-zero entry in test matrix).
        """
        relevant_items = self._test_matrix[user_idx]
        return relevant_items[0, item_idx] >= self._relevance_threshold

    def _set_top_n(self, n: int):
        """
        Re-generates top-N recommendations if N is changed.
        """
        self._top_n_list = self._generate_top_n(n)

    def _generate_top_n(self, top_n: int) -> dict[int, list[tuple[int, int]]]:
        """
        For each user in the test matrix, generates top-N recommendations by scoring
        all items the user has not previously interacted with.

        Returns:
            A dictionary:
                {
                    user_idx: [(item_idx, estimated_rating), ...]
                }
            where each list contains the top-N recommended items for that user, sorted
            in descending order of predicted rating.
        """
        logging.info(f"Create top-{top_n} recommendations' list")

        top_n_items = {}  # Will store top-N lists for each user
        n_users, n_items = self._test_matrix.shape

        # Iterate over every user in the dataset
        for user_idx in range(n_users):
            # Initialize with dummy entries so we can use heapq.heappushpop efficiently
            # Format: [(estimated_rating, item_idx), ...]
            # Note: estimated_rating goes first for heapq to sort by it
            top_n_items[user_idx] = [(0, -1)] * top_n

            # Retrieve the original user ID from the index
            user_id = self._idx_to_user_id[user_idx]

            # Items the user has already interacted with (we should not recommend them again)
            seen_items = set(self._test_matrix.getrow(user_idx).indices)

            # Determine which items are unseen (i.e., candidate items for recommendation)
            unseen_items = set(range(n_items)) - seen_items

            # Score each unseen item using the model
            for item_idx in unseen_items:
                item_id = self._idx_to_item_id[item_idx]
                r_est = self._model.predict(user_id, item_id)

                # Use a min-heap to maintain only the top-N items with highest estimated ratings
                if r_est > top_n_items[user_idx][0][0]:
                    heapq.heappushpop(top_n_items[user_idx], (r_est, item_idx))

            # After gathering top-N, sort by estimated rating in descending order
            # This gives us: [(item_idx, rating), ...] from highest to lowest
            top_n_items[user_idx] = [
                (item_idx, r_est)
                for r_est, item_idx in sorted(top_n_items[user_idx], reverse=True)
            ]

            logging.info(f"User: {user_idx} -- top {top_n} list -- {top_n_items[user_idx]}")

        return top_n_items

    def generate_metrics_summary_df(self, rmse: float = None) -> pd.DataFrame:
        """
        Generates a summary DataFrame with key evaluation metrics:
        RMSE, Recovery, Diversity, Novelty, Serendipity, etc.

        :return: pandas DataFrame with columns: Metric, Area, Value, Value Range, Meaning
        """
        # Check for recovery
        recovery = self.calculate_recovery()
        recovery_display = recovery if recovery is not None else "None"

        # Metrics calculation
        normalized_aggdiv = self.calculate_agg_div()
        normalized_aggdiv_coverage = self.calculate_agg_div(is_coverage=True)

        serendipity = self.calculate_serendipity(with_relevance=True)
        unexpectedness = self.calculate_serendipity(with_relevance=False)

        item_space_coverage = self.calculate_item_space_coverage()
        normalized_item_deg = self.calculate_normalized_item_deg()

        data = [
            ["Recovery", "Relevance", recovery_display, f"[0, {round(1 - 1 / self._top_n, 3)}]",
             "How early relevant items appear in top-N recommendations"],
            ["Normalized AggDiv (diversity)", "Inter-user diversity", normalized_aggdiv, "[0, 1]",
             "Proportion of unique items recommended across all users divided by the amount of recommendations"],
            ["Normalized AggDiv (coverage)", "Coverage", normalized_aggdiv_coverage, "[0, 1]",
             "Proportion of unique items recommended across all users divided by the size of catalog"],
            ["Item Space Coverage", "Coverage", round(item_space_coverage, 3), "[0, Not defined]",
             "Shows how many unique items and how often appears in the RLs (ideally a lot of different items recommended uniformly)"],
            ["Normalized ItemDeg", "Novelty", round(normalized_item_deg, 3), "[0, 1]",
             "Novelty of recommended items based on inverse (log) item popularity"],
            ["Unexpectedness (no relevance)", "Serendipity", round(unexpectedness, 3), "[0, 1]",
             "Proportion of items that are unexpected (less popular than average)"],
            ["Serendipity (with relevance)", "Serendipity", round(serendipity, 3), "[0, 1]",
             "Proportion of unexpected and relevant items in top-N recommendations"]
        ]

        if rmse:
            data.append(["RMSE", "Relevance", round(rmse, 3), "[0, 6]",
                         "Root Mean Square Error between predicted and actual ratings"])

        return pd.DataFrame(data, columns=["Metric", "Area", "Value", "Value Range", "Meaning"])

    def get_range_of_metrics(self):
        """
        Returns the theoretical min/max range of all metrics along with their explanations.
        """
        max_recovery = 1 - 1 / self._top_n

        return pd.DataFrame({
            "Metric": ["Item space coverage", "Recovery", "Normalized AggDiv (diversity)", "Normalized AggDiv (coverage)", "Unexpectedness (with_relevance=False)",
                       "Serendipity (with_relevance=True)", "Normalized ItemDeg"],
            "Min": [0, 0, 0, 0, 0, 0, 0],
            "Max": ["Not defined", max_recovery, 1, 1, 1, 1, 1],
            "Explanation": [
                "small - recommendations focuses on several item only or aren't balanced, big - recommendations are distributed uniformly across a lot of items",
                f"0 - all the relevant items on the top of the list, {max_recovery} - all relevant items in the bottom of the list, None - no relevant items in the RLs",
                "0 - only 1 item was recommended for everyone, 1 - all recommendations are different",
                "0 - only 1 item was recommended, 1 - all the items from catalog were recommended",
                "0 - there is no unexpected item (popularity below the average) in all RLs, 1 - all the items are unexpected",
                "0 - there is no serendipitous item (popularity below the average + relevant) in all RLs, 1 - all the items are serendipitous",
                "0 - the most popular items are used (no novelty), 1 - all items are the most unpopular (the best novelty)",
            ]
        })

    def get_test_set_statistic(self):
        """
        Returns basic statistics about the test matrix and item popularity.
        """
        relevant_pairs = (self._test_matrix.data > self._relevance_threshold).sum()
        potential_pairs = self._test_matrix.nnz
        all_pairs = self._test_matrix.shape[0] * self._test_matrix.shape[1]

        test_set_statistic = pd.Series({
            "Mean popularity": sum(self._item_popularity.values()) / len(self._item_popularity.values()),
            "Max popularity": max(self._item_popularity.values()),
            "Min popularity": min(self._item_popularity.values()),
            "Number of pairs": all_pairs,
            "Non-null pairs (u-i)": potential_pairs,
            "% of non-null pairs": potential_pairs / all_pairs * 100,
            "Relevant pairs (u-i)": relevant_pairs,
            "% of relevant pairs": relevant_pairs / all_pairs * 100,
        })

        test_set_statistic = test_set_statistic.apply(lambda x: f"{x:.6f}")

        return test_set_statistic

    def calculate_normalized_item_deg(self) -> float:
        """
        Calculates the normalized ItemDeg metric using log-popularity.

        This metric measures the average novelty of the recommended items across all users.
        Popularity is estimated as the proportion of users who interacted with the item.
        The novelty is derived from the log of item popularity — lower popularity means higher novelty.

        Normalization:
            The result is scaled to [0, 1], where:
                - 0 means all recommended items are very popular (low novelty)
                - 1 means all recommended items are very rare (high novelty)

        - :return A float between 0 and 1 representing the average novelty of recommendations.
        """
        total_users = 0  # Count of users for whom we have recommendations
        total_novelty = 0  # Accumulator for log-popularity scores (which inversely measure novelty)

        # Compute the max and min of log-popularity across all recommended items
        # Used for normalization later
        log_max_pop = math.log(max(self._item_popularity.values()))
        log_min_pop = math.log(min(self._item_popularity.values()))

        # To avoid division by zero if all items have the same popularity
        log_range = log_max_pop - log_min_pop if log_max_pop != log_min_pop else 1e-10

        # Iterate through top-N recommendations per user
        for user_idx, recs in self._top_n_list.items():
            total_users += 1
            for item_idx, _ in recs:
                # Add log of popularity (log values are negative: rarer = more negative)
                total_novelty += math.log(self._item_popularity[item_idx])

        # Calculate average log-popularity across all recommendations
        total_possible = total_users * self._top_n
        avg_log_pop = total_novelty / total_possible if total_possible > 0 else 0.0

        # Normalize to [0, 1]
        # Higher = rarer items were recommended (more novel)
        normalized_item_deg = (log_max_pop - avg_log_pop) / log_range

        return normalized_item_deg

    def calculate_item_space_coverage(self) -> float:
        """
        Calculates item space coverage using Shannon entropy over how often
        each item appears in users' top-N recommendation lists.

        :return: a float representing unnormalized entropy (higher = more diverse coverage).
        """
        # Count how many users received each item in their top-N
        item_occurrence = Counter()

        for top_list in self._top_n_list.values():
            unique_items = {item_idx for item_idx, _ in top_list}  # avoid duplicates per user
            item_occurrence.update(unique_items)

        total_users = len(self._top_n_list)

        # Convert counts into probabilities
        p_dict = {
            item_idx: count / total_users
            for item_idx, count in item_occurrence.items()
        }

        # Compute entropy (unnormalized)
        return -sum(p * math.log(p) for p in p_dict.values())



    def calculate_recovery(self):
        """
        Calculate the average recovery score across all users.

        Recovery measures how early relevant items appear in the top-N recommendations.
        For each user, it computes the average normalized rank position of relevant items
        (those that appear in the test set) among the top-N recommended items.

        :return: float or None: The average recovery score across users, or None if no relevant items are found.
        """
        # Initialize counters for total users considered and their cumulative recovery score
        total_users = 0
        total_recovery = 0.0

        # Iterate through each user's recommendation list
        for user_idx, recs in self._top_n_list.items():
            # Skip users who have no relevant items in the test matrix
            if self._test_matrix.indptr[user_idx] == self._test_matrix.indptr[user_idx + 1]:
                continue

            user_recovery = 0.0  # Tracks the recovery score for this user
            found_relevant = 0  # Count of relevant items found in the top-N list

            # Go through the ranked recommended items
            for rank, (item_idx, _) in enumerate(recs):
                # Check if the recommended item is relevant for the user (in the test set)
                if self._is_item_relevant(user_idx, item_idx):
                    # Accumulate normalized rank (lower ranks = better recovery)
                    user_recovery += (rank + 1) / self._top_n
                    found_relevant += 1

            # If relevant items were found, average the recovery and update totals
            if found_relevant > 0:
                user_recovery /= found_relevant  # Normalize by number of relevant items
                total_recovery += user_recovery
                total_users += 1

        # If no users had relevant items in top-N, print a warning and return None
        if total_users == 0:
            print(
                f"There is no one relevant item in the top-{self._top_n} recommendation list => Recovery can't be calculated")
            return None

        # Return the average recovery score across all users
        return total_recovery / total_users

    def calculate_agg_div(self, is_coverage=False):
        """
        Calculate the Normalized Aggregate Diversity of the recommendations.

        This metric measures the proportion of unique items recommended across all users,
        relative to the total number of items in the system. It reflects how diverse
        the recommendation set is at the system level.

        :return: float: A value between 0 and 1 representing normalized aggregate diversity.
        """
        agg_div_set = set()  # Set to store all unique recommended item IDs

        # Iterate through each user's top-N recommendation list
        for _, items_list in self._top_n_list.items():
            for item_rating in items_list:
                agg_div_set.add(item_rating[0])  # Add the item ID (not the rating)

        max_n = self._top_n * len(self._top_n_list.keys()) if not is_coverage else self._test_matrix.shape[1]
        return len(agg_div_set) / max_n

    def calculate_serendipity(self, with_relevance=True) -> float:
        """
        Calculate the average serendipity of the recommendation lists.

        Serendipity is the proportion of recommended items that are both:
          - Unexpected (i.e., less popular than average)
          - Relevant (if `with_relevance=True`)

        It rewards recommendations that surprise users with less obvious,
        yet potentially useful, items.

        :param with_relevance: bool If True, count only unexpected and relevant items.
                                    If False, consider only unexpectedness.

        :return: float: The average serendipity score across all users.
        """
        total_serendipity = 0.0  # Sum of user serendipity scores
        user_count = 0  # Number of users considered

        # Compute average item popularity as a threshold for "unexpectedness"
        mean_popularity = sum(self._item_popularity.values()) / len(self._item_popularity.values())

        # Loop through each user's top-N recommended items
        for user_idx, recs in self._top_n_list.items():
            serendipity_count = 0  # Number of serendipitous items for this user

            for item_idx, _ in recs:
                popularity = self._item_popularity[item_idx]

                # An item is considered serendipitous if it's less popular than average
                # and relevant (if with_relevance=True)
                if popularity < mean_popularity and (
                        self._is_item_relevant(user_idx, item_idx) or not with_relevance
                ):
                    serendipity_count += 1

            # Avoid division by zero if user has no recommendations
            if len(recs) > 0:
                user_score = serendipity_count / len(recs)
                total_serendipity += user_score
                user_count += 1

        # Return average serendipity across all users (or 0.0 if no users were counted)
        if user_count == 0:
            return 0.0

        return total_serendipity / user_count
