import logging
from typing import Dict, Tuple

import numpy as np
from scipy.sparse import csr_matrix


def _check_params(implicit_rating: Dict[str, Dict[str, float]], user_mapping: Dict[str, int],
                  item_mapping: Dict[str, int]):
    """
    Validates the input parameters for the SVD++ model with CSR matrix.

    :param implicit_rating: Dictionary of implicit feedback interactions
                             (keys - user id, values - ratings were given by users to item ids).
    :param user_mapping: Mapping from original user IDs to matrix indices.
    :param item_mapping: Mapping from original item IDs to matrix indices.
    """
    # Get user and item sets
    users_in_ratings = set(user_mapping.keys())
    items_in_ratings = set(item_mapping.keys())

    # Ensure implicit feedback does not contain users not present in the ratings matrix
    for user in implicit_rating.keys():
        if user not in users_in_ratings:
            raise ValueError(f"User {user} in implicit feedback is not present in the ratings matrix.")

    # Check if any user has more implicit interactions than the total number of items
    for user, interactions in implicit_rating.items():
        if len(interactions) > len(items_in_ratings):
            raise ValueError(f"User {user} has {len(interactions)} implicit interactions, "
                             f"which exceeds the number of available items ({len(items_in_ratings)}).")

        # Ensure all items in implicit feedback exist in the ratings matrix
        for item in interactions.keys():
            if item not in items_in_ratings:
                raise ValueError(
                    f"Item {item} in implicit feedback for user {user} is not present in the ratings matrix.")


class SVDpp:
    def __init__(self, n_factors=20, reg_all=0.02, lr_all=0.007, lr_bu=None, lr_bi=None,
                 lr_pu=None, lr_qi=None, lr_yj=None, reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_yj=None,
                 n_epochs=20, init_mean=0, init_std=0.1):
        """
        Initializes the SVD++ model with fine-grained control over learning rates and regularization.

        :param n_factors: Number of latent factors for users/items.
        :param reg_all: Global regularization term applied to all parameters.
        :param lr_all: Global learning rate applied to all parameters unless specified.
        :param lr_bu, lr_bi: Learning rates for user and item biases.
        :param lr_pu, lr_qi: Learning rates for user and item latent factors.
        :param lr_yj: Learning rate for implicit feedback factors.
        :param reg_bu, reg_bi, reg_pu, reg_qi, reg_yj: Regularization parameters for different components.
        :param n_epochs: Number of training iterations.
        :param init_mean, init_std: Mean and standard deviation for factor vector initialization.
        """
        self._n_factors = n_factors
        self._n_epochs = n_epochs

        # Learning rates (defaults to lr_all if not specified)
        self._lr_bu = lr_bu if lr_bu is not None else lr_all
        self._lr_bi = lr_bi if lr_bi is not None else lr_all
        self._lr_pu = lr_pu if lr_pu is not None else lr_all
        self._lr_qi = lr_qi if lr_qi is not None else lr_all
        self._lr_yj = lr_yj if lr_yj is not None else lr_all

        # Regularization terms (defaults to reg_all if not specified)
        self._reg_bu = reg_bu if reg_bu is not None else reg_all
        self._reg_bi = reg_bi if reg_bi is not None else reg_all
        self._reg_pu = reg_pu if reg_pu is not None else reg_all
        self._reg_qi = reg_qi if reg_qi is not None else reg_all
        self._reg_yj = reg_yj if reg_yj is not None else reg_all

        # Normal distribution parameters
        self._init_mean = init_mean
        self._init_std = init_std

    def predict(self, user_id: str, item_id: str) -> float:
        """
        Predicts the rating for a given user-item pair.

        :param user_id: User id.
        :param item_id: Item id.
        :return: Predicted rating.
        """

        # If both user and item are unknown, return the global mean rating
        if user_id not in self._user_id_to_idx and item_id not in self._item_id_to_idx:
            return self._global_mean

        # If the item is unknown, return the user's average rating
        if item_id not in self._item_id_to_idx:
            user_idx = self._user_id_to_idx[user_id]
            user_ratings = self._ratings[user_idx].data
            return np.mean(user_ratings) if len(user_ratings) > 0 else self._global_mean

        # If the user is unknown, return the item's average rating
        if user_id not in self._user_id_to_idx:
            item_idx = self._item_id_to_idx[item_id]
            item_ratings = self._ratings[:, item_idx].data
            return np.mean(item_ratings) if len(item_ratings) > 0 else self._global_mean

        # Otherwise, use the trained model to predict the rating
        user_idx = self._user_id_to_idx[user_id]
        item_idx = self._item_id_to_idx[item_id]
        predicted_rating, _, _ = self._predict_with_calculated_params(user_idx, item_idx)
        return predicted_rating

    def fit(self, ratings: csr_matrix, implicit_rating: Dict[str, Dict[str, float]],
            user_id_to_idx: Dict[str, int], item_id_to_idx: Dict[str, int]):
        """
        Trains the SVD++ model using a CSR matrix for ratings.

        :param ratings: csr_matrix where rows = users, columns = items, data = explicit ratings
        :param implicit_rating: Dictionary of implicit feedback interactions
                                 (keys - user id, values - ratings given by users to item ids).
        :param user_id_to_idx: Mapping from original user IDs to matrix indices.
        :param item_id_to_idx: Mapping from original item IDs to matrix indices.
        """
        # Store mappings
        self._user_id_to_idx = user_id_to_idx
        self._item_id_to_idx = item_id_to_idx

        self._reverse_user_mapping = {v: k for k, v in user_id_to_idx.items()}
        self._reverse_item_mapping = {v: k for k, v in item_id_to_idx.items()}

        # Validate parameters
        _check_params(implicit_rating, user_id_to_idx, item_id_to_idx)

        # Initialize parameters
        self._params_initialization(ratings, implicit_rating)

        # Iterate over non-zero entries in the sparse matrix
        for epoch in range(self._n_epochs):
            logging.info(f"Epoch: {epoch + 1}")

            # Get indices of non-zero elements
            rows, cols = ratings.nonzero()

            for user_idx, item_idx in zip(rows, cols):
                # logger.info(f"Epoch: {epoch + 1}, User: {user_idx}, Item: {item_idx}")

                # Perform gradient descent
                self._gradient_descent_iteration(user_idx, item_idx)

    def _random_normal_initialization(self, size: int) -> np.ndarray:
        """
        Initializes a matrix using a normal distribution.

        :param tech_list: Set of IDs used for index
        :return: A matrix initialized with values drawn from a normal distribution.
        """

        return np.random.normal(self._init_mean, self._init_std, (size, self._n_factors))

    def _params_initialization(self, ratings: csr_matrix, implicit_rating: Dict[str, Dict[str, float]]):
        """
        Initializes model parameters and matrices.

        :param ratings: csr_matrix where rows = users, columns = items, data = explicit ratings
        :param implicit_rating: Dictionary of implicit feedback interactions.
        """
        self._ratings = ratings
        num_users, num_items = ratings.shape

        # Initialize factor matrices
        self._user_factors = self._random_normal_initialization(num_users)
        self._item_factors = self._random_normal_initialization(num_items)
        self._implicit_factors = self._random_normal_initialization(num_items)

        # Initialize bias vectors
        self._bias_user = np.zeros(num_users)
        self._bias_item = np.zeros(num_items)

        # Calculate global mean of non-zero ratings
        self._global_mean = ratings.data.mean() if ratings.nnz > 0 else 0.0

        # Convert implicit feedback using internal indices (from user and item ids to indices)
        self._implicit_rating = {}
        for user_id, items in implicit_rating.items():
            user_idx = self._user_id_to_idx[user_id]
            self._implicit_rating[user_idx] = {
                self._item_id_to_idx[item_id]: weight
                for item_id, weight in items.items()
            }

    def _calculate_sqrt_n(self, user_idx: int) -> float:
        """
        Computes the square root of the total implicit feedback interactions for a given user.

        :param user_idx: Index of the user in the internal representation.
        :return: Square root of the sum of implicit feedback interactions or 1 if no interactions exist.
        """
        # Check for user presence in implicit feedback (otherwise 1)
        if user_idx not in self._implicit_rating:
            return 1.0

        sum_wuj = sum(self._implicit_rating[user_idx].values())
        return np.sqrt(sum_wuj) if sum_wuj > 0 else 1.0

    def _calculate_y_sum(self, user_idx: int) -> np.ndarray:
        """
        Computes the sum of implicit feedback factors weighted by interactions for a given user.

        :param user_idx: Index of the user in the internal representation.
        :return: Sum of weighted implicit feedback factors or a zero vector if no implicit feedback exists.
        """

        # Check for user presence in implicit feedback (otherwise 0)
        if user_idx not in self._implicit_rating or not self._implicit_rating[user_idx]:
            return np.zeros(self._n_factors)

        # Summarization of the current parameters * weight (>= 0)
        total_sum = np.zeros(self._n_factors)
        for item_idx, weight in self._implicit_rating[user_idx].items():
            total_sum += self._implicit_factors[item_idx] * weight

        return total_sum

    def _predict_with_calculated_params(self, user_idx: int, item_idx: int) -> Tuple[float, float, np.ndarray]:
        """
        Computes the predicted rating for a user-item pair using internal indices.

        :param user_idx: Index of the user in the internal representation.
        :param item_idx: Index of the item in the internal representation.
        :return: A tuple containing the predicted rating, sqrt_n, and y_sum.
        """
        sqrt_n = self._calculate_sqrt_n(user_idx)
        y_sum = self._calculate_y_sum(user_idx)

        user_factors = self._user_factors[user_idx]
        item_factors = self._item_factors[item_idx]

        # Compute predicted rating using bias terms and latent factors
        predicted_rating = (
                self._global_mean +
                self._bias_user[user_idx] +
                self._bias_item[item_idx] +
                np.dot(user_factors + y_sum / sqrt_n, item_factors)
        )

        return predicted_rating, sqrt_n, y_sum

    def _gradient_descent_iteration(self, user_idx: int, item_idx: int):
        """
        Performs a single iteration of gradient descent for a given user-item pair.

        :param user_idx: Index of the user in the internal representation.
        :param item_idx: Index of the item in the internal representation.
        """

        # An explicit rating
        rating = self._ratings[user_idx, item_idx]

        # Predict rating
        predicted_rating, sqrt_n, y_sum = self._predict_with_calculated_params(user_idx, item_idx)

        # Compute the prediction error
        error = rating - predicted_rating

        # Update user and item biases using gradient descent
        self._bias_user[user_idx] += self._lr_bu * (error - self._reg_bu * self._bias_user[user_idx])
        self._bias_item[item_idx] += self._lr_bi * (error - self._reg_bi * self._bias_item[item_idx])

        # Store copies of current factor vectors
        current_user_factors = self._user_factors[user_idx].copy()
        current_item_factors = self._item_factors[item_idx].copy()

        # Update user latent factors
        self._user_factors[user_idx] += self._lr_pu * (
                error * current_item_factors - self._reg_pu * current_user_factors
        )

        # Update item latent factors
        self._item_factors[item_idx] += self._lr_qi * (
                error * (current_user_factors + y_sum / sqrt_n) - self._reg_qi * current_item_factors
        )

        # Update implicit factors for all interacted items (if the user has implicit feedback)
        if user_idx in self._implicit_rating:
            for j_item_idx, implicit_rating in self._implicit_rating[user_idx].items():
                self._implicit_factors[j_item_idx] += self._lr_yj * (
                        error * current_item_factors * implicit_rating / sqrt_n -
                        self._reg_yj * self._implicit_factors[j_item_idx]
                )

        # Logging
        # logger.info(
        #     f"Rating: {rating}, Predicted: {predicted_rating}, Error: {error}, "
        #     f"User bias: {self._bias_user[user_idx]}, "
        #     f"Item bias: {self._bias_item[item_idx]}, "
        #     f"User factors: {self._user_factors[user_idx]}, "
        #     f"Item factors: {self._item_factors[item_idx]}"
        # )
