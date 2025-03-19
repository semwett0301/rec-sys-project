from typing import Dict

import numpy as np


def _check_params(ratings: np.ndarray, implicit_feedback: Dict[int, Dict[int, float]]):
    """
    Validates the input parameters for the SVD++ model.

    :param ratings: User-item ratings matrix.
    :param implicit_feedback: Dictionary of implicit feedback interactions.
    """
    n_user, n_items = ratings.shape

    # Ensure implicit feedback does not contain more users than exist in ratings matrix
    assert n_user >= len(implicit_feedback.keys()), "Implicit feedback dict contains more users that exist in 'rating' matrix"

    # Check if any user has more implicit interactions than the total number of items
    for u in implicit_feedback:
        length = len(implicit_feedback[u])
        if length > n_items:
            raise ValueError(f"User {u} has {length} implicit interactions, which exceeds the number of items ({n_items}).")

# TODO change ratings from 2-dimension np.array to pd.DataFrame
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
        self._item_factors = np.array([]) # Matrix of item latent factors
        self._user_factors = np.array([]) # Matrix of user latent factors

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

    def predict(self, u_idx: int, i_idx: int):
        """
        Predicts the rating for a given user-item pair.

        :param u_idx: User index.
        :param i_idx: Item index.
        :return: Predicted rating.
        """

        # If both user and item are unknown, return the global mean rating
        if u_idx >= self._n_users and i_idx >= self._n_items:
            return self._global_mean

        # If the item is unknown, return the user's average rating
        if i_idx >= self._n_items:
            return np.mean(self._ratings[u_idx, self._ratings[u_idx] > 0])

        # If the user is unknown, return the item's average rating
        if u_idx >= self._n_users:
            return np.mean(self._ratings[self._ratings[:, i_idx] > 0, i_idx])

        # Otherwise, use the trained model to predict the rating
        predicted_rating, _, _ = self._predict_with_calculated_params(u_idx, i_idx)
        return predicted_rating

    def fit(self, ratings: np.ndarray, implicit_feedback: Dict[int, Dict[int, float]]):
        """
        Trains the SVD++ model using a ratings matrix.

        :param ratings: 2D NumPy array where rows = users, cols = items, and values = ratings (0 if missing).
        :param implicit_feedback: Dictionary of implicit feedback interactions (keys - users, values - ratings were given by users to items).
        """
        _check_params(ratings, implicit_feedback)

        # Parameters initialization
        self._params_initialization(ratings, implicit_feedback)

        # Get non-zero rating indices
        user_indices, item_indices = np.where(ratings > 0)
        indices_pairs = zip(user_indices, item_indices)

        for epoch in range(self._n_epochs):
            for u_idx, i_idx in indices_pairs:
                self._gradient_descent_iteration(u_idx, i_idx)

    def _random_normal_initialization(self, x_size, y_size):
        """
        Initializes a matrix using a normal distribution.

        :param x_size: Number of rows in the matrix.
        :param y_size: Number of columns in the matrix.
        :return: A matrix initialized with values drawn from a normal distribution.
        """
        return np.random.normal(self._init_mean, self._init_std,(x_size, y_size))

    def _params_initialization(self, ratings: np.ndarray, implicit_feedback: Dict[int, Dict[int, float]]):
        """
        Initializes model parameters and matrices.

        :param ratings: 2D NumPy array where rows = users, cols = items, and values = ratings (0 if missing).
        :param implicit_feedback: Dictionary of implicit feedback interactions (keys - users, values - ratings were given by users to items).
        """
        self._ratings = ratings
        self._n_users, self._n_items = ratings.shape

        self._user_factors = self._random_normal_initialization(self._n_users, self._n_factors)
        self._item_factors = self._random_normal_initialization(self._n_items, self._n_factors)
        self._implicit_factors = self._random_normal_initialization(self._n_items, self._n_factors)

        self._bias_user: np.ndarray = np.zeros(self._n_users)
        self._bias_item: np.ndarray = np.zeros(self._n_items)

        self._global_mean = np.mean(ratings[ratings > 0]) # Mean rating across known ratings

        self._implicit_feedback = implicit_feedback

    def _calculate_sqrt_n(self, u_idx: int):
        """
        Computes the square root of the total implicit feedback interactions for a given user.

        :param u_idx: Index of the user.
        :return: Square root of the sum of implicit feedback interactions or 1 if no interactions exist.
        """
        sum_wuj = sum(self._implicit_feedback[u_idx].values()) if u_idx in self._implicit_feedback else 0
        return np.sqrt(sum_wuj) if sum_wuj > 0 else 1

    def _calculate_y_sum(self, u_idx: int):
        """
        Computes the sum of implicit feedback factors weighted by interactions for a given user.

        :param u_idx: Index of the user.
        :return: Sum of weighted implicit feedback factors or a zero vector if no implicit feedback exists.
        """
        return (
            np.sum([self._implicit_factors[j_idx] * self._implicit_feedback[u_idx][j_idx] for j_idx in
                    self._implicit_feedback.get(u_idx, {})], axis=0)
            if len(self._implicit_feedback.get(u_idx, {}).values()) > 0 else np.zeros(self._n_factors)
        )

    def _predict_with_calculated_params(self, u_idx: int, i_idx: int) -> tuple[float, float, float]:
        """
        Computes the predicted rating for a user-item pair.

        :param u_idx: Index of the user.
        :param i_idx: Index of the item.
        :return: A tuple containing the predicted rating, sqrt_n, and y_sum.
        """
        assert u_idx < self._n_users, "The user is unknown"
        assert i_idx < self._n_items, "The item is unknown"

        sqrt_n = self._calculate_sqrt_n(u_idx)  # Compute square root of implicit feedback sum
        y_sum = self._calculate_y_sum(u_idx)  # Compute sum of implicit factors weighted by feedback

        # Compute predicted rating using bias terms and latent factors
        predicted_rating = self._global_mean + self._bias_user[u_idx] + self._bias_item[i_idx] + np.dot(
            (self._user_factors[u_idx] + y_sum / sqrt_n), self._item_factors[i_idx])

        return predicted_rating, sqrt_n, y_sum

    def _gradient_descent_iteration(self, u_idx: int, i_idx: int):
        """
        Performs a single iteration of gradient descent for a given user-item pair.

        :param u_idx: Index of the user.
        :param i_idx: Index of the item.
        """
        current_rating = self._ratings[u_idx][i_idx] # Retrieve the actual explicit rating
        predicted_rating, sqrt_n, y_sum = self._predict_with_calculated_params(u_idx, i_idx) # Predict rating

        error = current_rating - predicted_rating # Compute the prediction error

        # Update user and item biases using gradient descent
        # Formula: bu = bu + lr_bu * (error - reg_bu * bu)
        # Formula: bi = bi + lr_bi * (error - reg_bi * bi)
        self._bias_user[u_idx] += self._lr_bu * (error - self._reg_bu * self._bias_user[u_idx])
        self._bias_item[i_idx] += self._lr_bi * (error - self._reg_bi * self._bias_item[i_idx])

        # Store copies of current factor vectors for calculations to not use updated versions of the coefficient
        current_user_factors = self._user_factors[u_idx].copy()
        current_item_factors = self._item_factors[i_idx].copy()

        # Update user latent factors:
        # Formula: pu = pu + lr_pu * (error * qi - reg_pu * pu)
        self._user_factors[u_idx] += self._lr_pu * (error * self._item_factors[i_idx] - self._reg_pu * self._user_factors[u_idx])
        self._item_factors[i_idx] += self._lr_qi * (
                    error * (current_user_factors + y_sum / sqrt_n) - self._reg_qi * self._item_factors[i_idx])

        # Update implicit factors for all interacted items
        for j in self._implicit_feedback.get(u_idx, {}):
            current_implicit_rating = self._implicit_feedback[u_idx][j]

            # Formula: yj = yj + lr_yj * (error * qi * implicit_rating / sqrt_n - reg_yj * yj)
            self._implicit_factors[j] += self._lr_yj * (
                    error * current_item_factors * current_implicit_rating / sqrt_n - self._reg_yj *
                    self._implicit_factors[j])





