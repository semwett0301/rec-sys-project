from typing import Dict

import numpy as np
import pandas as pd


def _check_params(ratings: pd.DataFrame, implicit_feedback: Dict[int, Dict[int, float]]):
    """
    Validates the input parameters for the SVD++ model.

    :param ratings: User-item ratings DataFrame (users as index, items as columns).
    :param implicit_feedback: Dictionary of implicit feedback interactions (keys - user id, values - ratings were given by users to item ids).
    """
    users_in_ratings = set(ratings.index)
    items_in_ratings = set(ratings.columns)

    # Ensure implicit feedback does not contain users not present in the ratings matrix
    for user in implicit_feedback.keys():
        if user not in users_in_ratings:
            raise ValueError(f"User {user} in implicit feedback is not present in the ratings DataFrame.")

    # Check if any user has more implicit interactions than the total number of items
    for user, interactions in implicit_feedback.items():
        if len(interactions) > len(items_in_ratings):
            raise ValueError(f"User {user} has {len(interactions)} implicit interactions, "
                             f"which exceeds the number of available items ({len(items_in_ratings)}).")

        # Ensure all items in implicit feedback exist in the ratings matrix
        for item in interactions.keys():
            if item not in items_in_ratings:
                raise ValueError(f"Item {item} in implicit feedback for user {user} is not present in the ratings DataFrame.")


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

    def predict(self, user_id: int, item_id: int):
        """
        Predicts the rating for a given user-item pair.

        :param user_id: User id.
        :param item_id: Item id.
        :return: Predicted rating.
        """

        # If both user and item are unknown, return the global mean rating
        if user_id not in self._users and item_id not in self._items:
            return self._global_mean

        # If the item is unknown, return the user's average rating
        if item_id not in self._items:
            return np.mean(self._ratings.loc[user_id][self._ratings.loc[user_id] > 0])

        # If the user is unknown, return the item's average rating
        if user_id not in self._users:
            return np.mean(self._ratings[item_id][self._ratings[item_id] > 0])

        # Otherwise, use the trained model to predict the rating
        predicted_rating, _, _ = self._predict_with_calculated_params(user_id, item_id)
        return predicted_rating

    def fit(self, ratings: pd.DataFrame, implicit_feedback: Dict[int, Dict[int, float]]):
        """
        Trains the SVD++ model using a ratings matrix.

        :param ratings: pd.DataFrame where index = user id, column = item id, values = explicit rating
        :param implicit_feedback: Dictionary of implicit feedback interactions (keys - user id, values - ratings were given by users to item ids).
        """
        _check_params(ratings, implicit_feedback)

        # Parameters initialization
        self._params_initialization(ratings, implicit_feedback)

        # Get non-zero rating indices
        user_indices, item_indices = np.where(ratings > 0)
        indices_pairs = [(ratings.index[r], ratings.columns[c]) for r, c in zip(user_indices, item_indices)]

        for epoch in range(self._n_epochs):
            for user_id, item_id in indices_pairs:
                self._gradient_descent_iteration(user_id, item_id)

    def _random_normal_initialization(self, tech_list: list[str]):
        """
        Initializes a matrix using a normal distribution.

        :param tech_list: Set of IDs used for index
        :return: A matrix initialized with values drawn from a normal distribution.
        """

        x_size = len(tech_list)
        y_size = self._n_factors

        random_values = np.random.normal(self._init_mean, self._init_std, (x_size, y_size))

        return pd.DataFrame(random_values, index=tech_list, columns=[i for i in range(self._n_factors)])

    def _params_initialization(self, ratings: pd.DataFrame, implicit_feedback: Dict[int, Dict[int, float]]):
        """
        Initializes model parameters and matrices.

        :param ratings: pd.DataFrame where index = user id, column = item id, values = explicit rating
        :param implicit_feedback: Dictionary of implicit feedback interactions (keys - user id, values - ratings were given by users to item ids).
        """
        self._ratings = ratings.fillna(0)

        self._users = ratings.index.to_list()
        self._items = ratings.columns.to_list()

        self._user_factors = self._random_normal_initialization(self._users)
        self._item_factors = self._random_normal_initialization(self._items)
        self._implicit_factors = self._random_normal_initialization(self._items)

        self._bias_user = pd.Series(np.zeros(len(self._users)), index=self._users)
        self._bias_item = pd.Series(np.zeros(len(self._items)), index=self._items)

        self._global_mean = np.mean(self._ratings[self._ratings > 0]) # Mean rating across known ratings

        self._implicit_feedback = implicit_feedback

    def _calculate_sqrt_n(self, user_id: int):
        """
        Computes the square root of the total implicit feedback interactions for a given user.

        :param user_id: ID of the user.
        :return: Square root of the sum of implicit feedback interactions or 1 if no interactions exist.
        """
        sum_wuj = sum(self._implicit_feedback[user_id].values()) if user_id in self._implicit_feedback else 0

        return np.sqrt(sum_wuj) if sum_wuj > 0 else 1

    def _calculate_y_sum(self, user_id: int):
        """
        Computes the sum of implicit feedback factors weighted by interactions for a given user.

        :param user_id: ID of the user.
        :return: Sum of weighted implicit feedback factors or a zero vector if no implicit feedback exists.
        """
        if user_id not in self._implicit_feedback or not self._implicit_feedback[user_id]:
            total_sum = np.zeros(self._n_factors)
        else:
            total_sum = sum(self._implicit_factors.loc[item_id] * weight for item_id, weight in self._implicit_feedback[user_id].items())

        return total_sum
    def _predict_with_calculated_params(self, user_id: int, item_id: int) -> tuple[float, float, float]:
        """
        Computes the predicted rating for a user-item pair.

        :param user_id: ID of the user.
        :param item_id: ID of the item.
        :return: A tuple containing the predicted rating, sqrt_n, and y_sum.
        """
        assert user_id in self._users, "The user is unknown"
        assert item_id in self._items, "The item is unknown"

        sqrt_n = self._calculate_sqrt_n(user_id)  # Compute square root of implicit feedback sum
        y_sum = self._calculate_y_sum(user_id)  # Compute sum of implicit factors weighted by feedback

        current_user_factors = self._user_factors.loc[user_id].to_numpy()
        current_item_factors = self._item_factors.loc[item_id].to_numpy()

        # Compute predicted rating using bias terms and latent factors
        predicted_rating = self._global_mean + self._bias_user[user_id] + self._bias_item[item_id] + np.dot(
            (current_user_factors + y_sum / sqrt_n), current_item_factors)

        return predicted_rating, sqrt_n, y_sum

    def _gradient_descent_iteration(self, user_id: int, item_id: int):
        """
        Performs a single iteration of gradient descent for a given user-item pair.

        :param user_id: ID of the user.
        :param item_id: ID of the item.
        """
        current_rating = self._ratings.at[user_id, item_id] # Retrieve the actual explicit rating
        predicted_rating, sqrt_n, y_sum = self._predict_with_calculated_params(user_id, item_id) # Predict rating

        error = current_rating - predicted_rating # Compute the prediction error

        # Update user and item biases using gradient descent
        # Formula: bu = bu + lr_bu * (error - reg_bu * bu)
        # Formula: bi = bi + lr_bi * (error - reg_bi * bi)
        self._bias_user[user_id] += self._lr_bu * (error - self._reg_bu * self._bias_user[user_id])
        self._bias_item[item_id] += self._lr_bi * (error - self._reg_bi * self._bias_item[item_id])

        # Store copies of current factor vectors for calculations to not use updated versions of the coefficient
        current_user_factors = self._user_factors.loc[user_id].copy()
        current_item_factors = self._item_factors.loc[item_id].copy()

        # Update user latent factors:
        # Formula: pu = pu + lr_pu * (error * qi - reg_pu * pu)
        self._user_factors.loc[user_id] += self._lr_pu * (
                    error * current_item_factors - self._reg_pu * current_user_factors)

        # Update item latent factors:
        # Formula: qi = qi + lr_qi * (error * (pu + y_sum/sqrt_n) - reg_qi * qi)
        self._item_factors.loc[item_id] += self._lr_qi * (
                error * (current_user_factors + y_sum / sqrt_n) - self._reg_qi * current_item_factors)

        # Update implicit factors for all interacted items
        for j_item_id in self._implicit_feedback.get(user_id, {}).keys():
            current_implicit_rating = self._implicit_feedback[user_id][j_item_id]

            # Formula: yj = yj + lr_yj * (error * qi * implicit_rating / sqrt_n - reg_yj * yj)
            self._implicit_factors.loc[j_item_id] += self._lr_yj * (
                    error * current_item_factors * current_implicit_rating / sqrt_n - self._reg_yj *
                    self._implicit_factors.loc[j_item_id])





