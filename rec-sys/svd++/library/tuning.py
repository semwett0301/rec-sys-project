import logging

from scipy.sparse import csr_matrix
from sklearn.model_selection import ParameterGrid

from metrics import RmseCalculator
from svdpp import SVDpp

class GridSearchSvdPP:
    """
    Class to perform grid search for hyperparameter tuning of the SVD++ model.
    """

    def __init__(self,
                 train_matrix: csr_matrix,
                 implicit_rating: dict,
                 val_matrix: csr_matrix,
                 user_mapping: dict[str, dict],
                 item_mapping: dict[str, dict],
                 param_grid):
        """
        Initializes the grid search object with data and parameters.

        :param train_matrix: CSR-format sparse matrix of explicit ratings for training.
        :param implicit_rating: Dictionary of implicit feedback (e.g., views, clicks).
        :param val_matrix: CSR-format sparse matrix of validation ratings.
        :param user_mapping: Dicts for user ID to index and index to user ID mappings.
        :param item_mapping: Dicts for item ID to index and index to item ID mappings.
        :param param_grid: Dictionary of hyperparameters to search over.
        """
        self._param_grid = ParameterGrid(param_grid)

        self._train_matrix = train_matrix
        self._val_matrix = val_matrix
        self._implicit_rating = implicit_rating

        self._user_mapping = user_mapping
        self._item_mapping = item_mapping

    def _create_model(self, params):
        """
        Instantiates and fits the SVD++ model with the given parameters.

        :param params: Dictionary of hyperparameters.
        :return: Trained SVDpp model.
        """
        model = SVDpp(**params)
        model.fit(ratings=self._train_matrix, implicit_rating=self._implicit_rating,
                  user_id_to_idx=self._user_mapping['id_to_idx'], item_id_to_idx=self._item_mapping['id_to_idx'])

        return model

    def _create_metrics_calculator(self, model: SVDpp):
        """
        Creates an evaluation object for computing metrics on the validation set.

        :param model: Trained SVD++ model.
        :return: SvdMetricsCalculator instance.
        """
        return RmseCalculator(
            model=model,
            idx_to_item_id=self._item_mapping['idx_to_id'],
            idx_to_user_id=self._user_mapping['idx_to_id'],
            matrix=self._val_matrix
        )

    def run(self) -> tuple[tuple[int] | None, float, SVDpp | None]:
        """
        Runs grid search over all parameter combinations, evaluates each model,
        and returns the best configuration based on RMSE.

        :return: Tuple of (best parameters, best RMSE score, best model).
        """
        best_score = float("inf") # Initialize with the worst possible RMSE
        best_params = None
        best_model = None

        for idx, params in enumerate(self._param_grid):
            logging.info(f"Try number: {idx + 1}")
            logging.info(f"Train with params: {params}")

            # Train model with current hyperparameters
            model = self._create_model(params)

            # Compute validation RMSE
            metric_calculator = self._create_metrics_calculator(model)
            common_metric = metric_calculator.calculate_rmse()
            logging.info(f"Current common score: {common_metric}")

            if common_metric < best_score:
                best_score = common_metric
                best_params = params
                best_model = model

        return best_params, best_score, best_model
