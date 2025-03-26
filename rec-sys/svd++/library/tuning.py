from typing import Callable, Tuple, Any
from venv import logger

from scipy.sparse import csr_matrix
from sklearn.model_selection import ParameterGrid

from metrics import SvdMetricsCalculator
from svdpp import SVDpp

_DEFAULT_PARAM_GRID = {
    'n_factors': [10, 20, 30],
    'n_epochs': [10, 20],
    'lr_all': [0.005, 0.007],
    'reg_all': [0.1]
}

class GridSearchSvdPP:
    def __init__(self,
                 train_matrix: csr_matrix,
                 train_implicit_rating: dict,
                 val_matrix: csr_matrix,
                 user_mapping: dict[str, dict],
                 item_mapping: dict[str, dict],
                 param_grid: dict[str, list] = _DEFAULT_PARAM_GRID):
        self._param_grid = ParameterGrid(param_grid)

        self._train_matrix = train_matrix
        self._val_matrix = val_matrix
        self._train_implicit_rating = train_implicit_rating

        self._user_mapping = user_mapping
        self._item_mapping = item_mapping

    def _create_model(self, params):
        model = SVDpp(**params)
        model.fit(ratings=self._train_matrix, implicit_rating=self._train_implicit_rating,
                  user_id_to_idx=self._user_mapping['id_to_idx'], item_id_to_idx=self._item_mapping['id_to_idx'])

        return model

    def _create_metrics_calculator(self, model: SVDpp):
        return SvdMetricsCalculator(model=model, idx_to_item_id=self._item_mapping['idx_to_id'],
                                    idx_to_user_id=self._user_mapping['idx_to_id'], test_matrix=self._val_matrix)

    def run(self, explicit_rating_max: int) -> tuple[tuple[int] | None, float, SVDpp | None]:
        best_score = float("inf")
        best_params = None
        best_model = None

        for idx, params in enumerate(self._param_grid):
            logger.info(f"Try number: {idx + 1}")
            logger.info(f"Train with params: {params}")

            model = self._create_model(params)

            metric_calculator = self._create_metrics_calculator(model)
            common_metric = metric_calculator.calculate_common_metric(explicit_rating_max)
            logger.info(f"Current common score: {common_metric}")

            if common_metric < best_score:
                best_score = common_metric
                best_params = params
                best_model = model

        return best_params, best_score, best_model
