import numpy as np
from scipy.sparse import csr_matrix
from svdpp import SVDpp


class SvdMetricsCalculator:
    def __init__(self, test_matrix: csr_matrix, model: SVDpp, idx_to_user_id: dict[int, str],
                 idx_to_item_id: dict[int, str]):
        self._model = model
        self._test_matrix = test_matrix

        result_lil = test_matrix.tolil()

        coo = test_matrix.tocoo()
        for row, col in zip(coo.row, coo.col):
            result_lil[row, col] = model.predict(idx_to_user_id[row], idx_to_item_id[col])

        self._result_matrix = result_lil.tocsr()

    def calculate_rmse(self) -> float:
        return np.sqrt(np.mean((self._test_matrix.data - self._result_matrix.data) ** 2))

    def calculate_common_metric(self, explicit_rating_max: int) -> float:
        rmse = self.calculate_rmse()

        return rmse / explicit_rating_max
