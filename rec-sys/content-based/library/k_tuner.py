import logging

import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix

from item_knn_content import ItemKnnContent
from metrics import RankingMetricsEvaluator, RankingPredictable
from user_knn_content import UserKnnContent


def plot_k_vs_f1(k, f1_score, method_type):
    """
    Plots the F1 score for different values of k.

    :param k: List of k values.
    :param f1_score: Corresponding list of F1 scores.
    :param method_type: String indicating the model type ("user" or "item").
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k, f1_score, marker='o')
    plt.title(f'Validation F-1 vs. K in {method_type} content-based')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Validation F1')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class KTuner:
    def __init__(
            self,
            val_matrix: csr_matrix,
            similarity_features: pd.DataFrame,
            user_mapping: dict[str, dict],
            item_mapping: dict[str, dict],
            k_values: list[int],
            type: str = 'item',
            top_n: int = 10
    ):
        """
        Initializes the KTuner to evaluate and tune the best K value for KNN-based recommenders.

        :param val_matrix: Validation user-item interaction matrix (sparse format).
        :param similarity_features: Feature matrix for similarity computation (items or users).
        :param user_mapping: Dictionary with 'id_to_idx' and 'idx_to_id' for users.
        :param item_mapping: Dictionary with 'id_to_idx' and 'idx_to_id' for items.
        :param k_values: List of K values to evaluate.
        :param type: Either "item" or "user" for model type.
        :param top_n: Number of top items to recommend per user.
        :raises ValueError: If the type is not "item" or "user".
        """
        if type not in ['item', 'user']:
            raise ValueError('type must be either "item" or "user"')

        self._type = type
        self._val_matrix = val_matrix
        self._similarity_features = similarity_features
        self._user_mapping = user_mapping
        self._item_mapping = item_mapping
        self._k_values = k_values
        self._top_n = top_n

    def tune(self) -> tuple[RankingPredictable, int, float]:
        """
        Tunes and selects the best K for the chosen KNN model by evaluating F1 scores.

        :return: A tuple containing:
            - The best-performing model (RankingPredictable),
            - The best value of K (int),
            - The best F1 score (float).
        """
        calculated_f1_scores = []

        best_k = None
        best_f1 = -1.0
        best_model = None

        for k in self._k_values:
            logging.info(f"Evaluating k={k}...")

            # Prepare model parameters
            params_for_training = {
                'rating_matrix': self._val_matrix,
                'similarity_features': self._similarity_features,
                'user_mapping': self._user_mapping,
                'item_mapping': self._item_mapping,
                'k': k
            }

            # Instantiate model based on type
            model = ItemKnnContent(**params_for_training) if self._type == 'item' else UserKnnContent(**params_for_training)

            # Evaluate model using validation matrix
            evaluator = RankingMetricsEvaluator(
                matrix=self._val_matrix,
                model=model,
                user_mapping=self._user_mapping,
                item_mapping=self._item_mapping,
                top_n=self._top_n
            )

            # Calculate F1 score for the current model
            f1 = evaluator.calculate_f1()
            calculated_f1_scores.append(f1)
            logging.info(f"F1 score @ top-{self._top_n} for k={k}: {f1:.4f}")

            # Track the best model and its parameters
            if f1 > best_f1:
                best_f1 = f1
                best_k = k
                best_model = model

        logging.info(f"\nBest k: {best_k} with F1 score: {best_f1:.4f}")

        # Plot F1 vs. K
        plot_k_vs_f1(self._k_values, calculated_f1_scores, self._type)

        return best_model, best_k, best_f1
