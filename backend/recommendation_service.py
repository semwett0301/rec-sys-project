import pickle

import pandas as pd

import sys

sys.path.append("../rec-sys/svd++/library")
sys.path.append("../rec-sys/item-knn/library")
sys.path.append("../rec-sys/content-based/library")
sys.path.append("../rec-sys/common")

RATING_MODELS = ['content-user', 'content-item']

BASE_PATH = "../models"
BASE_DATASET_PATH = "../eda/dataset_samples"

FILE_PATHS = {
    'svd-yelp': f'{BASE_PATH}/svd_pp_yelp.pkl',
    'svd-movie': f'{BASE_PATH}/svd_pp_movie_lens.pkl',
    'collab-movie': f'{BASE_PATH}/item_knn_colab_movie_lens.pkl',
    'collab-netflix': f'{BASE_PATH}/item_knn_colab_netflix.pkl',
    'content-user-yelp': f'{BASE_PATH}/content_based_user_knn_yelp.pkl',
    'content-item-yelp': f'{BASE_PATH}/content_based_item_knn_yelp.pkl',
    'content-user-movie': f'{BASE_PATH}/content_based_user_knn_movie_lens.pkl',
    'content-item-movie': f'{BASE_PATH}/content_based_item_knn_movie_lens.pkl',
}

DATASETS_PATHS = {
    'yelp': f'{BASE_DATASET_PATH}/df_yelp_review_open_health_10.parquet',
    'netflix': f'{BASE_DATASET_PATH}/sampled_netflix.parquet',
    'movie': f'{BASE_DATASET_PATH}/df_movie_lens.parquet'
}

ITEMS_FEATURES = {
    'yelp': 'business_id',
    'movie': 'movie_id',
    'netflix': 'movie_id',
}


class RecommendationService:
    def __init__(self):
        models = {}
        datasets = {}

        for key, path in DATASETS_PATHS.items():
            datasets[key] = pd.read_parquet(path)

        for key, path in FILE_PATHS.items():
            with open(path, 'rb') as f:
                models[key] = pickle.load(f)

        self._models = models
        self._dataset = datasets

    def recommend(self, model: str, dataset: str, user_id: str, top_n: int) -> list[str]:
        return self._predict_list_with_score(model, dataset, user_id, top_n) if model not in RATING_MODELS else \
            self._models[f"{model}-{dataset}"].predict(user_id, top_n)

    def _predict_list_with_score(self, model: str, dataset: str, user_id: str, top_n: int) -> list[str]:
        model_obj = self._models[f"{model}-{dataset}"]
        dataset_obj = self._dataset[dataset]
        item_feature = ITEMS_FEATURES[dataset]

        ratings = []

        for item_id in dataset_obj[item_feature].unique():
            ratings.append((model_obj.predict(user_id, item_id), item_id))

        return [s for _, s in sorted(ratings, key=lambda x: x[0], reverse=True)[:top_n]]




