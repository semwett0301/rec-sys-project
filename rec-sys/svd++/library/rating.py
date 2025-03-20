from collections import defaultdict

import pandas as pd
from scipy.sparse import csr_matrix

# The function for extraction explicit ratings out of provided dataset
def get_explicit_rating(df: pd.DataFrame, user_field: str, item_field: str, rating_field: str) -> tuple[
    csr_matrix, dict[str, int], dict[str, int]]:
    """
    Creates a user-item explicit rating matrix from a given dataset.

    :param df: The input DataFrame containing user, item, and rating data.
    :param user_field: The column name representing users.
    :param item_field: The column name representing items.
    :param rating_field: The column name representing explicit ratings.

    :return: A tuple containing:
             - A sparse CSR matrix where rows represent users, columns represent items,
               and values represent the mean rating given by users to items.
             - A dictionary mapping user IDs to matrix row indices.
             - A dictionary mapping item IDs to matrix column indices.
    """
    # Create user and item index mappings
    user_ids = df[user_field].unique()
    item_ids = df[item_field].unique()

    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    # Convert user and item IDs to corresponding indices
    row_indices = df[user_field].map(user_to_index).values
    col_indices = df[item_field].map(item_to_index).values
    data = df[rating_field].values

    # Create sparse matrix
    sparse_matrix = csr_matrix((data, (row_indices, col_indices)),
                               shape=(len(user_ids), len(item_ids)))

    return sparse_matrix, user_to_index, item_to_index


# The function for extraction implicit ratings out of explicit ratings provided in the dataset:
# - An assumption: the amount of positive reviews can be interpreted as a level of engagement that a particular user has got from a particular business
# - Only ratings above or equal to **4** (explicit ratings are from 1 to 5) are considered
# - Negative ratings aren't considered since it would be necessary to create negative ratings for them, but SVD++ doesn't work with them (its assumption is that all the ratings are positive)
def get_implicit_rating_out_of_positive_ratings(df: pd.DataFrame, user_field: str, item_field: str,
                                                rating_field: str, implicit_threshold: int) -> dict:
    """
    Converts a DataFrame into a dictionary {user_id: {item_id: number of times rating >= implicit_threshold}}.

    :param df: The input DataFrame containing user, item, and rating data.
    :param user_field: The column name representing the user ID.
    :param item_field: The column name representing the item ID.
    :param rating_field: The column name representing the rating.
    :param implicit_threshold: Threshold for selecting positive ratings

    :return: A dictionary in the format {user_id: {item_id: number_of_interactions}}.
    """
    # Filter out interactions where the rating is below the implicit threshold
    filtered_df = df[df[rating_field] >= implicit_threshold]

    # Count the number of times each user-item pair meets the threshold
    interaction_counts = filtered_df.groupby([user_field, item_field]).size().reset_index(name='count')

    # Convert the result into a nested dictionary structure {user_id: {item_id: count}}
    user_item_dict = defaultdict(dict)
    for user, item, count in interaction_counts.itertuples(index=False):
        user_item_dict[user][item] = count

    return user_item_dict
