from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# The function for extraction explicit ratings out of provided dataset
def get_explicit_rating(df: pd.DataFrame, user_field: str, item_field: str, rating_field: str, date_field: str) -> \
tuple[csr_matrix, csr_matrix, dict[str, int], dict[str, int]]:
    """
    Creates a user-item explicit rating matrix from a given dataset.

    :param df: The input DataFrame containing user, item, and rating data.
    :param user_field: The column name representing users.
    :param item_field: The column name representing items.
    :param rating_field: The column name representing explicit ratings.
    :param date_field: The column name representing date of review.

    :return: A tuple containing:
             - A sparse CSR matrix where rows represent users, columns represent items,
               and values represent the mean rating given by users to items
             - A sparse CSR matrix where rows represent users, columns represent items,
               and values represent the time of the last review was given from user to item
             - A dictionary mapping user IDs to matrix row indices.
             - A dictionary mapping item IDs to matrix column indices.
    """
    # Create user and item index mappings
    user_ids = df[user_field].unique()
    item_ids = df[item_field].unique()

    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    # Convert user and item IDs to corresponding indices
    df["user_idx"] = df[user_field].map(user_to_index)
    df["item_idx"] = df[item_field].map(item_to_index)

    # Aggregate data: Compute mean rating and latest review timestamp
    rating_agg = df.groupby(["user_idx", "item_idx"])[rating_field].mean().reset_index()
    latest_review_agg = df.groupby(["user_idx", "item_idx"])[date_field].max().reset_index()

    # Create sparse matrices
    rating_matrix = csr_matrix((rating_agg[rating_field],
                                (rating_agg["user_idx"], rating_agg["item_idx"])),
                               shape=(len(user_ids), len(item_ids)))

    latest_review_matrix = csr_matrix((latest_review_agg[date_field],
                                       (latest_review_agg["user_idx"], latest_review_agg["item_idx"])),
                                      shape=(len(user_ids), len(item_ids)))

    return rating_matrix, latest_review_matrix, user_to_index, item_to_index


# The functions for extraction implicit ratings out of explicit ratings provided in the dataset:
# - An assumption: the amount of positive reviews can be interpreted as a level of engagement that a particular user has got from a particular business
# - Only ratings above or equal to **4** (explicit ratings are from 1 to 5) are considered
# - Negative ratings aren't considered since it would be necessary to create negative ratings for them, but SVD++ doesn't work with them (its assumption is that all the ratings are not negative)
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
        user_item_dict[user][item] += count

    return user_item_dict


def split_matrix(ratings: csr_matrix, timestamps: csr_matrix, ratios: list) -> list[csr_matrix]:
    """
    Algorithm Description:
    This function splits the input matrices (ratings and timestamps) into multiple output matrices
    based on specified ratios. Each output matrix contains a portion of the original data, selected by sorting
    the elements in each row by timestamp (descending) and then dividing them according to the given proportions.

    Steps:
    1. Validate that the input matrices have the same shape and that the ratios sum to 1.
    2. Initialize containers to hold the split data for each output matrix.
    3. For each row in the matrices:
        a. Extract ratings, timestamps, and column indices for that row.
        b. Sort the entries in descending order of timestamp.
        c. Partition the sorted entries according to the provided ratios.
        d. Append each partition's data, indices, and row pointer to the respective output structure.
    4. After all rows are processed, reconstruct matrices from the collected data for each partition.
    5. Return the list of resulting matrices.


    :param ratings: matrix containing ratings
    :param timestamps: matrix containing timestamps
    :param ratios: List of ratios (must sum to 1)

    :return: List of sparse matrices (ratings) corresponding to the given ratios
    """
    if ratings.shape != timestamps.shape:
        raise ValueError("Ratings and timestamps matrices must have the same shape")

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Sum of ratios must be equal to 1")

    num_parts = len(ratios)
    n_rows = ratings.shape[0]

    # Initialize lists to store data for new sparse matrices
    new_data = [[] for _ in range(num_parts)]
    new_indices = [[] for _ in range(num_parts)]
    new_indptr = [[0] for _ in range(num_parts)]

    for i in range(n_rows):
        row_start = ratings.indptr[i]
        row_end = ratings.indptr[i + 1]

        # Extract the row data from the original matrices
        row_ratings = ratings.data[row_start:row_end]
        row_timestamps = timestamps.data[row_start:row_end]
        row_indices = ratings.indices[row_start:row_end]

        # Sort indices based on timestamps in descending order
        sorted_indices = np.argsort(row_timestamps)[::-1]
        row_ratings = row_ratings[sorted_indices]
        row_indices = row_indices[sorted_indices]

        # Split data into parts based on given ratios
        start_idx = 0
        for j, ratio in enumerate(ratios):
            end_idx = start_idx + int(len(row_ratings) * ratio)

            # Store partitioned data into corresponding lists
            new_data[j].extend(row_ratings[start_idx:end_idx])
            new_indices[j].extend(row_indices[start_idx:end_idx])
            new_indptr[j].append(new_indptr[j][-1] + (end_idx - start_idx))

            # Update start index for the next partition
            start_idx = end_idx

    # Construct new CSR matrices from partitioned data
    result_matrices = [
        csr_matrix((new_data[j], new_indices[j], new_indptr[j]), shape=ratings.shape)
        for j in range(num_parts)
    ]

    return result_matrices

