from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# The function for extraction explicit ratings out of provided dataset
def get_explicit_rating(df: pd.DataFrame, user_field: str, item_field: str, rating_field: str, date_field: str) -> \
        tuple[csr_matrix, csr_matrix, dict[str, dict[str, int]], dict[str, dict[str, int]]]:
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
             - A dictionary with two dictionaries mapping user IDs to matrix row indices and vice versa.
             - A dictionary with two dictionaries mapping item IDs to matrix column indices and vice versa.
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

    # Create sparse matrice of explicit rating
    rating_matrix = csr_matrix((rating_agg[rating_field],
                                (rating_agg["user_idx"], rating_agg["item_idx"])),
                               shape=(len(user_ids), len(item_ids)))

    # Create sparse matrice of last review timestamp
    latest_review_matrix = csr_matrix((latest_review_agg[date_field],
                                       (latest_review_agg["user_idx"], latest_review_agg["item_idx"])),
                                      shape=(len(user_ids), len(item_ids)))

    # Form mappers (object that contains maps that let to convert id to idx and vice versa)
    user_mapper = {
        'id_to_idx': user_to_index,
        'idx_to_id': {v: k for k, v in user_to_index.items()}
    }

    item_mapper = {
        'id_to_idx': item_to_index,
        'idx_to_id': {v: k for k, v in item_to_index.items()}
    }

    return rating_matrix, latest_review_matrix, user_mapper, item_mapper


# The functions for extraction implicit ratings out of explicit ratings provided in the dataset:
# - An assumption: the amount of positive reviews can be interpreted as a level of engagement that a particular user has got from a particular business
# - Only ratings above or equal to **4** (explicit ratings are from 1 to 5) are considered
# - Negative ratings aren't considered since it would be necessary to create negative ratings for them, but SVD++ doesn't work with them (its assumption is that all the ratings are not negative)
def get_implicit_rating_out_of_positive_ratings_csr(matrix: csr_matrix, idx_to_user_id: dict[int, str],
                                                    idx_to_item_id: dict[int, str], implicit_threshold: int) -> dict:
    """
    Converts a sparse user-item rating matrix into a nested dictionary format based on a rating threshold.

    Each entry in the output represents how many times a user has interacted positively
    (i.e., rating >= implicit_threshold) with a specific item.

    :param idx_to_user_id: a map that contains indexes of columns in csr matrix as a key and item id as a value
    :param idx_to_item_id: a map that contains indexes of rows in csr matrix as a key and user id as a value
    :param matrix: A CSR sparse matrix where rows are users, columns are items, and data contains ratings.
    :param implicit_threshold: Minimum rating considered as a positive (implicit) interaction.

    :return: A dictionary in the format {user_id: {item_id: <number_of_positive_reviews>}}.
    """

    user_item_dict = defaultdict(dict)

    # Iterate over each user (row in the matrix)
    for user_idx in range(matrix.shape[0]):
        # Get the start and end pointers for the current row in the CSR format
        start_ptr, end_ptr = matrix.indptr[user_idx], matrix.indptr[user_idx + 1]

        # Get the item indices and their corresponding ratings for this user
        item_ids = matrix.indices[start_ptr:end_ptr]
        ratings = matrix.data[start_ptr:end_ptr]

        # Count positive interactions
        for item_idx, rating in zip(item_ids, ratings):
            if rating >= implicit_threshold:
                user_id, item_id = idx_to_user_id[user_idx], idx_to_item_id[item_idx]
                user_item_dict[user_id][item_id] = user_item_dict[user_id].get(item_id, 0) + 1

    return user_item_dict


# The functions for extraction implicit ratings out of explicit ratings provided in the dataset:
# - An assumption: the amount of positive reviews can be interpreted as a level of engagement that a particular user has got from a particular business
# - Only ratings above or equal to **4** (explicit ratings are from 1 to 5) are considered
# - Negative ratings aren't considered since it would be necessary to create negative ratings for them, but SVD++ doesn't work with them (its assumption is that all the ratings are not negative)
def get_implicit_rating_out_of_positive_ratings_df(df: pd.DataFrame, user_field: str, item_field: str,
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


def split_matrix_csr(ratings: csr_matrix, timestamps: csr_matrix, ratios: list) -> list[csr_matrix]:
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
