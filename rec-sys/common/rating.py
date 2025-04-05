import math

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


def sanity_check_explicit_matrix(explicit_ratings: csr_matrix, last_dates: csr_matrix, review_df: pd.DataFrame,
                                 user_field: str, item_field: str) -> pd.DataFrame:
    """
    Performs a consistency check between explicit rating matrices and the original DataFrame.

    The function compares:
    - Number of non-zero entries (interactions) in the explicit_ratings matrix
    - Number of non-zero entries in the last_dates matrix
    - Number of unique (user_id, business_id) pairs in the original review DataFrame

    :param explicit_ratings: CSR matrix where each non-zero entry represents
                              the average rating from a user to a business.
    :param last_dates: CSR matrix where each non-zero entry represents
                       the timestamp of the last review from a user to a business.
    :param review_df: Filtered DataFrame containing user, business, and review data.
    :param user_field: The column name representing users.
    :param item_field: The column name representing items.

    :return: A DataFrame summarizing the number of interactions and unique pairs across sources.
    """
    # Count of interactions in each matrix
    num_ratings = explicit_ratings.nnz
    num_dates = last_dates.nnz

    # Number of unique (user_id, business_id) pairs in the source DataFrame
    num_unique_pairs = review_df.groupby([user_field, item_field]).ngroups

    # Build summary table
    sanity_df = pd.DataFrame({
        'Source': ['Explicit ratings matrix', 'Last dates matrix', 'Filtered review DataFrame'],
        'Calculated metrics': ['Non-zero entries', 'Non-zero entries', 'Unique (user_id, business_id) pairs'],
        'Value': [num_ratings, num_dates, num_unique_pairs]
    })

    return sanity_df


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
    user_item_dict = {}

    for user, item, count in interaction_counts.itertuples(index=False):
        if user not in user_item_dict:
            user_item_dict[user] = {}

        if item not in user_item_dict[user]:
            user_item_dict[user][item] = 0

        user_item_dict[user][item] += count

    return user_item_dict


def sanity_check_implicit_rating(initial_df, implicit_ratings, implicit_threshold, user_field: str, item_field: str,
                                 rating_field: str):
    """
        Prints a sanity check table comparing filtered reviews with the implicit_ratings structure.

        The function compares:
        - Total number of reviews with stars >= IMPLICIT_THRESHOLD
        - Total number of interactions recorded in implicit_ratings
        - Number of unique users and businesses in both datasets

        :param initial_df: DataFrame containing the original review data.
        :param implicit_ratings: Dictionary of the form {user_id: {business_id: count}}, representing the derived implicit interactions.
        :param implicit_threshold: Minimum star rating to be considered a positive implicit interaction.
        :param user_field: The column name representing the user.
        :param item_field: The column name representing the item.
        :param rating_field: The column name representing the rating.

        :return: A DataFrame containing the sanity check metrics.
        """
    # Filter reviews above the threshold
    fit_reviews = initial_df[initial_df[rating_field] >= implicit_threshold]

    # Length of filtered reviews
    len_of_fit_reviews = len(fit_reviews)

    # Total interactions stored in implicit_ratings
    len_of_review_in_implicit_rating = sum(
        sum(business_dict.values()) for business_dict in implicit_ratings.values()
    )

    # Unique user counts
    users_in_fit_reviews = fit_reviews[user_field].nunique()
    users_in_implicit_ratings = len(implicit_ratings)

    # Unique business counts
    businesses_in_fit_reviews = fit_reviews[item_field].nunique()
    unique_business_ids = {
        business_id
        for business_dict in implicit_ratings.values()
        for business_id in business_dict
    }
    businesses_in_implicit_ratings = len(unique_business_ids)

    # Create a DataFrame to display the check
    sanity_df = pd.DataFrame({
        'Metric': [
            'Number of reviews (stars >= threshold)',
            'Number of reviews in implicit_ratings',
            'Unique users in initial reviews',
            'Unique users in implicit_ratings',
            'Unique items in initial reviews',
            'Unique items in implicit_ratings',
        ],
        'Value': [
            len_of_fit_reviews,
            len_of_review_in_implicit_rating,
            users_in_fit_reviews,
            users_in_implicit_ratings,
            businesses_in_fit_reviews,
            businesses_in_implicit_ratings,
        ]
    })

    return sanity_df


def split_matrix_csr(ratings: csr_matrix, timestamps: csr_matrix, ratios: list) -> list[csr_matrix]:
    """
    Algorithm Description:
    This function splits the input matrices (ratings and timestamps) into multiple output matrices
    based on specified ratios. Each output matrix contains a portion of the original data, selected by sorting
    the elements in each row by timestamp (descending) and then dividing them according to the given proportions.
    Steps:
    1. Validate that the ratings and timestamps matrices have the same shape.
    2. Ensure the sum of all ratios is equal to 1.0.
    3. Initialize empty containers (data, indices, indptr) for each target split.
    4. Precompute the ideal number of non-zero entries per split based on the total nnz and the given ratios.
    5. For each row in the matrix:
        1. Extract non-zero ratings, timestamps, and their column indices.
        2. Sort the entries by timestamp in descending order (most recent first).
        3. Iterate through each ratio:
            * Compute how many elements to assign to the current split.
            * Use floor or round to balance the global target distribution.
            * For the last split, assign all remaining elements to ensure no data is lost.
            * Append the selected entries to the corresponding split containers.
    6. Construct new CSR matrices for each split using the collected data.
    7. Return the list of split CSR matrices.


    :param ratings: matrix containing ratings
    :param timestamps: matrix containing timestamps
    :param ratios: List of ratios (must sum to 1) - the first ratios will be from the newest timestamps

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

    # Initialize ideal amount of filled cells according to ratios (float)
    filled_sells_according_to_ratios = [ratings.nnz * ratio for ratio in ratios]
    total_filled_cells = [0] * num_parts

    for i in range(n_rows):
        row_start = ratings.indptr[i]
        row_end = ratings.indptr[i + 1]

        # Extract the row data from the original matrices
        row_ratings = ratings.data[row_start:row_end]
        row_timestamps = timestamps.data[row_start:row_end]
        row_indices = ratings.indices[row_start:row_end]

        # Sort indices based on timestamps in descending order
        sorted_idx = np.argsort(row_timestamps)[::-1]
        sorted_ratings = row_ratings[sorted_idx]
        sorted_indices = row_indices[sorted_idx]

        # Split data into parts based on given ratios
        start_idx = 0
        for j, ratio in enumerate(ratios):
            # Calculate offset (round depends on the current amount in each split / for the last split assign the rest of elements)
            float_offset = len(sorted_ratings) * ratio if j < num_parts - 1 else len(sorted_ratings) - start_idx
            offset = round(float_offset) if total_filled_cells[j] >= filled_sells_according_to_ratios[
                j] * ratio else math.floor(float_offset)

            # Calculate end index and increase the total cells in the current split
            end_idx = min(start_idx + offset, len(sorted_ratings))
            total_filled_cells[j] += end_idx - start_idx

            # Store partitioned data into corresponding lists
            new_data[j].extend(sorted_ratings[start_idx:end_idx])
            new_indices[j].extend(sorted_indices[start_idx:end_idx])
            new_indptr[j].append(new_indptr[j][-1] + end_idx - start_idx)

            # Update start index for the next partition
            start_idx = end_idx

    # Construct new CSR matrices from partitioned data
    result_matrices = [
        csr_matrix((new_data[j], new_indices[j], new_indptr[j]), shape=ratings.shape)
        for j in range(num_parts)
    ]

    return result_matrices


def sanity_check_explicit_split(train_matrix: csr_matrix, validation_matrix: csr_matrix, test_matrix,
                                explicit_matrix: csr_matrix) -> pd.DataFrame:
    """
    Generates a sanity check summary for CSR matrices produced by an explicit rating split.

    The function calculates:
    - Number of interactions (non-zero entries) in each split
    - Total number of interactions
    - Percentage distribution across train, validation, and test matrices

    :param train_matrix: CSR matrix representing the training set.
    :param validation_matrix: CSR matrix representing the validation set.
    :param test_matrix: CSR matrix representing the test set.
    :param explicit_matrix: CSR matrix representing the explicit ratings that were split.

    :return: DataFrame summarizing the number and percentage of interactions per split.
    """
    # Count non-zero elements (interactions) in each matrix
    train_n = train_matrix.nnz
    validation_n = validation_matrix.nnz
    test_n = test_matrix.nnz

    sum_n = train_n + validation_n + test_n
    factual_n = explicit_matrix.nnz

    # Calculate percentages (integer division)
    calculate_pct = lambda n: round(n * 100 / factual_n, 2)

    train_pct = calculate_pct(train_n)
    validation_pct = calculate_pct(validation_n)
    test_pct = calculate_pct(test_n)
    sum_pct = calculate_pct(sum_n)

    # Create and return the summary DataFrame
    summary_df = pd.DataFrame({
        'Split': ['Train', 'Validation', 'Test', 'Explicit total', 'Factual total'],
        'Number of interactions': [train_n, validation_n, test_n, sum_n, factual_n],
        'Part of factual interactions': [f"{train_pct}%", f"{validation_pct}%", f"{test_pct}%",
                                         f"{sum_pct}%", "100%"]
    })

    return summary_df
