import numpy as np
from scipy.sparse import csr_matrix
import logging

def categorize_ids(arrays):
    """
    Categorizes unique IDs into groups based on which arrays they appear in.

    Uses sparse matrices for efficient membership tracking.

    Parameters:
    - arrays (list of np.ndarray): List of numpy arrays containing IDs.

    Returns:
    - unique_ids (np.ndarray): Sorted array of all unique IDs.
    - presence_matrix (scipy.sparse.csr_matrix): Sparse binary matrix where (i, j) is 1 if unique_ids[j] appears in arrays[i].
    - category_mask (np.ndarray): Boolean mask indicating which IDs are in each category.

    Example:
    >>> ARRAY_1 = np.array([4, 5, 6, 7, 8, 2, 4, 1, 2, 3, 6, 6, 1, 10, 23])
    >>> ARRAY_2 = np.array([5, 6, 7, 8, 9, 10, 15, 6, 9])
    >>> ARRAY_3 = np.array([2, 5, 10, 11, 12, 6, 23, 7])
    >>> ID_ARRAYS = [ARRAY_1, ARRAY_2, ARRAY_3]
    >>> unique_ids, presence_matrix, category_mask = categorize_ids(ID_ARRAYS)
    >>> unique_ids
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 15, 23])
    >>> presence_matrix.toarray()
    array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1]])
    >>> category_mask
    array([[ True,  True,  True,  True,  True,  True,  True,  True, False,  True, False, False, False,  True],
           [False, False, False, False,  True,  True,  True,  True,  True,  True, False, False,  True, False],
           [False,  True, False, False,  True,  True,  True, False, False,  True,  True,  True, False,  True]])
    """

    logging.info("├── 🚀 Starting ID categorization process...")

    if not arrays or any(arr.size == 0 for arr in arrays):
        logging.error("|    ├── ❌ ERROR: One or more input arrays are empty.")
        raise ValueError("All input arrays must be non-empty.")

    # Extract unique IDs
    unique_ids = np.unique(np.concatenate(arrays))
    logging.info(f"|    ├── 📌 Found {unique_ids.size} unique IDs across {len(arrays)} arrays.")

    # Build the presence matrix efficiently
    try:
        row_indices = np.concatenate([np.full(len(arr), i) for i, arr in enumerate(arrays)])
        col_indices = np.searchsorted(unique_ids, np.concatenate(arrays))
        presence_matrix = csr_matrix((np.ones_like(col_indices), (row_indices, col_indices)), 
                                     shape=(len(arrays), len(unique_ids)))
        logging.info(f"|    ├── Presence matrix created with shape: {presence_matrix.shape}")
    except Exception as e:
        logging.error(f"|    ├── ❌ Error constructing presence matrix: {e}")
        raise

    # Convert to a boolean mask for easy indexing
    category_mask = presence_matrix.toarray().astype(bool)
    logging.info(f"|    ├── Category mask created with shape: {category_mask.shape}")

    return unique_ids, presence_matrix, category_mask

def get_all_indices(arrays, unique_ids):
    """
    Finds all occurrences of each unique ID in all target arrays efficiently.
    
    Uses `np.argsort` and `np.searchsorted` for vectorized lookup.

    Parameters:
    - arrays (list of np.ndarray): List of numpy arrays.
    - unique_ids (np.ndarray): Sorted array of all unique IDs.

    Returns:
    - indices_matrix (np.ndarray): 2D object array where each row corresponds to a unique ID, 
                                    and each column contains index arrays for the respective array.

    Example:
    >>> ARRAY_1 = np.array([4, 5, 6, 7, 8, 2, 4, 1, 2, 3, 6, 6, 1, 10, 23])
    >>> ARRAY_2 = np.array([5, 6, 7, 8, 9, 10, 15, 6, 9])
    >>> ARRAY_3 = np.array([2, 5, 10, 11, 12, 6, 23, 7])
    >>> ID_ARRAYS = [ARRAY_1, ARRAY_2, ARRAY_3]
    >>> unique_ids, _, _ = categorize_ids(ID_ARRAYS)
    >>> indices_matrix = get_all_indices(ID_ARRAYS, unique_ids)
    >>> indices_matrix[0]  # Indices for ID=1 in ARRAY_1, ARRAY_2, ARRAY_3
    [array([7, 12]), array([], dtype=int64), array([], dtype=int64)]
    >>> indices_matrix[4]  # Indices for ID=5 in all arrays
    [array([1]), array([0]), array([1])]
    >>> indices_matrix[6]  # Indices for ID=7 in all arrays
    [array([3]), array([2]), array([6])]
    """

    logging.info("├── 🚀 Starting index retrieval process...")

    if not arrays or any(arr.size == 0 for arr in arrays):
        logging.error("|    ├── ❌ ERROR: One or more input arrays are empty.")
        raise ValueError("All input arrays must be non-empty.")

    num_ids = len(unique_ids)
    num_arrays = len(arrays)
    logging.info(f"|    ├── 📌 Processing {num_ids} unique IDs across {num_arrays} arrays.")

    # Initialize an empty object array to store indices
    indices_matrix = np.empty((num_ids, num_arrays), dtype=object)

    # Efficiently find indices for each array
    for i, arr in enumerate(arrays):
        logging.debug(f"|    ├── 🔍 Processing array {i+1}/{num_arrays} (size={arr.size})...")

        sorted_idx = np.argsort(arr)  # Get sorting order
        sorted_arr = arr[sorted_idx]  # Sort array accordingly

        search_left = np.searchsorted(sorted_arr, unique_ids, side="left")
        search_right = np.searchsorted(sorted_arr, unique_ids, side="right")

        # Extract index ranges for each unique ID
        indices_matrix[:, i] = [sorted_idx[start:end] for start, end in zip(search_left, search_right)]
    
    return indices_matrix

def crossmatch_two_arrays(unique_ids, category_mask, indices_matrix):
    """
    Extracts categorized unique IDs and their corresponding indices 
    for the case of two arrays.

    Parameters:
    - unique_ids (np.ndarray): Sorted array of all unique IDs.
    - category_mask (np.ndarray): Boolean mask indicating presence of each ID in each array.
    - indices_matrix (np.ndarray): 2D object array with indices for each array.

    Returns:
    - only_in_1, only_in_2, in_both (np.ndarray): Unique IDs in each category.
    - indices_1_only, indices_2_only, indices_1_both, indices_2_both (list of np.ndarray): Corresponding indices in each array.

    Example:
    >>> ARRAY_1 = np.array([1, 2, 3, 4, 5])
    >>> ARRAY_2 = np.array([4, 5, 6, 7])
    >>> ID_ARRAYS = [ARRAY_1, ARRAY_2]
    >>> unique_ids, presence_matrix, category_mask = categorize_ids(ID_ARRAYS)
    >>> indices_matrix = get_all_indices(ID_ARRAYS, unique_ids)
    >>> only_in_1, only_in_2, in_both, indices_1_only, indices_2_only, indices_1_both, indices_2_both = postprocess_two_arrays(unique_ids, category_mask, indices_matrix)
    >>> only_in_1
    array([1, 2, 3])
    >>> only_in_2
    array([6, 7])
    >>> in_both
    array([4, 5])
    >>> indices_1_both  # Indices where 4 and 5 appear in ARRAY_1
    [array([3]), array([4])]
    >>> indices_2_both  # Indices where 4 and 5 appear in ARRAY_2
    [array([0]), array([1])]
    """

    logging.info("├── 🚀 Starting post-processing of unique IDs across two arrays...")

    # Validate input dimensions
    if category_mask.shape[0] != 2:
        logging.error("|    ├── ❌ ERROR: This function is only valid for exactly two arrays.")
        raise ValueError("category_mask must have exactly two rows (one for each array).")

    logging.debug(f"|    ├── 📌 Processing {len(unique_ids)} unique IDs.")

    # Extract presence masks for each array
    mask_1, mask_2 = category_mask

    # Categorize unique IDs into three groups
    only_in_1 = unique_ids[mask_1 & ~mask_2]
    only_in_2 = unique_ids[mask_2 & ~mask_1]
    in_both = unique_ids[mask_1 & mask_2]

    # Retrieve corresponding indices for each category
    indices_1_only = indices_matrix[mask_1 & ~mask_2, 0]
    indices_2_only = indices_matrix[mask_2 & ~mask_1, 1]
    indices_1_both = indices_matrix[mask_1 & mask_2, 0]
    indices_2_both = indices_matrix[mask_1 & mask_2, 1]

    # loggin message
    total_unique_ids = len(only_in_1) + len(only_in_2) + len(in_both)
    only_in_1_pct = round(100 * len(only_in_1) / total_unique_ids, 2) if total_unique_ids > 0 else 0
    only_in_2_pct = round(100 * len(only_in_2) / total_unique_ids, 2) if total_unique_ids > 0 else 0
    in_both_pct = round(100 * len(in_both) / total_unique_ids, 2) if total_unique_ids > 0 else 0
    logging.info(f"|    ├── Processing complete: {len(only_in_1)} IDs only in Array 1 ({only_in_1_pct}%).")
    logging.info(f"|    ├── Processing complete: {len(only_in_2)} IDs only in Array 2 ({only_in_2_pct}%).")
    logging.info(f"|    ├── Processing complete: {len(in_both)} IDs in both arrays ({in_both_pct}%).")

    return only_in_1, only_in_2, in_both, indices_1_only, indices_2_only, indices_1_both, indices_2_both

def crossmatch_IDs_two_datasets(IDs1, IDs2):
    """
    Crossmatches IDs between two datasets (IDs1 and IDs2) and identifies which IDs
    are only in one set or in both. Also provides the indices in the original arrays.

    Parameters
    ----------
    IDs1 : np.ndarray
        Array of IDs from dataset 1 (e.g., JPAS).
    IDs2 : np.ndarray
        Array of IDs from dataset 2 (e.g., DESI).

    Returns
    -------
    IDs_only_1, IDs_only_2, IDs_both : np.ndarray
        Unique IDs exclusive to dataset 1, dataset 2, and those shared by both.
    idxs_only_1, idxs_only_2 : list of np.ndarray
        Indices in IDs1 and IDs2 corresponding to IDs_only_1 and IDs_only_2.
    idxs_both_1, idxs_both_2 : list of np.ndarray
        Indices in IDs1 and IDs2 corresponding to IDs_both.
    """
    logging.info("🔍 crossmatch_IDs_two_datasets()...")

    ID_ARRAYS = [IDs1, IDs2]

    # Step 1: Categorize unique IDs and build presence matrix
    unique_ids, presence_matrix, category_mask = categorize_ids(ID_ARRAYS)

    # Step 2: Retrieve indices efficiently
    indices_matrix = get_all_indices(ID_ARRAYS, unique_ids)

    # Step 3: Post-process to extract useful results for two arrays
    IDs_only_1, IDs_only_2, IDs_both, idxs_only_1, idxs_only_2, idxs_both_1, idxs_both_2 = (
        crossmatch_two_arrays(unique_ids, category_mask, indices_matrix)
    )

    assert len(idxs_both_1) == len(idxs_both_2), "❌ The number of matched targets in IDs1 and IDs2 should be the same."

    logging.info("✅ Finished crossmatch_IDs_two_datasets()")

    return IDs_only_1, IDs_only_2, IDs_both, idxs_only_1, idxs_only_2, idxs_both_1, idxs_both_2