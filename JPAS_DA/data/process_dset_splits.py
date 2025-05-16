import numpy as np
import logging
from typing import List, Dict, Tuple, Any

def split_indexes_shuffled(NN, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits shuffled indices into three parts for train, validation, and test sets.

    Parameters:
    - NN (int): Total number of indices to split.
    - train_ratio (float): Ratio of indices to assign to the training set.
    - val_ratio (float): Ratio of indices to assign to the validation set.
    - test_ratio (float): Ratio of indices to assign to the test set.
    - seed (int): Seed for the random number generator.

    Returns:
    - train_idx, val_idx, test_idx (np.ndarray): Shuffled indices for each split.

    Example:
    >>> train_idx, val_idx, test_idx = split_indexes_shuffled_with_indices(100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    >>> train_idx.shape
    (70,)
    >>> val_idx.shape
    (15,)
    >>> test_idx.shape
    (15,)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        logging.error("    |    ├── ❌ ERROR: The sum of train, validation, and test ratios must be exactly 1.0.")
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    np.random.seed(seed)  # Set random seed for reproducibility
    shuffled_indices = np.random.permutation(NN)  # Shuffle indices instead of IDs directly

    # Compute split points
    train_end = int(train_ratio * NN)
    val_end = train_end + int(val_ratio * NN)

    # Split indices
    train_idx, val_idx, test_idx = np.split(shuffled_indices, [train_end, val_end])

    return train_idx, val_idx, test_idx

def split_LoA(LoA, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits a List of Anything (LoA) into training, validation, and testing subsets 
    based on provided ratios. The order is randomized with a given seed.

    Parameters:
    - LoA (list): List of elements to be split.
    - train_ratio (float): Proportion of elements for training.
    - val_ratio (float): Proportion of elements for validation.
    - test_ratio (float): Proportion of elements for testing.
    - seed (int): Random seed for reproducible splits.

    Returns:
    - LoA_split (dict): Dictionary with keys "train", "val", and "test" mapping to
      their respective subsets of the input list.

    """
    logging.info("├── ✂️ Splitting list of arrays (LoA) into train/val/test subsets...")

    # Get shuffled indices for train/val/test
    tmp_idxs_split = {}
    tmp_idxs_split["train"], tmp_idxs_split["val"], tmp_idxs_split["test"] = split_indexes_shuffled(
        len(LoA), train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    # Create subset dictionary based on indices
    LoA_split = {}
    for dset_type_key in tmp_idxs_split:
        LoA_split[dset_type_key] = [LoA[i] for i in tmp_idxs_split[dset_type_key]]
        logging.debug(f"    |    ├── {dset_type_key}: {len(LoA_split[dset_type_key])} entries")

    logging.info("├── Finished splitting.")

    return LoA_split

def extract_subset_info(LoA, data_dict, keys):
    """
    Maps grouped indices to continuous subset indices, and extracts associated values from a dictionary.

    Parameters:
    - LoA: list of arrays with original indices
    - data_dict: dictionary containing arrays or lists from which values will be extracted
    - keys: list of keys from data_dict to extract values from

    Returns:
    - dict with:
        - 'LoA': list of same shape as LoA with new local indices
        - One entry per requested key with the extracted values
    """
    if not all(k in data_dict for k in keys):
        missing = [k for k in keys if k not in data_dict]
        raise KeyError(f"The following keys are missing in the data_dict: {missing}")

    subset_LoA = []
    extracted = {key: [] for key in keys}
    counter = 0

    for group in LoA:
        local_indices = []
        for idx in group:
            local_indices.append(counter)
            for key in keys:
                extracted[key].append(data_dict[key][idx])
            counter += 1
        subset_LoA.append(local_indices)

    # Convert lists to numpy arrays
    extracted = {key: np.array(vals) for key, vals in extracted.items()}
    extracted["LoA"] = subset_LoA

    return extracted


def extract_and_combine_DESI_data(
    LoA_split_only_DESI: List[List[int]],
    LoA_split_both_DESI: List[List[int]],
    DATA: Dict[str, Any],
    keys_xx: List[str],
    keys_yy: List[str]
) -> Tuple[List[List[int]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extracts and concatenates DESI-only and matched DESI data subsets from LoA structures.

    Returns:
    - LoA: Combined list of index lists.
    - xx: Dictionary of concatenated feature arrays.
    - yy: Dictionary of concatenated label arrays.
    """
    logging.info("|    ├── 🔧 extract_and_combine_DESI_data()")

    # Step 1: Extract subset data
    logging.info("|    ├── Extracting features and labels from DESI-only subset...")
    sub_only = extract_subset_info(LoA_split_only_DESI, DATA, keys=keys_xx + keys_yy)

    logging.info("|    ├── Extracting features and labels from DESI-matched subset...")
    sub_both = extract_subset_info(LoA_split_both_DESI, DATA, keys=keys_xx + keys_yy)

    # Step 2: Offset LoA indices from 'both' to avoid overlaps
    shift = np.max(np.concatenate(sub_only['LoA'])) + 1
    sub_both['LoA'] = [[idx + shift for idx in group] for group in sub_both['LoA']]
    LoA_combined = sub_only['LoA'] + sub_both['LoA']
    logging.info(f"|    ├── Applied index shift of {shift} to matched DESI group to ensure uniqueness")

    # Step 3: Concatenate feature arrays
    xx = {}
    for key in keys_xx:
        xx[key] = np.concatenate([sub_only[key], sub_both[key]], axis=0)
        logging.debug(f"|    ├── ✔ Concatenated feature '{key}' with shape {xx[key].shape}")

    # Step 4: Concatenate label arrays
    yy = {}
    for key in keys_yy:
        yy[key] = np.concatenate([sub_only[key], sub_both[key]], axis=0)
        logging.debug(f"|    ├── ✔ Concatenated label '{key}' with shape {yy[key].shape}")

    logging.info("|    ├── Finished extract_and_combine_DESI_data()")
    return LoA_combined, xx, yy


def extract_data_matched(
    LoA: List[List[int]],
    DATA: Dict[str, Any],
    keys_xx: List[str],
    keys_yy: List[str]
) -> Tuple[List[List[int]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extracts data from a matched set of indices using LoA.

    Returns:
    - LoA: Same as input (repackaged).
    - xx: Dictionary of feature arrays.
    - yy: Dictionary of label arrays.
    """
    logging.info("|    ├── 🔧 extract_data_matched()")

    # Step 1: Extract subset info
    logging.info("|    ├── Extracting features and labels from matched dataset...")
    sub = extract_subset_info(LoA, DATA, keys=keys_xx + keys_yy)
    LoA = sub['LoA']

    # Step 2: Organize feature arrays
    xx = {}
    for key in keys_xx:
        xx[key] = sub[key]
        logging.debug(f"|    ├── ✔ Retrieved feature '{key}' with shape {xx[key].shape}")

    # Step 3: Organize label arrays
    yy = {}
    for key in keys_yy:
        yy[key] = sub[key]
        logging.debug(f"|    ├── ✔ Retrieved label '{key}' with shape {yy[key].shape}")

    logging.info("|    ├── Finished extract_data_matched()")
    return LoA, xx, yy
