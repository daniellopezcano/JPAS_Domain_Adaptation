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


def extract_from_block_by_LoA(
    block: Dict[str, Any],
    LoA: List[List[int]],
    keys_xx: List[str],
    keys_yy: List[str],
) -> Tuple[List[List[int]], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extract arrays from a single dataset block using one LoA (list of index groups).

    Requirements:
      - Each key in `keys_xx` and `keys_yy` must exist either:
          • as a top-level key in `block` (e.g., 'all_observations_normalized'), or
          • as a key in `block['all_pd']`.
      - LoA is a list of lists of row indices. Groups are concatenated in order.

    Returns:
      - LoA_local: local reindexed groups matching the concatenated outputs
      - xx: dict of concatenated feature arrays
      - yy: dict of concatenated label arrays
    """
    logging.info("|    ├── 🔧 extract_from_block_by_LoA()")

    # --- helper: get array for a key from block (top level or all_pd)
    def _fetch_array(b: Dict[str, Any], key: str) -> np.ndarray:
        if key in b:
            return np.asarray(b[key])
        if "all_pd" in b and isinstance(b["all_pd"], dict) and key in b["all_pd"]:
            return np.asarray(b["all_pd"][key])
        raise KeyError(f"Requested key '{key}' not found in block (top level or all_pd).")

    # --- helper: slice by groups and concatenate (supports 1D/2D arrays)
    def _slice_by_groups(arr: np.ndarray, groups: List[List[int]]) -> np.ndarray:
        parts = []
        for g in groups:
            idx = np.asarray(g, dtype=int)
            parts.append(arr[idx] if arr.ndim == 1 else arr[idx, ...])
        if not parts:
            return np.empty((0,) + arr.shape[1:], dtype=arr.dtype) if arr.ndim > 1 else np.empty((0,), dtype=arr.dtype)
        return np.concatenate(parts, axis=0)

    # --- helper: local reindex [0..total-1] per group
    def _reindex_locally(groups: List[List[int]]) -> List[List[int]]:
        reindexed, offset = [], 0
        for g in groups:
            L = len(g)
            reindexed.append(list(range(offset, offset + L)))
            offset += L
        return reindexed

    want_keys = list(dict.fromkeys(keys_xx + keys_yy))  # unique, keep order

    # pick first available key as reference to validate dims later
    ref_key = None
    for k in want_keys:
        try:
            _ = _fetch_array(block, k)
            ref_key = k
            break
        except KeyError:
            continue
    if ref_key is None:
        raise KeyError("[extract_from_block_by_LoA] None of the requested keys exist in this block.")

    logging.info(f"|    ├── Using reference key: '{ref_key}'")

    arrays: Dict[str, np.ndarray] = {}
    for k in want_keys:
        arr = _fetch_array(block, k)
        arrays[k] = _slice_by_groups(arr, LoA)
        logging.debug(f"|    │   • key='{k}', out shape={arrays[k].shape}")

    LoA_local = _reindex_locally(LoA)

    xx, yy = {}, {}
    for k in keys_xx:
        if k not in arrays:
            raise KeyError(f"[extract_from_block_by_LoA] Requested feature key '{k}' was not extracted.")
        xx[k] = arrays[k]
        logging.debug(f"|    ├── ✔ Feature '{k}' → {xx[k].shape}")

    for k in keys_yy:
        if k not in arrays:
            raise KeyError(f"[extract_from_block_by_LoA] Requested label key '{k}' was not extracted.")
        yy[k] = arrays[k]
        logging.debug(f"|    ├── ✔ Label '{k}' → {yy[k].shape}")

    logging.info("|    ├── Finished extract_from_block_by_LoA()")
    return LoA_local, xx, yy