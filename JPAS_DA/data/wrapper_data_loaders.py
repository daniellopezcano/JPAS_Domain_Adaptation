from typing import List, Dict, Tuple, Any, Optional
import logging

from JPAS_DA.data import loading_tools
from JPAS_DA.data import cleaning_tools
from JPAS_DA.data import crossmatch_tools
from JPAS_DA.data import process_dset_splits
from JPAS_DA.data import data_loaders

def wrapper_data_loaders(
    root_path: str,
    load_JPAS_data: List[Dict[str, Any]],
    load_DESI_data: List[Dict[str, Any]],
    random_seed_load: int,
    apply_masks: List[str],
    mask_indices: List[int],
    magic_numbers: List[float],
    i_band_sn_threshold: float,
    z_lim_QSO_cut: float,
    train_ratio_both: float,
    val_ratio_both: float,
    test_ratio_both: float,
    random_seed_split_both: int,
    train_ratio_only_DESI: float,
    val_ratio_only_DESI: float,
    test_ratio_only_DESI: float,
    random_seed_split_only_DESI: int,
    define_dataset_loaders_keys: Optional[List[str]],
    keys_xx: List[str],
    keys_yy: List[str],
    normalization_source_key: Optional[str],
    normalize: bool,
    provided_normalization: Optional[Tuple[List[float], List[float]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    High-level wrapper that orchestrates the full data preparation pipeline for domain adaptation 
    between JPAS and DESI datasets. It performs the following steps:

    1. Loads raw data files for JPAS and DESI from disk, applying optional downsampling.
    2. Applies multiple user-defined cleaning and masking filters to the photometry.
    3. Crossmatches sources between surveys using `TARGETID` to identify shared and exclusive targets.
    4. Splits the data into training, validation, and test sets based on user-defined proportions.
    5. Constructs standardized `DataLoader` objects for each subset, reusing normalization statistics 
       from the training set to ensure consistency.

    Parameters:
    ----------
    root_path : str
        Path to the base folder where data files are stored.
    load_JPAS_data : list of dict
        Each dict must contain 'name', 'npy', 'csv', and optionally 'sample_percentage'.
    load_DESI_data : list of dict
        Each dict must contain 'name', 'npy', 'csv', and 'sample_percentage'.
    random_seed_load : int
        Seed for sampling during dataset loading.
    apply_masks : list of str
        Masking rules to apply (e.g. 'magic_numbers', 'unreliable', etc.).
    mask_indices : list of int
        Indices in the wavelength/filter space where to apply masks.
    magic_numbers : list of float
        Values to treat as invalid (e.g. 99, -99).
    i_band_sn_threshold : float
        Minimum signal-to-noise required in the i-band (or relevant band).
    z_lim_QSO_cut : float
        Upper redshift limit for QSO cut in cleaning.
    train_ratio_both / val_ratio_both / test_ratio_both : float
        Splitting ratios for sources matched in both JPAS and DESI.
    random_seed_split_both : int
        Seed for splitting matched samples.
    train_ratio_only_DESI / val_ratio_only_DESI / test_ratio_only_DESI : float
        Splitting ratios for DESI-only sources.
    random_seed_split_only_DESI : int
        Seed for splitting DESI-only sources.
    define_dataset_loaders_keys : list of str
        List of keys to include in the DataLoaders (e.g. ['DESI_combined', 'DESI_only', 'DESI_matched', 'JPAS_matched']).
    keys_xx : list of str
        Feature keys to include (e.g. ['OBS', 'ERR', 'MORPHTYPE_int']).
    keys_yy : list of str
        Label keys to include (e.g. ['SPECTYPE_int']).
    normalization_source_key : str
        Key to use for normalization (e.g. 'DESI_combined').
    normalize : bool
        Whether to normalize input features. Normalization statistics are taken from the training set.
    provided_normalization : tuple of list of float, optional
        Pre-computed mean and std values for normalization.

    Returns:
    -------
    dset_loaders : dict
        Dictionary with keys:
            - 'DESI_combined': DataLoaders from DESI-only + matched samples.
            - 'DESI_matched': DataLoaders from matched DESI samples only.
            - 'JPAS_matched': DataLoaders from matched JPAS samples only.
        Each entry contains 'train', 'val', 'test' subkeys.
    """

    logging.info("📦 Starting full data preparation pipeline...")

    # ───────────────────────────────────────────────────── #
    # 1. Load raw JPAS and DESI datasets
    # ───────────────────────────────────────────────────── #
    logging.info("\n\n1️⃣: Loading datasets from disk...")
    DATA = loading_tools.load_dsets(
        root_path=root_path,
        datasets_jpas=load_JPAS_data,
        datasets_desi=load_DESI_data,
        random_seed=random_seed_load
    )

    # ───────────────────────────────────────────────────── #
    # 2. Apply cleaning and masking procedures
    # ───────────────────────────────────────────────────── #
    logging.info("\n\n2️⃣: Cleaning and masking data...")
    DATA = cleaning_tools.clean_and_mask_data(
        DATA=DATA,
        apply_masks=apply_masks,
        mask_indices=mask_indices,
        magic_numbers=magic_numbers,
        i_band_sn_threshold=i_band_sn_threshold,
        z_lim_QSO_cut=z_lim_QSO_cut
    )

    # ───────────────────────────────────────────────────── #
    # 3. Crossmatch JPAS and DESI using TARGETID
    # ───────────────────────────────────────────────────── #
    logging.info("\n\n3️⃣: Crossmatching JPAS and DESI TARGETIDs...")
    Dict_LoA = {"both": {}, "only": {}}
    IDs_only_DESI, IDs_only_JPAS, IDs_both, \
    Dict_LoA["only"]["DESI"], Dict_LoA["only"]["JPAS"], \
    Dict_LoA["both"]["DESI"], Dict_LoA["both"]["JPAS"] = crossmatch_tools.crossmatch_IDs_two_datasets(
        DATA["DESI"]['TARGETID'], DATA["JPAS"]['TARGETID']
    )

    # ───────────────────────────────────────────────────── #
    # 4. Perform train/val/test splits
    # ───────────────────────────────────────────────────── #
    logging.info("\n\n4️⃣: Splitting data into train/val/test...")
    Dict_LoA_split = {"both": {}, "only": {}}

    Dict_LoA_split["both"]["JPAS"] = process_dset_splits.split_LoA(
        Dict_LoA["both"]["JPAS"], train_ratio_both, val_ratio_both, test_ratio_both, seed=random_seed_split_both
    )
    Dict_LoA_split["both"]["DESI"] = process_dset_splits.split_LoA(
        Dict_LoA["both"]["DESI"], train_ratio_both, val_ratio_both, test_ratio_both, seed=random_seed_split_both
    )
    Dict_LoA_split["only"]["DESI"] = process_dset_splits.split_LoA(
        Dict_LoA["only"]["DESI"], train_ratio_only_DESI, val_ratio_only_DESI, test_ratio_only_DESI, seed=random_seed_split_only_DESI
    )

    # ───────────────────────────────────────────────────── #
    # 5. Create DataLoaders for all training subsets
    # ───────────────────────────────────────────────────── #
    logging.info("\n\n5️⃣: Initializing DataLoader objects...")

    if define_dataset_loaders_keys is None:
        define_dataset_loaders_keys = ["DESI_combined", "DESI_only", "DESI_matched", "JPAS_matched"]

    # Ensure the requested dataset keys are valid
    valid = {"DESI_combined", "DESI_only", "DESI_matched", "JPAS_matched"}
    bad = set(define_dataset_loaders_keys) - valid
    if bad:
        raise ValueError(f"Unknown dataset keys: {bad}")

    # Ensure the requested normalization source is valid
    if normalization_source_key is not None and normalization_source_key not in define_dataset_loaders_keys:
        raise ValueError(f"Normalization source '{normalization_source_key}' was requested, "
                        f"but it is not among the selected loaders: {define_dataset_loaders_keys}")

    # Initialize empty loader dict
    dset_loaders = {key: {} for key in define_dataset_loaders_keys}

    # ─────────────────────────────────────── #
    # 5.1 First pass: compute normalization
    # ─────────────────────────────────────── #
    if provided_normalization is None and normalization_source_key is not None:
        logging.info(f"📏 Computing normalization from anchor '{normalization_source_key}' (train only)")
        key_dset = "train"
        if normalization_source_key == "DESI_combined":
            LoA, xx, yy = process_dset_splits.extract_and_combine_DESI_data(
                Dict_LoA_split["only"]["DESI"][key_dset], Dict_LoA_split["both"]["DESI"][key_dset], DATA["DESI"], keys_xx, keys_yy
            )
        elif normalization_source_key == "DESI_only":
            LoA, xx, yy = process_dset_splits.extract_data_using_LoA(
                Dict_LoA_split["only"]["DESI"][key_dset], DATA["DESI"], keys_xx, keys_yy
            )
        elif normalization_source_key == "DESI_matched":
            LoA, xx, yy = process_dset_splits.extract_data_using_LoA(
                Dict_LoA_split["both"]["DESI"][key_dset], DATA["DESI"], keys_xx, keys_yy
            )
        elif normalization_source_key == "JPAS_matched":
            LoA, xx, yy = process_dset_splits.extract_data_using_LoA(
                Dict_LoA_split["both"]["JPAS"][key_dset], DATA["JPAS"], keys_xx, keys_yy
            )

        anchor_loader = data_loaders.DataLoader(xx, yy, normalize=True)
        provided_normalization = (anchor_loader.means, anchor_loader.stds)

    # ─────────────────────────────────────── #
    # 5.2 Second pass: build all DataLoaders
    # ─────────────────────────────────────── #
    for key_dset in ["train", "val", "test"]:
        logging.info(f"⚙️ Preparing split: {key_dset}")
        for key_loader in define_dataset_loaders_keys:
            logging.info(f"├── {key_loader}")
            if key_loader == "DESI_combined":
                LoA, xx, yy = process_dset_splits.extract_and_combine_DESI_data(
                    Dict_LoA_split["only"]["DESI"][key_dset], Dict_LoA_split["both"]["DESI"][key_dset], DATA["DESI"], keys_xx, keys_yy
                )
            elif key_loader == "DESI_only":
                LoA, xx, yy = process_dset_splits.extract_data_using_LoA(
                    Dict_LoA_split["only"]["DESI"][key_dset], DATA["DESI"], keys_xx, keys_yy
                )
            elif key_loader == "DESI_matched":
                LoA, xx, yy = process_dset_splits.extract_data_using_LoA(
                    Dict_LoA_split["both"]["DESI"][key_dset], DATA["DESI"], keys_xx, keys_yy
                )
            elif key_loader == "JPAS_matched":
                LoA, xx, yy = process_dset_splits.extract_data_using_LoA(
                    Dict_LoA_split["both"]["JPAS"][key_dset], DATA["JPAS"], keys_xx, keys_yy
                )
            dset_loaders[key_loader][key_dset] = data_loaders.DataLoader(
                xx, yy, normalize=normalize, provided_normalization=provided_normalization
            )

    logging.info("✅ DataLoader preparation complete.")
    return dset_loaders
