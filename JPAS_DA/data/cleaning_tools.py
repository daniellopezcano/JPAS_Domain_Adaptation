import numpy as np
import logging
import gc

def mask_out_unreliable_mock_data_columns(JPAS_obs, JPAS_err, DESI_mean, DESI_err, mask_indices):
    """
    Masks specified indices (e.g., [0, -2]) in DESI data that are known to be unreliable.
    
    Parameters:
    - JPAS_obs (np.ndarray): JPAS observation data.
    - JPAS_err (np.ndarray): JPAS error data.
    - DESI_mean (np.ndarray): DESI simulated means.
    - DESI_err (np.ndarray): DESI simulated errors.
    - mask_indices (list): Indices to be removed from the datasets.

    Returns:
    - JPAS_obs (np.ndarray): Updated JPAS observations.
    - JPAS_err (np.ndarray): Updated JPAS errors.
    - DESI_mean (np.ndarray): Updated DESI means.
    - DESI_err (np.ndarray): Updated DESI errors.
    """

    logging.info(f"├── Masking out indices {mask_indices} (unreliable in DESI).")

    mask = np.ones(JPAS_obs.shape[-1], dtype=bool)
    mask[mask_indices] = False  # Set the unwanted indices to False

    # Apply the mask to each dataset
    JPAS_obs = JPAS_obs[:, mask]
    JPAS_err = JPAS_err[:, mask]
    DESI_mean = DESI_mean[:, mask]
    DESI_err = DESI_err[:, mask]

    logging.info(f"│   ├── Updated JPAS obs/err shape: {JPAS_obs.shape}")
    logging.info(f"│   ├── Updated DESI mean/err shape: {DESI_mean.shape}")

    return JPAS_obs, JPAS_err, DESI_mean, DESI_err

def mask_magic_numbers_99(JPAS_obs, DESI_mean):
    """
    Detects and replaces "magic numbers" (99, -99) in the datasets.
    
    Parameters:
    - JPAS_obs (np.ndarray): JPAS observation data.
    - DESI_mean (np.ndarray): DESI simulated means.

    Returns:
    - JPAS_obs (np.ndarray): Updated observations.
    - DESI_mean (np.ndarray): Updated simulated means.
    - masks (dict): Dictionary containing masks for 99 and -99 values.
    """

    logging.info("├── Checking for magic numbers (99 and -99) in datasets.")

    # Create masks
    JPAS_mask_99 = (JPAS_obs == 99).astype(bool)
    JPAS_mask_neg99 = (JPAS_obs == -99).astype(bool)
    DESI_mask_99 = (DESI_mean == 99).astype(bool)
    DESI_mask_neg99 = (DESI_mean == -99).astype(bool)

    logging.info(f"│   ├── # objects containing some -99 entry in JPAS: {np.sum(np.sum(JPAS_mask_neg99, axis=1) != 0)}" + "(" + str(np.round(np.sum(np.sum(JPAS_mask_neg99, axis=1) != 0) / JPAS_obs.shape[0] * 100, 2)) + "%)")
    logging.info(f"│   ├── # objects containing some 99 entry in JPAS: {np.sum(np.sum(JPAS_mask_99, axis=1) != 0)}" + "(" + str(np.round(np.sum(np.sum(JPAS_mask_99, axis=1) != 0) / JPAS_obs.shape[0] * 100, 2)) + "%)")
    logging.info(f"│   ├── # objects containing some -99 entry in DESI: {np.sum(np.sum(DESI_mask_neg99, axis=1) != 0)}" + "(" + str(np.round(np.sum(np.sum(DESI_mask_neg99, axis=1) != 0) / DESI_mean.shape[0] * 100, 2)) + "%)")
    logging.info(f"│   ├── # objects containing some 99 entry in DESI: {np.sum(np.sum(DESI_mask_99, axis=1) != 0)}" + "(" + str(np.round(np.sum(np.sum(DESI_mask_99, axis=1) != 0) / DESI_mean.shape[0] * 100, 2)) + "%)")

    # Replace magic numbers with 0
    JPAS_obs[JPAS_mask_99] = 0
    JPAS_obs[JPAS_mask_neg99] = 0
    DESI_mean[DESI_mask_99] = 0
    DESI_mean[DESI_mask_neg99] = 0

    masks = {
        "JPAS_mask_99": JPAS_mask_99,
        "JPAS_mask_neg99": JPAS_mask_neg99,
        "DESI_mask_99": DESI_mask_99,
        "DESI_mask_neg99": DESI_mask_neg99
    }
    
    return JPAS_obs, DESI_mean, masks

def take_abs_value_of_negative_errors(JPAS_err, DESI_err):
    """
    Detects negative errors and replaces with the absolute value in the datasets.
    
    Parameters:
    - JPAS_err (np.ndarray): JPAS error data.
    - DESI_err (np.ndarray): DESI simulated errors.

    Returns:
    - JPAS_err (np.ndarray): Updated errors.
    - DESI_err (np.ndarray): Updated simulated errors.
    - masks (dict): Dictionary containing masks for negative errors.
    """

    logging.info("├── Checking for negative errors in datasets.")

    # Create masks for negative errors
    JPAS_mask_neg_errors = (JPAS_err < 0).astype(bool)
    DESI_mask_neg_errors = (DESI_err < 0).astype(bool)
    logging.info(f"│   ├── # objects containing some negative error entry in JPAS: {np.sum(np.sum(JPAS_mask_neg_errors, axis=1) != 0)}" + "(" + str(np.round(np.sum(np.sum(JPAS_mask_neg_errors, axis=1) != 0) / JPAS_err.shape[0] * 100, 2)) + "%)")
    logging.info(f"│   ├── # objects containing some negative error entry in DESI: {np.sum(np.sum(DESI_mask_neg_errors, axis=1) != 0)}" + "(" + str(np.round(np.sum(np.sum(DESI_mask_neg_errors, axis=1) != 0) / DESI_err.shape[0] * 100, 2)) + "%)")

    # Replace negative errors with the absolute value
    JPAS_err = np.abs(JPAS_err)
    DESI_err = np.abs(DESI_err)

    masks = {
        "JPAS_mask_neg_errors": JPAS_mask_neg_errors,
        "DESI_mask_neg_errors": DESI_mask_neg_errors
    }

    return JPAS_err, DESI_err, masks

def filter_preserve_type(data, valid_indices):
    """
    Filters dictionary entries while preserving their original types.
    
    Supported types:
    - NumPy arrays (preserves dtype)
    - Lists of strings (efficiently filtered via NumPy conversion)
    
    Logs a warning for unsupported types.
    """
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # Preserve dtype and apply filtering
            data[key] = value[valid_indices].astype(value.dtype)

        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            # Convert list to numpy array, filter, and convert back to list
            data[key] = np.array(value, dtype=object)[valid_indices].tolist()

        else:
            # Log a warning for unexpected types
            logging.warning(
                f"Key '{key}' has an unsupported type ({type(value)}). "
                "This key was not modified and remains unchanged."
            )
    return data

def remove_invalid_NaN_rows(JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA, magic_numbers=[-99, 99]):
    """
    Removes rows from JPAS and DESI datasets that contain only NaNs or magic numbers.
    Also updates corresponding dictionary entries in DATA.
    
    Parameters:
    - DATA (dict): Dictionary containing JPAS and DESI data.
    - magic_numbers (list): List of numbers considered as invalid.
    
    Returns:
    - DATA (dict): Updated dictionary with invalid rows removed.
    """
    logging.info("├── remove_invalid_NaN_rows()")
    
    # Identify invalid rows
    JPAS_obs_invalid_rows = np.where(np.all(np.isnan(JPAS_obs) | np.isin(JPAS_obs, magic_numbers), axis=1))[0]
    DESI_mean_invalid_rows = np.where(np.all(np.isnan(DESI_mean) | np.isin(DESI_mean, magic_numbers), axis=1))[0]

    logging.info(f"│   ├── # objects filled with NaNs in JPAS: {len(JPAS_obs_invalid_rows)}" + "(" + str(round(len(JPAS_obs_invalid_rows)/JPAS_obs.shape[0] * 100, 2)) + "%)")
    logging.info(f"│   ├── # objects filled with NaNs in DESI: {len(DESI_mean_invalid_rows)}" + "(" + str(round(len(DESI_mean_invalid_rows)/DESI_mean.shape[0] * 100, 2)) + "%)")

    # Get valid rows
    valid_JPAS_indices = np.setdiff1d(np.arange(JPAS_obs.shape[0]), JPAS_obs_invalid_rows)
    valid_DESI_indices = np.setdiff1d(np.arange(DESI_mean.shape[0]), DESI_mean_invalid_rows)

    # Filter observation entries for JPAS and DESI
    JPAS_obs = JPAS_obs[valid_JPAS_indices]
    JPAS_err = JPAS_err[valid_JPAS_indices]
    DESI_mean = DESI_mean[valid_DESI_indices]
    DESI_err = DESI_err[valid_DESI_indices]
    # Filter dictionary entries from pd dictionaries
    DATA_pd = {}
    DATA_pd["JPAS"] = filter_preserve_type(DATA["JPAS"]["all_pd"], valid_JPAS_indices)
    DATA_pd["DESI"] = filter_preserve_type(DATA["DESI"]["all_pd"], valid_DESI_indices)

    # Assertions to verify correctness
    assert not np.isnan(JPAS_obs).any(), "NaNs found in JPAS_obs after filtering!"
    assert not np.isnan(JPAS_err).any(), "NaNs found in JPAS_err after filtering!"
    assert not np.isnan(DESI_mean).any(), "NaNs found in DESI_mean after filtering!"
    assert not np.isnan(DESI_err).any(), "NaNs found in DESI_err after filtering!"

    return JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd

def apply_additional_filters(JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd, i_band_sn_threshold=20):
    """
    Applies additional filters based on the i-band signal-to-noise ratio (S/N).
    Removes rows from JPAS and DESI datasets where the i-band S/N is below the given threshold.
    
    Parameters:
    - JPAS_obs (np.ndarray): JPAS observation data.
    - JPAS_err (np.ndarray): JPAS error data.
    - DESI_mean (np.ndarray): DESI simulated means.
    - DESI_err (np.ndarray): DESI simulated errors.
    - DATA_pd (dict): Dictionary containing JPAS and DESI CSV data.
    - sn_threshold (float): S/N threshold. Rows with i-band S/N below this value will be removed.
    
    Returns:
    - JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd: The filtered arrays and updated dictionary.
    """
    logging.info("├── apply_additional_filters()")
    
    # Compute i-band Signal-to-Noise ratio (S/N)
    JPAS_i_band_SN = JPAS_obs[:, -1] / np.abs(JPAS_err[:, -1])  # Assuming i-band is index -1
    DESI_i_band_SN = DESI_mean[:, -1] / np.abs(DESI_err[:, -1])   # Assuming i-band is index -1

    # Create masks based on the S/N threshold
    valid_JPAS_indices = np.where(JPAS_i_band_SN >= i_band_sn_threshold)[0]
    valid_DESI_indices = np.where(DESI_i_band_SN >= i_band_sn_threshold)[0]
    
    JPAS_valid_pct = round(len(valid_JPAS_indices) / JPAS_obs.shape[0] * 100, 2) if JPAS_obs.shape[0] > 0 else 0
    DESI_valid_pct = round(len(valid_DESI_indices) / DESI_mean.shape[0] * 100, 2) if DESI_mean.shape[0] > 0 else 0
    logging.info(f"│   ├── JPAS: {len(valid_JPAS_indices)} valid rows (S/N ≥ {i_band_sn_threshold}) ({JPAS_valid_pct}%)")
    logging.info(f"│   ├── DESI: {len(valid_DESI_indices)} valid rows (S/N ≥ {i_band_sn_threshold}) ({DESI_valid_pct}%)")
    
    # Apply the filtering to the arrays
    JPAS_obs = JPAS_obs[valid_JPAS_indices]
    JPAS_err = JPAS_err[valid_JPAS_indices]
    DESI_mean = DESI_mean[valid_DESI_indices]
    DESI_err = DESI_err[valid_DESI_indices]
    
    # Filter dictionary entries (CSV data) using the same valid indices
    DATA_pd["JPAS"] = filter_preserve_type(DATA_pd["JPAS"], valid_JPAS_indices)
    DATA_pd["DESI"] = filter_preserve_type(DATA_pd["DESI"], valid_DESI_indices)
    
    logging.info("│   ├── Additional filters applied successfully.")

    return JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd

def encode_strings_to_integers(
    string_list: list,
    reference_mapping: dict = None
):
    """
    Encodes strings to integers using a consistent mapping.

    If `reference_mapping` is provided, it is used for encoding.
    If not, a new mapping is created from the input list.

    Parameters:
    - string_list (list of str): Input list to encode.
    - reference_mapping (dict, optional): Predefined mapping.

    Returns:
    - int_array (np.ndarray): Encoded integers.
    - mapping (dict): Used string-to-int mapping.
    """
    logging.info("├── 🔑 Starting encoding process for string list...")

    if not isinstance(string_list, list) or not all(isinstance(item, str) for item in string_list):
        logging.error("|    ├── ❌ Input must be a list of strings.")
        raise ValueError("Input must be a list of strings.")

    string_array = np.array(string_list, dtype=str)

    if reference_mapping is None:
        unique_strings = np.unique(string_array)
        mapping = {string: idx for idx, string in enumerate(sorted(unique_strings))}
        encoded = np.array([mapping[s] for s in string_array])
        logging.info(f"|    ├── 📌 New Mapping Created: {mapping}")
    else:
        mapping = reference_mapping
        encoded = np.array([mapping.get(s, -1) for s in string_array])
        if -1 in encoded:
            missing = set(string_array) - set(mapping.keys())
            logging.warning(f"|    ├── ⚠️ Unmapped categories found: {missing}")
        logging.info(f"|    ├── 📌 Used Provided Mapping: {mapping}")

    logging.info(f"├── Encoding complete ({len(mapping)} categories).")
    return encoded, mapping

def clean_and_mask_data(
        DATA, apply_masks=['unreliable', 'apply_additional_filters', 'magic_numbers', 'negative_errors', 'nan_values'],
        mask_indices=[0, -2], magic_numbers=[99, -99], i_band_sn_threshold=20, z_lim_QSO_cut=None
    ):
    """
    Main function to clean and mask observational and simulated datasets.

    Parameters:
    - DATA
    - mask_indices (list): Indices to be removed from the datasets.
    - apply_masks (list): List of masks to apply

    Returns:
    - DAtA_clean (dict): Dictionary containing cleaned and masked datasets.
    """

    logging.info("🧽 Cleaning and masking data...")
    
    JPAS_obs = DATA['JPAS']["all_observations"]
    JPAS_err = DATA['JPAS']["all_errors"]
    DESI_mean = DATA['DESI']["all_np"][..., 0]
    DESI_err = DATA['DESI']["all_np"][..., 2]

    # Step 1: Handle NaN values
    if "nan_values" in apply_masks:
        JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd = remove_invalid_NaN_rows(JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA, magic_numbers=magic_numbers)
    else:
        DATA_pd = DATA
    del DATA
    gc.collect()
    logging.info("├── 🧹 Deleted cleaned DATA_clean dictionary to free memory.")

    # Step 2: Apply additional filters
    if "apply_additional_filters" in apply_masks:
        JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd = apply_additional_filters(
            JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd, i_band_sn_threshold=i_band_sn_threshold
        )

    masks = {}  # Dictionary to store applied masks

    # Step 2: Remove unreliable entries
    if "unreliable" in apply_masks:
        JPAS_obs, JPAS_err, DESI_mean, DESI_err = mask_out_unreliable_mock_data_columns(JPAS_obs, JPAS_err, DESI_mean, DESI_err, mask_indices)
    
    # Step 3: Handle magic numbers (99, -99)
    if "magic_numbers" in apply_masks:
        JPAS_obs, DESI_mean, magic_masks = mask_magic_numbers_99(JPAS_obs, DESI_mean)
        masks.update(magic_masks)
    
    # Step 4: Handle negative errors
    if "negative_errors" in apply_masks:
        JPAS_err, DESI_err, error_masks = take_abs_value_of_negative_errors(JPAS_err, DESI_err)
        masks.update(error_masks)
    
    # Step 5: Split between High and Low redshift quasars
    if z_lim_QSO_cut != None:
        logging.info("├── Splitting between High and Low z QSOs")
        for survey in list(DATA_pd.keys()):
            for ii in range(len(DATA_pd[survey]['SPECTYPE'])):
                if DATA_pd[survey]['SPECTYPE'][ii] == "QSO":
                    if DATA_pd[survey]['REDSHIFT'][ii] < z_lim_QSO_cut:
                        DATA_pd[survey]['SPECTYPE'][ii] = "QSO_low"
                    else:
                        DATA_pd[survey]['SPECTYPE'][ii] = "QSO_high"

    # Step 6: Sample DESI with the corresponding simulated variance
    DESI_obs = np.random.normal(loc=DESI_mean, scale=DESI_err)

    # Step 7: Encode strings to integers
    all_morphs = list(DATA_pd["DESI"]['SPECTYPE']) + list(DATA_pd["JPAS"]['SPECTYPE'])
    _, shared_mapping = encode_strings_to_integers(all_morphs)
    DESI_SPECTYPE_int, _ = encode_strings_to_integers(DATA_pd["DESI"]['SPECTYPE'], reference_mapping=shared_mapping)
    JPAS_SPECTYPE_int, _ = encode_strings_to_integers(DATA_pd["JPAS"]['SPECTYPE'], reference_mapping=shared_mapping)

    all_morphs = list(DATA_pd["DESI"]['MORPHTYPE']) + list(DATA_pd["JPAS"]['MORPHTYPE'])
    _, shared_mapping = encode_strings_to_integers(all_morphs)
    DESI_MORPHTYPE_int, _ = encode_strings_to_integers(DATA_pd["DESI"]['MORPHTYPE'], reference_mapping=shared_mapping)
    JPAS_MORPHTYPE_int, _ = encode_strings_to_integers(DATA_pd["JPAS"]['MORPHTYPE'], reference_mapping=shared_mapping)

    # Step 7: Create clean dictionary
    DATA_clean = {}
    for survey in list(DATA_pd.keys()):
        DATA_clean[survey] = {}
        for key in list(DATA_pd[survey].keys()):
            DATA_clean[survey][key] = DATA_pd[survey][key]
        for key in list(masks.keys()):
            if survey in key:
                new_key = key[len(survey) + 1:]
                DATA_clean[survey][new_key] = masks[key]

    survey = "DESI"
    DATA_clean[survey]["MEAN"] = DESI_mean
    DATA_clean[survey]["OBS"] = DESI_obs
    DATA_clean[survey]["ERR"] = DESI_err
    DATA_clean[survey]["SPECTYPE_int"] = DESI_SPECTYPE_int
    DATA_clean[survey]["MORPHTYPE_int"] = DESI_MORPHTYPE_int

    survey = "JPAS"
    DATA_clean[survey]["OBS"] = JPAS_obs
    DATA_clean[survey]["ERR"] = JPAS_err
    DATA_clean[survey]["SPECTYPE_int"] = JPAS_SPECTYPE_int
    DATA_clean[survey]["MORPHTYPE_int"] = JPAS_MORPHTYPE_int

    logging.info(f"✅ Finished clean_and_mask_data()")

    return DATA_clean