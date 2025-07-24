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

    # Log warnings for NaN values
    jpas_nan_count = np.isnan(JPAS_obs).any(axis=1).sum()
    jpas_nan_perc = round(jpas_nan_count / JPAS_obs.shape[0] * 100, 2)
    logging.warning(f"│   ├── # objects with NaNs in JPAS_obs: {jpas_nan_count} ({jpas_nan_perc}%)")
    
    jpas_err_nan_count = np.isnan(JPAS_err).any(axis=1).sum()
    jpas_err_nan_perc = round(jpas_err_nan_count / JPAS_err.shape[0] * 100, 2)
    logging.warning(f"│   ├── # objects with NaNs in JPAS_err: {jpas_err_nan_count} ({jpas_err_nan_perc}%)")
    
    desi_nan_count = np.isnan(DESI_mean).any(axis=1).sum()
    desi_nan_perc = round(desi_nan_count / DESI_mean.shape[0] * 100, 2)
    logging.warning(f"│   ├── # objects with NaNs in DESI_mean: {desi_nan_count} ({desi_nan_perc}%)")
    
    desi_err_nan_count = np.isnan(DESI_err).any(axis=1).sum()
    desi_err_nan_perc = round(desi_err_nan_count / DESI_err.shape[0] * 100, 2)
    logging.warning(f"│   ├── # objects with NaNs in DESI_err: {desi_err_nan_count} ({desi_err_nan_perc}%)")

    return JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd

def apply_additional_filters(
    JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd,
    i_band_sn_threshold=20,
    magnitude_flux_key=None, magnitude_threshold=None
):
    """
    Applies filters based on the i-band S/N and optionally a magnitude threshold for a given DESI_FLUX_X key.

    Parameters:
    - JPAS_obs (np.ndarray): JPAS observation data.
    - JPAS_err (np.ndarray): JPAS error data.
    - DESI_mean (np.ndarray): DESI simulated means.
    - DESI_err (np.ndarray): DESI simulated errors.
    - DATA_pd (dict): Dictionary containing JPAS and DESI CSV data.
    - i_band_sn_threshold (float): Minimum i-band S/N to keep a sample.
    - magnitude_flux_key (str, optional): Flux key to compute magnitude ('DESI_FLUX_R', etc.).
    - magnitude_threshold (float, optional): Upper magnitude limit to keep samples.

    Returns:
    - Filtered JPAS_obs, JPAS_err, DESI_mean, DESI_err, and updated DATA_pd dictionary.
    """
    logging.info("├── apply_additional_filters()")

    # Step 1: Apply i-band S/N filtering (assume i-band is last column)
    logging.info(f"│   ├── Applying i-band S/N ≥ {i_band_sn_threshold}")
    sn_masks = {}
    sn_masks["JPAS"] = JPAS_obs[:, -1] / np.abs(JPAS_err[:, -1]) >= i_band_sn_threshold
    sn_masks["DESI"] = DESI_mean[:, -1] / np.abs(DESI_err[:, -1]) >= i_band_sn_threshold

    for key, mask in sn_masks.items():
        n_total = JPAS_obs.shape[0] if key == "JPAS" else DESI_mean.shape[0]
        logging.info(f"│   │   ├── {key}: {np.sum(mask)} valid rows ({100 * np.sum(mask) / n_total:.2f}%)")

    # Step 2: Apply optional magnitude cut for both JPAS and DESI
    mag_masks = { "JPAS": np.ones_like(sn_masks["JPAS"], dtype=bool), "DESI": np.ones_like(sn_masks["DESI"], dtype=bool) }

    if magnitude_flux_key is not None and magnitude_threshold is not None:
        logging.info(f"│   ├── Applying magnitude filter: {magnitude_flux_key} ≤ {magnitude_threshold}")

        for key in ["JPAS", "DESI"]:
            if magnitude_flux_key in DATA_pd[key]:
                flux = np.array(DATA_pd[key][magnitude_flux_key])
                with np.errstate(divide="ignore", invalid="ignore"):
                    mag = 22.5 - 2.5 * np.log10(flux)
                mask_valid = np.isfinite(mag) & (mag <= magnitude_threshold)
                mag_masks[key] = mask_valid
                n_valid = np.sum(mask_valid)
                n_total = flux.shape[0]
                logging.info(f"│   │   ├── {key}: {n_valid} valid rows after magnitude filter ({100 * n_valid / n_total:.2f}%)")
            else:
                logging.warning(f"│   │   ├── {key}: flux key '{magnitude_flux_key}' not found in DATA_pd[{key}] → skipping magnitude cut")

    # Step 3: Combine S/N and magnitude masks
    combined_masks = {
        "JPAS": sn_masks["JPAS"] & mag_masks["JPAS"],
        "DESI": sn_masks["DESI"] & mag_masks["DESI"]
    }

    # Step 4: Apply filters to arrays
    JPAS_obs = JPAS_obs[combined_masks["JPAS"]]
    JPAS_err = JPAS_err[combined_masks["JPAS"]]
    DESI_mean = DESI_mean[combined_masks["DESI"]]
    DESI_err = DESI_err[combined_masks["DESI"]]

    # Step 5: Filter pandas-like structures
    DATA_pd["JPAS"] = filter_preserve_type(DATA_pd["JPAS"], np.where(combined_masks["JPAS"])[0])
    DATA_pd["DESI"] = filter_preserve_type(DATA_pd["DESI"], np.where(combined_masks["DESI"])[0])

    logging.info("│   ├── Final filtering applied successfully.")

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
    total = len(string_array)

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

    # Log counts per category
    logging.info(f"│    ├── 📊 Category Breakdown ({total} total):")
    inverse_mapping = {v: k for k, v in mapping.items()}
    for idx in range(len(mapping)):
        count = np.sum(encoded == idx)
        pct = 100 * count / total if total > 0 else 0.0
        label = inverse_mapping[idx]
        logging.info(f"│    │   ├── {label:<15} → {count:>5} ({pct:.2f}%)")

    logging.info(f"├── ✅ Encoding complete ({len(mapping)} categories).")
    return encoded, mapping

def clean_and_mask_data(
        DATA, apply_masks=['unreliable', "jpas_ignasi_dense", 'apply_additional_filters', 'magic_numbers', 'negative_errors', 'nan_values'],
        mask_indices=[0, -2], magic_numbers=[99, -99], i_band_sn_threshold=20, magnitude_flux_key=None, magnitude_threshold=None, z_lim_QSO_cut=None
    ):
    """
    Main function to clean and mask observational and simulated datasets.

    Parameters:
    - DATA (dict): Input data dictionary.
    - apply_masks (list): List of masks to apply. Available options include:
        'nan_values', 'jpas_ignasi_dense', 'apply_additional_filters',
        'unreliable', 'magic_numbers', 'negative_errors'
    - mask_indices (list): Indices to be removed from the datasets.
    - magic_numbers (list): Values to mask out as invalid.
    - i_band_sn_threshold (float): Threshold for signal-to-noise filtering.
    - z_lim_QSO_cut (float or None): Optional redshift threshold to split QSOs.

    Returns:
    - DATA_clean (dict): Dictionary containing cleaned and masked datasets.
    """

    logging.info("🧽 Cleaning and masking data...")

    JPAS_obs = DATA['JPAS']["all_observations"]
    JPAS_err = DATA['JPAS']["all_errors"]
    DESI_mean = DATA['DESI']["all_np"][..., 0]
    DESI_err = DATA['DESI']["all_np"][..., 2]

    # Step 1: Handle NaN values
    if "nan_values" in apply_masks:
        JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd = remove_invalid_NaN_rows(
            JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA, magic_numbers=magic_numbers
        )
    else:
        DATA_pd = DATA
    del DATA
    gc.collect()
    logging.info("├── 🧹 Deleted cleaned DATA_clean dictionary to free memory.")

    # Optional mask: filter JPAS entries that are in Ignasi’s catalog and dense regions
    if "jpas_ignasi_dense" in apply_masks:
        logging.info("├── Applying JPAS Ignasi ∩ dense-region mask")

        total_before = len(DATA_pd['JPAS']['mask_in_Ignasi'])
        jpas_mask = (
            np.asarray(DATA_pd['JPAS']['mask_in_Ignasi']) &
            (np.asarray(DATA_pd['JPAS']['mask_dense_if_matched']) == 1.0)
        )
        total_after = np.count_nonzero(jpas_mask)
        frac = total_after / total_before if total_before > 0 else 0.0

        logging.info(f"│   ├── JPAS entries before mask: {total_before}")
        logging.info(f"│   ├── JPAS entries after mask:  {total_after}")
        logging.info(f"│   └── Retained fraction: {frac:.3%}")

        for key in DATA_pd['JPAS'].keys():
            DATA_pd['JPAS'][key] = np.asarray(DATA_pd['JPAS'][key])[jpas_mask]

        JPAS_obs = JPAS_obs[jpas_mask]
        JPAS_err = JPAS_err[jpas_mask]

    # Step 2: Apply additional filters
    if "apply_additional_filters" in apply_masks:
        JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd = apply_additional_filters(
            JPAS_obs, JPAS_err, DESI_mean, DESI_err, DATA_pd,
            i_band_sn_threshold=i_band_sn_threshold,
            magnitude_flux_key=magnitude_flux_key, magnitude_threshold=magnitude_threshold
        )

    masks = {}  # Dictionary to store applied masks

    # Step 3: Remove unreliable entries
    if "unreliable" in apply_masks:
        JPAS_obs, JPAS_err, DESI_mean, DESI_err = mask_out_unreliable_mock_data_columns(
            JPAS_obs, JPAS_err, DESI_mean, DESI_err, mask_indices
        )

    # Step 4: Handle magic numbers (e.g., 99, -99)
    if "magic_numbers" in apply_masks:
        JPAS_obs, DESI_mean, magic_masks = mask_magic_numbers_99(JPAS_obs, DESI_mean)
        masks.update(magic_masks)

    # Step 5: Handle negative errors
    if "negative_errors" in apply_masks:
        JPAS_err, DESI_err, error_masks = take_abs_value_of_negative_errors(JPAS_err, DESI_err)
        masks.update(error_masks)
    
    # Step 6: Split between High and Low redshift quasars
    if z_lim_QSO_cut is not None:
        logging.info("├── Splitting between High and Low QSOs, z_lim_QSO_cut: " + str(z_lim_QSO_cut))
        for survey in list(DATA_pd.keys()):
            DATA_pd[survey]['SPECTYPE'] = list(DATA_pd[survey]['SPECTYPE'])
            for ii in range(len(DATA_pd[survey]['SPECTYPE'])):
                if DATA_pd[survey]['SPECTYPE'][ii] == "QSO":
                    if DATA_pd[survey]['REDSHIFT'][ii] < z_lim_QSO_cut:
                        DATA_pd[survey]['SPECTYPE'][ii] = "QSO_low"
                    else:
                        DATA_pd[survey]['SPECTYPE'][ii] = "QSO_high"

    # Step 7: Encode strings to integers (SPECTYPE and MORPHTYPE)
    all_specs = list(DATA_pd["DESI"]['SPECTYPE']) + list(DATA_pd["JPAS"]['SPECTYPE'])
    _, shared_mapping = encode_strings_to_integers(all_specs)
    DESI_SPECTYPE_int, _ = encode_strings_to_integers(list(DATA_pd["DESI"]['SPECTYPE']), reference_mapping=shared_mapping)
    JPAS_SPECTYPE_int, _ = encode_strings_to_integers(list(DATA_pd["JPAS"]['SPECTYPE']), reference_mapping=shared_mapping)

    all_morphs = list(DATA_pd["DESI"]['MORPHTYPE']) + list(DATA_pd["JPAS"]['MORPHTYPE'])
    _, shared_mapping = encode_strings_to_integers(all_morphs)
    DESI_MORPHTYPE_int, _ = encode_strings_to_integers(list(DATA_pd["DESI"]['MORPHTYPE']), reference_mapping=shared_mapping)
    JPAS_MORPHTYPE_int, _ = encode_strings_to_integers(list(DATA_pd["JPAS"]['MORPHTYPE']), reference_mapping=shared_mapping)

    # Step 8: Create clean dictionary
    DATA_clean = {}
    for survey in list(DATA_pd.keys()):
        DATA_clean[survey] = {}
        for key in list(DATA_pd[survey].keys()):
            DATA_clean[survey][key] = DATA_pd[survey][key]
        for key in list(masks.keys()):
            if survey in key:
                new_key = key[len(survey) + 1:]
                DATA_clean[survey][new_key] = masks[key]

    # Step 9: Sample DESI with the corresponding simulated variance
    DESI_obs = np.random.normal(loc=DESI_mean, scale=DESI_err)

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
