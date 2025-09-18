import numpy as np
import logging
import gc
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, List
import os
import pickle

def encode_strings_to_integers(
    string_list: list,
    reference_mapping: dict = None
):
    """
    Encodes strings to integers using a consistent mapping.

    If `reference_mapping` is provided:
      - Values present in the mapping are encoded accordingly.
      - Values NOT present are encoded as -1 (unmapped), and a WARNING is logged
        showing which categories were unmapped and their counts.
      - The provided mapping is returned unchanged.

    If `reference_mapping` is not provided:
      - A new mapping is created from the unique strings in the input list.

    Parameters
    ----------
    string_list : list of str
        Input list to encode.
    reference_mapping : dict, optional
        Predefined mapping {string: int}.

    Returns
    -------
    int_array : np.ndarray
        Encoded integers (unmapped -> -1 when reference_mapping is given).
    mapping : dict
        The mapping used. If reference_mapping was given, it is returned unchanged.
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
        encoded = np.array([mapping[s] for s in string_array], dtype=np.int64)
        logging.info(f"|    ├── 📌 New Mapping Created: {mapping}")
    else:
        mapping = dict(reference_mapping)  # keep caller's mapping unchanged
        # Encode with -1 for unseen labels
        encoded = np.array([mapping.get(s, -1) for s in string_array], dtype=np.int64)

        # Report unmapped categories (assigned -1)
        mask_unmapped = (encoded == -1)
        if mask_unmapped.any():
            unmapped_vals, unmapped_counts = np.unique(string_array[mask_unmapped], return_counts=True)
            missing_list = ", ".join(f"{val}:{cnt}" for val, cnt in zip(unmapped_vals, unmapped_counts))
            logging.warning(f"|    ├── ⚠️ Unmapped categories found (assigned -1): {list(unmapped_vals)}")
            logging.warning(f"|    │      counts → {missing_list}")
        logging.info(f"|    ├── 📌 Used Provided Mapping: {mapping}")

    # Log counts per category (plus unmapped if present)
    logging.info(f"│    ├── 📊 Category Breakdown ({total} total):")
    inverse_mapping = {v: k for k, v in mapping.items()}
    for idx in sorted(inverse_mapping.keys()):
        count = int(np.sum(encoded == idx))
        pct = 100 * count / total if total > 0 else 0.0
        label = inverse_mapping[idx]
        logging.info(f"│    │   ├── {label:<15} → {count:>5} ({pct:.2f}%)")

    # Include unmapped bucket if present
    if np.any(encoded == -1):
        count = int(np.sum(encoded == -1))
        pct = 100 * count / total if total > 0 else 0.0
        logging.info(f"│    │   ├── {'UNMAPPED(-1)':<15} → {count:>5} ({pct:.2f}%)")

    logging.info(f"├── ✅ Encoding complete ({len(mapping)} categories + unmapped bucket if any).")
    return encoded, mapping


def filter_preserve_type(data: Dict[str, Any], valid_indices: np.ndarray) -> Dict[str, Any]:
    """
    Filters dictionary entries while preserving their original types.

    Supported types:
      - NumPy arrays (dtype preserved)
      - Lists of strings (filtered via NumPy, returned as list)

    Other types are left unchanged with a warning.
    """
    out = dict(data)  # don't mutate caller's dict
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            out[key] = value[valid_indices].astype(value.dtype, copy=False)
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            out[key] = np.asarray(value, dtype=object)[valid_indices].tolist()
        else:
            logging.warning(
                f"Key '{key}' has an unsupported type ({type(value)}). "
                "This key was not modified and remains unchanged."
            )
    return out


def remove_NaNs(
    block: Dict[str, Any],
    *,
    check: Literal["obs", "err", "both"] = "both",
    keep_rows_partially_filled_with_NaNs: bool = True,  # NEW: keep rows partially filled with NaNs (interpolate) or drop them
) -> Dict[str, Any]:
    """
    Handle NaNs in a dataset block:

    1) Drop rows that are entirely NaN (per `check`: observations and/or errors).
    2) If keep_rows_partially_filled_with_NaNs=True (default):
         - Save a binary mask of observation NaNs in all_pd["mask_NaNs"] with shape (N, F).
         - Interpolate NaNs along the filter axis (edges use nearest valid).
         - If `check` includes 'err' and errors exist, interpolate NaNs in errors too.
       If keep_rows_partially_filled_with_NaNs=False:
         - Also drop rows that have ANY NaN (i.e., partially NaN rows).
         - No interpolation needed (no NaNs remain in kept rows).
    3) Log counts/percentages for fully-NaN rows and rows with SOME NaNs.

    Returns a NEW cleaned block.
    """
    if "all_pd" not in block or "all_observations" not in block:
        raise KeyError("Block must contain 'all_pd' and 'all_observations'.")

    obs: np.ndarray = block["all_observations"]
    err: Optional[np.ndarray] = block.get("all_errors", None)
    pd_dict: Dict[str, Any] = block["all_pd"]

    if not isinstance(obs, np.ndarray) or obs.ndim != 2:
        raise ValueError("'all_observations' must be a 2D numpy array (N, F).")
    if check in ("err", "both") and not isinstance(err, np.ndarray):
        logging.warning("[clean] 'check' includes 'err' but 'all_errors' missing; using 'obs' only.")
        check = "obs"

    n_rows, n_filters = obs.shape
    logging.info(f"├── remove_NaNs(check='{check}', keep_rows_partially_filled_with_NaNs={keep_rows_partially_filled_with_NaNs})")

    # Build NaN masks on original arrays
    is_nan_obs = np.isnan(obs)
    if check in ("err", "both") and isinstance(err, np.ndarray):
        is_nan_err = np.isnan(err)
    else:
        is_nan_err = np.zeros_like(obs, dtype=bool)

    # Fully-NaN rows to drop
    full_nan_obs = is_nan_obs.all(axis=1)
    full_nan_err = is_nan_err.all(axis=1) if check in ("err", "both") else np.zeros(n_rows, bool)
    full_nan_rows = full_nan_obs | full_nan_err
    n_full = int(full_nan_rows.sum())
    logging.info(f"│   ├── rows fully NaN (drop): {n_full}/{n_rows} ({(n_full / n_rows):.2%})")

    # Rows with some NaNs (but not full)
    any_nan_obs = is_nan_obs.any(axis=1)
    any_nan_err = is_nan_err.any(axis=1) if check in ("err", "both") else np.zeros(n_rows, bool)
    some_nan_rows = (any_nan_obs | any_nan_err) & ~full_nan_rows
    n_some = int(some_nan_rows.sum())
    logging.info(f"│   ├── rows with SOME NaNs:   {n_some}/{n_rows} ({(n_some / n_rows):.2%})")

    # Keep mask (drop fully-NaN always; drop partial if keep_rows_partially_filled_with_NaNs=False)
    if keep_rows_partially_filled_with_NaNs:
        valid_mask = ~full_nan_rows
    else:
        valid_mask = ~(full_nan_rows | some_nan_rows)  # i.e., keep only rows with NO NaNs at all

    kept_idx = np.flatnonzero(valid_mask)
    drop_idx = np.flatnonzero(~valid_mask)
    logging.info(f"│   ├── rows dropped due to policy: {drop_idx.size - n_full if not keep_rows_partially_filled_with_NaNs else n_full}/{n_rows}")
    logging.info(f"│   └── final kept: {kept_idx.size}/{n_rows} ({(kept_idx.size / n_rows):.2%})")

    # Work copies on kept rows
    obs_kept = obs[valid_mask].astype(np.float64, copy=True)
    err_kept = err[valid_mask].astype(np.float64, copy=True) if isinstance(err, np.ndarray) else None
    is_nan_obs_kept = is_nan_obs[valid_mask]
    is_nan_err_kept = is_nan_err[valid_mask] if isinstance(err, np.ndarray) else None

    # Save (full) observation NaN mask; it will be row-sliced by filter_preserve_type
    pd_enhanced = dict(pd_dict)
    pd_enhanced["mask_NaNs"] = is_nan_obs.astype(np.bool_)

    # If keeping partial rows, interpolate NaNs along filter axis
    def _interp_row_inplace(row: np.ndarray, nan_mask_row: np.ndarray) -> None:
        if not nan_mask_row.any():
            return
        x = np.arange(n_filters)
        good = ~nan_mask_row
        if good.sum() == 0:
            return  # fully-NaN rows already removed
        if good.sum() == 1:
            row[~good] = row[good][0]                 # flat-fill if single valid point
        else:
            row[~good] = np.interp(x[~good], x[good], row[good])  # linear + edge hold

    if keep_rows_partially_filled_with_NaNs:
        for i in range(obs_kept.shape[0]):
            _interp_row_inplace(obs_kept[i], is_nan_obs_kept[i])
        if check in ("err", "both") and isinstance(err_kept, np.ndarray):
            for i in range(err_kept.shape[0]):
                _interp_row_inplace(err_kept[i], is_nan_err_kept[i])

    # Assemble cleaned block
    cleaned = dict(block)
    cleaned["all_pd"] = filter_preserve_type(pd_enhanced, kept_idx)
    cleaned["all_observations"] = obs_kept
    if isinstance(err, np.ndarray):
        cleaned["all_errors"] = err_kept
    return cleaned


def remove_magic_rows(
    block: Dict[str, Any],
    magic_numbers: Iterable[float] = (-99, 99),
    *,
    check: Literal["obs", "err", "both"] = "obs",
    keep_rows_partially_filled_with_magic: bool = True,  # NEW: keep rows with some magic (interpolate) or drop them
) -> Dict[str, Any]:
    """
    Handle 'magic numbers' in a dataset block:

    1) Drop rows that are entirely magic numbers (per `check`).
    2) If keep_rows_partially_filled_with_magic=True (default):
         - Add binary masks in all_pd: mask_magic_<magic> with shape (N, F)
           (True where that magic value occurs in observations).
         - Interpolate magic values in observations along the filter axis
           (edges use nearest valid). If `check` includes 'err' and errors
           exist, interpolate magic values in errors too.
       If keep_rows_partially_filled_with_magic=False:
         - Also drop rows that have ANY magic (i.e., partially-magic rows).
         - No interpolation needed (no magic remains in kept rows).
    3) Log counts/percentages for fully-magic and partially-magic rows.

    Returns a new block with filtered content and masks saved in all_pd.
    """
    if "all_pd" not in block or "all_observations" not in block:
        raise KeyError("Block must contain 'all_pd' and 'all_observations'.")

    obs: np.ndarray = block["all_observations"]
    err: Optional[np.ndarray] = block.get("all_errors", None)
    pd_dict: Dict[str, Any] = block["all_pd"]

    if not isinstance(obs, np.ndarray) or obs.ndim != 2:
        raise ValueError("'all_observations' must be a 2D numpy array (N, F).")
    if check in ("err", "both") and not isinstance(err, np.ndarray):
        logging.warning("[clean] 'check' includes 'err' but 'all_errors' missing; using 'obs' only.")
        check = "obs"

    n_rows, n_filters = obs.shape
    magic = np.asarray(list(magic_numbers), dtype=float)

    logging.info(f"├── remove_magic_rows(check='{check}', keep_rows_partially_filled_with_magic={keep_rows_partially_filled_with_magic}, magic_numbers={list(magic_numbers)})")

    # --- Magic masks (on original arrays)
    is_magic_obs = np.isin(obs, magic)
    if check in ("err", "both") and isinstance(err, np.ndarray):
        is_magic_err = np.isin(err, magic)
    else:
        is_magic_err = np.zeros_like(obs, dtype=bool)

    # --- Fully-magic rows (drop)
    full_magic_obs = is_magic_obs.all(axis=1)
    full_magic_err = is_magic_err.all(axis=1) if check in ("err", "both") else np.zeros(n_rows, bool)
    full_magic_rows = full_magic_obs | full_magic_err

    n_full = int(full_magic_rows.sum())
    logging.info(f"│   ├── rows fully magic (drop): {n_full}/{n_rows} ({(n_full / n_rows):.2%})")

    # --- Partially-magic rows (some but not full)
    any_magic_obs = is_magic_obs.any(axis=1)
    any_magic_err = is_magic_err.any(axis=1) if check in ("err", "both") else np.zeros(n_rows, bool)
    some_magic_rows = (any_magic_obs | any_magic_err) & ~full_magic_rows

    n_some = int(some_magic_rows.sum())
    logging.info(f"│   ├── rows with SOME magic:  {n_some}/{n_rows} ({(n_some / n_rows):.2%})")

    # --- Keep policy
    if keep_rows_partially_filled_with_magic:
        valid_mask = ~full_magic_rows
        extra_drop = 0
    else:
        valid_mask = ~(full_magic_rows | some_magic_rows)  # keep only rows with NO magic at all
        extra_drop = int(some_magic_rows.sum())

    kept_idx = np.flatnonzero(valid_mask)
    drop_idx = np.flatnonzero(~valid_mask)

    if not keep_rows_partially_filled_with_magic:
        logging.info(f"│   ├── rows dropped due to partial-magic policy: {extra_drop}/{n_rows} ({(extra_drop/n_rows):.2%})")
    logging.info(f"│   └── final kept: {kept_idx.size}/{n_rows} ({(kept_idx.size / n_rows):.2%})")

    # --- Work copies on kept rows
    obs_kept = obs[valid_mask].astype(np.float64, copy=True)
    err_kept = err[valid_mask].astype(np.float64, copy=True) if isinstance(err, np.ndarray) else None
    is_magic_obs_kept = is_magic_obs[valid_mask]
    is_magic_err_kept = is_magic_err[valid_mask] if isinstance(err, np.ndarray) else None

    # --- Build mask_magic_* (binary masks on full array; row-sliced later)
    pd_enhanced = dict(pd_dict)
    for m in magic:
        full_mask = (obs == m)  # observations only
        key = f"mask_magic_{int(m) if np.isclose(m, int(round(m))) else str(m)}"
        pd_enhanced[key] = full_mask.astype(np.bool_)

    # --- Interpolate (only if we kept partial rows)
    def _interp_row(row: np.ndarray, magic_mask_row: np.ndarray) -> None:
        if not magic_mask_row.any():
            return
        x = np.arange(n_filters)
        good = ~magic_mask_row
        if good.sum() == 0:
            return  # fully-magic rows already pruned
        if good.sum() == 1:
            row[~good] = row[good][0]                # flat fill if single valid point
        else:
            row[~good] = np.interp(x[~good], x[good], row[good])  # linear + edge hold

    if keep_rows_partially_filled_with_magic:
        for i in range(obs_kept.shape[0]):
            _interp_row(obs_kept[i], is_magic_obs_kept[i])
        if check in ("err", "both") and isinstance(err_kept, np.ndarray):
            for i in range(err_kept.shape[0]):
                _interp_row(err_kept[i], is_magic_err_kept[i])

    # --- Assemble cleaned block
    cleaned = dict(block)
    cleaned["all_pd"] = filter_preserve_type(pd_enhanced, kept_idx)
    cleaned["all_observations"] = obs_kept
    if isinstance(err, np.ndarray):
        cleaned["all_errors"] = err_kept
    return cleaned


def apply_selection_cuts(
    block: Dict[str, Any],
    *,
    i_band_sn_threshold: float = 20.0,
    magnitude_flux_key: Optional[str] = None,
    magnitude_threshold: Optional[float] = None,
    i_band_index: int = -1,   # last column by default
) -> Dict[str, Any]:
    """
    Block-style apply_selection_cuts:
      1) i-band S/N >= threshold (computed from all_observations/all_errors at i_band_index)
      2) optional magnitude cut from a flux column in all_pd:  mag = 22.5 - 2.5*log10(flux) <= threshold

    Returns a NEW cleaned block (does not mutate the input).

    Expected block keys:
      - block["all_observations"]: (N, F) ndarray
      - block["all_errors"]      : (N, F) ndarray
      - block["all_pd"]          : dict of 1D arrays/lists
    """
    # --- validations
    if "all_pd" not in block or "all_observations" not in block:
        raise KeyError("Block must contain 'all_pd' and 'all_observations'.")
    obs = block["all_observations"]
    err = block.get("all_errors", None)
    pd_dict = block["all_pd"]

    if not isinstance(obs, np.ndarray) or obs.ndim != 2:
        raise ValueError("'all_observations' must be a 2D numpy array (N, F).")
    if not isinstance(err, np.ndarray) or err.ndim != 2:
        raise ValueError("'all_errors' must be a 2D numpy array (N, F).")

    n_rows, n_bands = obs.shape
    # normalize i-band index to python indexing
    if not (-n_bands <= i_band_index < n_bands):
        raise IndexError(f"i_band_index {i_band_index} out of range for F={n_bands}.")

    logging.info("├── apply_selection_cuts()")

    # ---------- 1) i-band S/N cut ----------
    logging.info(f"│   ├── i-band S/N ≥ {i_band_sn_threshold}")
    num = obs[:, i_band_index]
    den = np.abs(err[:, i_band_index])
    with np.errstate(divide="ignore", invalid="ignore"):
        sn = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=(den > 0))
    sn_mask = sn >= float(i_band_sn_threshold)
    kept_sn = int(sn_mask.sum())
    logging.info(f"│   │   └── rows kept after S/N: {kept_sn}/{n_rows} ({kept_sn/n_rows:.2%})")

    # ---------- 2) optional magnitude cut from flux column ----------
    mag_mask = np.ones(n_rows, dtype=bool)
    if magnitude_flux_key is not None and magnitude_threshold is not None:
        if magnitude_flux_key in pd_dict:
            logging.info(f"│   ├── mag({magnitude_flux_key}) ≤ {magnitude_threshold}")
            flux = np.asarray(pd_dict[magnitude_flux_key], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                mag = 22.5 - 2.5 * np.log10(flux)
            mag_mask = np.isfinite(mag) & (mag <= float(magnitude_threshold))
            kept_mag = int(mag_mask.sum())
            logging.info(f"│   │   └── rows kept after mag: {kept_mag}/{n_rows} ({kept_mag/n_rows:.2%})")
        else:
            logging.warning(f"│   ├── flux key '{magnitude_flux_key}' not in all_pd → skipping magnitude cut")

    # ---------- combine & filter ----------
    combined = sn_mask & mag_mask
    kept_idx = np.flatnonzero(combined)
    drop_idx = np.flatnonzero(~combined)
    logging.info(f"│   └── final kept: {kept_idx.size}/{n_rows} ({kept_idx.size/n_rows:.2%})")

    # build cleaned copy
    cleaned = dict(block)
    cleaned["all_pd"] = filter_preserve_type(pd_dict, kept_idx)
    cleaned["all_observations"] = obs[combined]
    cleaned["all_errors"] = err[combined]
    return cleaned


def mask_out_unreliable_columns(
    block: Dict[str, Any],
    mask_unreliable_filters_indices: Iterable[int],
) -> Dict[str, Any]:
    """
    Remove specified filter columns (by index) across a dataset block.

    Applies the column mask to:
      - block["all_observations"] (N, F)
      - block["all_errors"] (N, F) if present
      - any 2D np.ndarrays in block["all_pd"] with shape (N, F)

    Parameters
    ----------
    block : dict
        Dataset block with keys: "all_pd", "all_observations", optionally "all_errors".
    mask_unreliable_filters_indices : iterable of int
        Column indices to remove (supports negatives).

    Returns
    -------
    cleaned : dict
        New block with masked columns removed and shapes updated.
    """
    if "all_observations" not in block or "all_pd" not in block:
        raise KeyError("Block must contain 'all_observations' and 'all_pd'.")

    obs = block["all_observations"]
    err = block.get("all_errors", None)
    pd_dict = block["all_pd"]

    if not isinstance(obs, np.ndarray) or obs.ndim != 2:
        raise ValueError("'all_observations' must be a 2D numpy array of shape (N, F).")

    n_rows, n_filters = obs.shape
    logging.info(f"├── mask_out_unreliable_columns(mask_unreliable_filters_indices={list(mask_unreliable_filters_indices)})")

    # Normalize/validate indices
    raw = list(mask_unreliable_filters_indices)
    norm = []
    for idx in raw:
        j = idx % n_filters  # supports negatives
        if 0 <= j < n_filters:
            norm.append(j)
    if len(norm) != len(raw):
        logging.warning("│   ├── Some mask indices were out of range and ignored.")

    rm_idx = sorted(set(norm))
    if not rm_idx:
        logging.info("│   └── Nothing to mask (empty/invalid indices). Returning original block.")
        return dict(block)

    keep_mask = np.ones(n_filters, dtype=bool)
    keep_mask[rm_idx] = False
    kept = int(keep_mask.sum())

    # Slice arrays
    new_obs = obs[:, keep_mask]
    new_err = err[:, keep_mask] if isinstance(err, np.ndarray) and err.ndim == 2 else err

    # Slice any (N, F) arrays inside all_pd as well
    new_pd = dict(pd_dict)
    for k, v in pd_dict.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] == n_rows and v.shape[1] == n_filters:
            new_pd[k] = v[:, keep_mask].astype(v.dtype, copy=False)

    logging.info(f"│   ├── Removed columns: {rm_idx}")
    logging.info(f"│   ├── New #filters: {kept}")
    logging.info(f"│   ├── Updated observations shape: {new_obs.shape}")
    if isinstance(new_err, np.ndarray):
        logging.info(f"│   ├── Updated errors shape: {new_err.shape}")
    # (2D arrays in all_pd updated silently; add logs if you track specific keys)

    cleaned = dict(block)
    cleaned["all_pd"] = new_pd
    cleaned["all_observations"] = new_obs
    if isinstance(err, np.ndarray):
        cleaned["all_errors"] = new_err
    return cleaned


def fix_and_mask_negative_errors(
    block: Dict[str, Any],
    *,
    mask_key: str = "mask_negative_errors",
) -> Dict[str, Any]:
    """
    Detect negative entries in all_errors, log stats, store a binary mask (N,F)
    in all_pd[mask_key], and replace negatives with their absolute value.

    Parameters
    ----------
    block : dict
        Dataset block with:
          - "all_observations": (N, F) ndarray   [not modified]
          - "all_errors"      : (N, F) ndarray   [abs() applied where < 0]
          - "all_pd"          : dict             [mask stored here]

    mask_key : str
        Name for the mask stored in all_pd (default: "mask_negative_errors").

    Returns
    -------
    cleaned : dict
        New block with updated "all_errors" and mask saved into "all_pd".
    """
    if "all_pd" not in block or "all_errors" not in block:
        raise KeyError("Block must contain 'all_pd' and 'all_errors'.")

    err = block["all_errors"]
    if not isinstance(err, np.ndarray) or err.ndim != 2:
        raise ValueError("'all_errors' must be a 2D numpy array (N, F).")

    n_rows, n_filters = err.shape
    logging.info("├── fix_and_mask_negative_errors()")

    # Binary mask of negatives (before fixing)
    neg_mask = (err < 0)
    rows_with_neg = neg_mask.any(axis=1)
    n_rows_with_neg = int(rows_with_neg.sum())
    pct_rows_with_neg = n_rows_with_neg / n_rows if n_rows > 0 else 0.0

    logging.info(
        f"│   ├── rows with some negative errors: {n_rows_with_neg}/{n_rows} ({pct_rows_with_neg:.2%})"
    )

    # (optional extra info)
    rows_all_neg = neg_mask.all(axis=1)
    n_rows_all_neg = int(rows_all_neg.sum())
    if n_rows_all_neg > 0:
        logging.info(
            f"│   ├── rows with all errors negative: {n_rows_all_neg}/{n_rows} ({n_rows_all_neg/n_rows:.2%})"
        )

    # Replace negatives with absolute values
    fixed_err = np.abs(err, dtype=np.float64)

    # Save mask in all_pd (shape: N x F)
    new_pd = dict(block["all_pd"])
    new_pd[mask_key] = neg_mask.astype(np.bool_)

    # Build cleaned block (do not mutate input)
    cleaned = dict(block)
    cleaned["all_pd"] = new_pd
    cleaned["all_errors"] = fixed_err
    return cleaned


def split_qso_by_redshift(
    block: Dict[str, Any],
    *,
    z_lim_QSO_cut: float,
    spectype_key: str = "SPECTYPE",
    redshift_key: str = "REDSHIFT",
    redshift_key_fallback: str = "Z_DESI",
    qso_label: str = "QSO",
    low_z_QSO_label: str = "QSO_low",
    high_z_QSO_label: str = "QSO_high"
) -> Dict[str, Any]:
    """
    Relabel SPECTYPE 'QSO' into 'QSO_low' / 'QSO_high' using a redshift cutoff.
    - Uses `redshift_key` if present; otherwise tries `redshift_key_fallback`.
    - Non-finite (NaN) redshifts leave SPECTYPE unchanged.
    - Uses object-dtype for strings to avoid truncation.
    """
    if "all_pd" not in block:
        raise KeyError("Block must contain 'all_pd'.")

    pd_dict = block["all_pd"]
    if spectype_key not in pd_dict:
        logging.warning(f"[split_qso_by_redshift] '{spectype_key}' not in all_pd; nothing to do.")
        return dict(block)

    # choose redshift source: primary -> fallback
    if redshift_key in pd_dict:
        rz_key = redshift_key
    elif redshift_key_fallback in pd_dict:
        rz_key = redshift_key_fallback
        logging.info(f"[split_qso_by_redshift] Using fallback redshift key '{rz_key}'.")
    else:
        logging.warning(
            f"[split_qso_by_redshift] Neither '{redshift_key}' nor fallback '{redshift_key_fallback}' found; nothing done."
        )
        return dict(block)

    # object dtype to avoid fixed-width Unicode truncation
    spectype = np.array(pd_dict[spectype_key], dtype=object)

    # robust redshift coercion to float with NaN for bad/empty values
    rz_raw = np.array(pd_dict[rz_key], dtype=object)

    def _to_float(x):
        try:
            if x is None:
                return np.nan
            if isinstance(x, str):
                s = x.strip()
                if s == "" or s.lower() in ("nan", "none"):
                    return np.nan
                return float(s)
            return float(x)
        except Exception:
            return np.nan

    redshift = np.vectorize(_to_float, otypes=[float])(rz_raw)

    n = spectype.shape[0]
    if redshift.shape[0] != n:
        raise ValueError(
            f"[split_qso_by_redshift] Length mismatch: {spectype_key} ({n}) vs {rz_key} ({redshift.shape[0]})."
        )

    is_qso    = (spectype == qso_label)
    finite_z  = np.isfinite(redshift)
    low_mask  = is_qso & finite_z & (redshift <  float(z_lim_QSO_cut))
    high_mask = is_qso & finite_z & (redshift >= float(z_lim_QSO_cut))

    n_qso = int(is_qso.sum())
    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())
    n_qso_missing_z = int((is_qso & ~finite_z).sum())

    logging.info(f"├── split_qso_by_redshift(z_lim_QSO_cut={z_lim_QSO_cut}, redshift_key='{rz_key}')")
    logging.info(f"│   ├── total QSOs: {n_qso}")
    logging.info(f"│   ├── relabeled: low={n_low}, high={n_high}")
    if n_qso_missing_z > 0:
        logging.info(f"│   └── QSOs with non-finite {rz_key} (unchanged): {n_qso_missing_z}")

    # Assign using object array (no truncation)
    spectype[low_mask]  = low_z_QSO_label
    spectype[high_mask] = high_z_QSO_label

    cleaned = dict(block)
    new_pd = dict(pd_dict)
    new_pd[spectype_key] = spectype.tolist()
    cleaned["all_pd"] = new_pd
    return cleaned


def encode_keys_strings_to_integers(
    block: Dict[str, Any],
    *,
    columns: Optional[Dict[str, str]] = None,          # optional; if None, derived from shared_mappings' keys
    shared_mappings: Optional[Dict[str, Dict[str, int]]] = None,
    target_suffix: str = "_int",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    """
    Encode categorical string columns in block['all_pd'] using
    `encode_strings_to_integers`. If `columns` is not provided, the source
    columns are taken from the first-level keys of `shared_mappings`, and
    output keys are auto-named as `<source><target_suffix>`.

    Parameters
    ----------
    block : dict
        Dataset block with key "all_pd".
    columns : dict[str, str] | None
        Optional mapping {source_key -> output_key}. If None, derive from
        `shared_mappings` as {k: f"{k}{target_suffix}"}.
    shared_mappings : dict[str, dict[str,int]] | None
        Optional per-column string->int mapping. If a mapping is missing
        for a column, a new one is learned from THIS BLOCK’s values.
    target_suffix : str
        Suffix for auto output keys when columns is None (default "_int").

    Returns
    -------
    new_block : dict
        Copy of `block` with encoded arrays added to all_pd.
    used_mappings : dict[str, dict[str,int]]
        The mapping actually used per column (provided or newly created).
    """
    if "all_pd" not in block:
        raise KeyError("Block must contain 'all_pd'.")

    pd_dict = block["all_pd"]
    shared_mappings = shared_mappings or {}

    # If columns not provided, infer from shared_mappings' keys
    if columns is None:
        if not shared_mappings:
            raise ValueError(
                "When 'columns' is None, 'shared_mappings' must provide the source columns as its first-level keys."
            )
        columns = {src_key: f"{src_key}{target_suffix}" for src_key in shared_mappings.keys()}

    new_pd = dict(pd_dict)

    for src_key, out_key in columns.items():
        if src_key not in pd_dict:
            logging.warning(f"[encode_keys_strings_to_integers] '{src_key}' not in all_pd; skipping.")
            continue

        # Coerce to list[str] for the encoder
        values_list = np.asarray(pd_dict[src_key]).astype(str).tolist()

        ref_map = shared_mappings.get(src_key)  # may be None -> learn from this block
        encoded, mapping = encode_strings_to_integers(values_list, reference_mapping=ref_map)

        # Store encoded ints as int64 NumPy array
        new_pd[out_key] = encoded.astype(np.int64, copy=False)

    new_block = dict(block)
    new_block["all_pd"] = new_pd
    return new_block


def collapse_spectype_one_vs_rest(
    block: Dict[str, Any],
    *,
    positive_collapse_label: Optional[str],
    spectype_key: str = "SPECTYPE",
    negative_prefix: str = "no_",
) -> Dict[str, Any]:
    """
    Convert SPECTYPE into a one-vs-rest scheme:
      - Entries equal to `positive_collapse_label` stay as `positive_collapse_label`
      - All other entries become f"{negative_prefix}{positive_label}"

    If `positive_collapse_label` is None, this function is a no-op and returns the block unchanged.
    """
    if "all_pd" not in block:
        raise KeyError("Block must contain 'all_pd'.")

    if positive_collapse_label is None:
        logging.info("├── collapse_spectype_one_vs_rest: positive_collapse_label=None → no collapse performed.")
        return dict(block)

    pd_dict = block["all_pd"]
    if spectype_key not in pd_dict:
        logging.warning(f"[collapse_spectype_one_vs_rest] '{spectype_key}' not in all_pd; nothing to do.")
        return dict(block)

    # Use object dtype to avoid fixed-width Unicode truncation
    spec = np.array(pd_dict[spectype_key], dtype=object)
    pos_mask = (spec == positive_collapse_label)

    n_total = spec.size
    n_pos = int(pos_mask.sum())
    n_neg = n_total - n_pos

    logging.info(f"├── collapse_spectype_one_vs_rest(positive_collapse_label='{positive_collapse_label}')")
    logging.info(f"│   ├── positives kept: {n_pos}/{n_total} ({n_pos/n_total:.2%})")
    logging.info(f"│   └── negatives set to '{negative_prefix}{positive_collapse_label}': {n_neg}/{n_total} ({n_neg/n_total:.2%})")

    neg_label = f"{negative_prefix}{positive_collapse_label}"
    spec[~pos_mask] = neg_label

    new_block = dict(block)
    new_pd = dict(pd_dict)
    new_pd[spectype_key] = spec.tolist()
    new_block["all_pd"] = new_pd
    return new_block


def apply_normalization_from_file(
    block: Dict[str, Any],
    *,
    file_path: str,
    obs_key: str = "all_observations",
    err_key: str = "all_errors",
    pd_key: str  = "all_pd",
    suffix: str  = "_normalized",
) -> Dict[str, Any]:
    """
    Load precomputed means/stds (pickle with NORM schema) and normalize:
      - all_observations -> all_observations_normalized
      - all_errors      -> all_errors_normalized
      - selected all_pd numeric keys -> <key>_normalized

    Also stores the USED stats under block["normalization"].
    """
    logging.info(f"[apply_normalization_from_file] Starting normalization using file: {file_path}")

    if not os.path.isfile(file_path):
        logging.error(f"[apply_normalization_from_file] File not found: {file_path}")
        return dict(block)

    with open(file_path, "rb") as f:
        NORM = pickle.load(f)
    logging.info("[apply_normalization_from_file] Loaded normalization dictionary.")

    means = NORM.get("mean", {})
    stds  = NORM.get("std", {})
    logging.info(f"[apply_normalization_from_file] Available mean keys: {list(means.keys())}")
    logging.info(f"[apply_normalization_from_file] Available std  keys: {list(stds.keys())}")

    cleaned = dict(block)
    pd_dict = cleaned.get(pd_key, {})
    norm_used = {"source": file_path, "mean": {}, "std": {}}

    # --- helper for (x - mu)/sigma with safe sigma
    def _zscore(X: np.ndarray, mu: np.ndarray) -> np.ndarray:
        sigma = stds_arr = None
        return X  # placeholder overwritten below

    # --- normalize all_observations
    if obs_key in cleaned and obs_key in means and obs_key in stds:
        X = np.asarray(cleaned[obs_key], dtype=float)
        mu = np.asarray(means[obs_key], dtype=float)
        sd = np.asarray(stds[obs_key], dtype=float)
        logging.info(f"[apply_normalization_from_file] Normalizing '{obs_key}' with shapes: X={X.shape}, mean={mu.shape}, std={sd.shape}")

        if mu.shape != sd.shape or mu.shape != X.shape[1:]:
            logging.error(f"[apply_normalization_from_file] Shape mismatch for '{obs_key}': "
                          f"X {X.shape}, mean {mu.shape}, std {sd.shape} — skipping.")
        else:
            sd_safe = sd.copy()
            zero = (sd_safe == 0)
            if zero.any():
                logging.warning(f"[apply_normalization_from_file] {obs_key}: {zero.sum()} std entries are zero; using 1.0 there.")
                sd_safe[zero] = 1.0
            Xn = (X - mu) / sd_safe
            cleaned[f"{obs_key}{suffix}"] = Xn
            norm_used["mean"][obs_key] = mu
            norm_used["std"][obs_key]  = sd
            logging.info(f"[apply_normalization_from_file] Wrote '{obs_key}{suffix}' with shape {Xn.shape}")
    else:
        logging.info(f"[apply_normalization_from_file] Skipping '{obs_key}' normalization (keys missing in block or NORM).")

    # --- normalize all_errors
    if err_key in cleaned and err_key in means and err_key in stds:
        E = np.asarray(cleaned[err_key], dtype=float)
        mu = np.asarray(means[err_key], dtype=float)
        sd = np.asarray(stds[err_key], dtype=float)
        logging.info(f"[apply_normalization_from_file] Normalizing '{err_key}' with shapes: E={E.shape}, mean={mu.shape}, std={sd.shape}")

        if mu.shape != sd.shape or mu.shape != E.shape[1:]:
            logging.error(f"[apply_normalization_from_file] Shape mismatch for '{err_key}': "
                          f"E {E.shape}, mean {mu.shape}, std {sd.shape} — skipping.")
        else:
            sd_safe = sd.copy()
            zero = (sd_safe == 0)
            if zero.any():
                logging.warning(f"[apply_normalization_from_file] {err_key}: {zero.sum()} std entries are zero; using 1.0 there.")
                sd_safe[zero] = 1.0
            En = (E - mu) / sd_safe
            cleaned[f"{err_key}{suffix}"] = En
            norm_used["mean"][err_key] = mu
            norm_used["std"][err_key]  = sd
            logging.info(f"[apply_normalization_from_file] Wrote '{err_key}{suffix}' with shape {En.shape}")
    else:
        logging.info(f"[apply_normalization_from_file] Skipping '{err_key}' normalization (keys missing in block or NORM).")

    # --- normalize selected all_pd numeric keys
    pd_norm_mean = means.get(pd_key, {})
    pd_norm_std  = stds.get(pd_key, {})
    if isinstance(pd_norm_mean, dict) and isinstance(pd_norm_std, dict):
        logging.info(f"[apply_normalization_from_file] all_pd normalization candidates: {list(pd_norm_mean.keys())}")
        new_pd = dict(pd_dict)
        used_pd_means = {}
        used_pd_stds  = {}
        n_done = 0
        for k, mu in pd_norm_mean.items():
            if k not in pd_dict:
                logging.warning(f"[apply_normalization_from_file] all_pd['{k}'] not found; skipping.")
                continue
            if k not in pd_norm_std:
                logging.warning(f"[apply_normalization_from_file] std for all_pd['{k}'] missing; skipping.")
                continue

            vals = np.asarray(pd_dict[k])
            if not np.issubdtype(vals.dtype, np.number):
                logging.warning(f"[apply_normalization_from_file] all_pd['{k}'] is not numeric (dtype={vals.dtype}); skipping.")
                continue

            mu = float(np.asarray(mu).reshape(()))   # scalar
            sd = float(np.asarray(pd_norm_std[k]).reshape(()))
            sd_safe = 1.0 if sd == 0.0 else sd
            if sd == 0.0:
                logging.warning(f"[apply_normalization_from_file] all_pd['{k}']: std is zero; using 1.0.")

            new_pd[f"{k}{suffix}"] = (vals.astype(float) - mu) / sd_safe
            used_pd_means[k] = mu
            used_pd_stds[k]  = sd
            n_done += 1
            logging.info(f"[apply_normalization_from_file] Wrote all_pd['{k}{suffix}'] (len={len(new_pd[f'{k}{suffix}'])})")

        cleaned[pd_key] = new_pd
        if used_pd_means:
            norm_used["mean"][pd_key] = used_pd_means
            norm_used["std"][pd_key]  = used_pd_stds
        logging.info(f"[apply_normalization_from_file] all_pd normalized keys count: {n_done}")
    else:
        logging.info("[apply_normalization_from_file] No all_pd normalization directives found in NORM file.")

    cleaned["normalization"] = norm_used
    logging.info(f"[apply_normalization_from_file] Normalization complete. Stored stats for keys: {list(norm_used['mean'].keys())}")
    return cleaned


def clean_data_pipeline(
    DATA: Dict[str, Any],
    *,
    config: Dict[str, Any],
    apply_order: Optional[List[str]] = None,
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Run the end-to-end cleaning pipeline over all blocks in DATA using a modular config.

    Expected config schema (all keys optional; sensible defaults assumed by your functions):
    {
      "mask_unreliable": {
        "enabled": True,
        "indices": [0, -2],                         # mask_unreliable_filters_indices
      },
      "nan": {
        "enabled": True,
        "keep_partial": False,                      # keep_rows_partially_filled_with_NaNs
      },
      ...
    }

    Parameters
    ----------
    DATA : dict[str, block]
        Each block must follow your { "all_pd", "all_observations", "all_errors" } convention.
    config : dict
        See schema above.
    apply_order : list[str] | None
        Optional explicit step order. Defaults to the canonical order below.
    in_place : bool
        If False (default), returns a new DATA dict; if True, mutates the input.

    Returns
    -------
    dict
        Cleaned DATA (new dict unless in_place=True).
    """
    # canonical order
    default_order = [
        "mask_unreliable",
        "nan",
        "magic",
        "selection",
        "neg_errors",
        "qso_split",
        "collapse",
        "encoding",
        "normalization"
    ]
    steps = apply_order or default_order

    out = DATA if in_place else {k: v for k, v in DATA.items()}

    for dset_key in list(out.keys()):
        logging.info(f"🧹 Cleaning dataset: {dset_key}")
        block = out[dset_key]

        for step in steps:
            cfg = config.get(step, {})
            if not cfg or not cfg.get("enabled", True):
                continue

            if step == "mask_unreliable":
                idx = cfg.get("indices", [])
                block = mask_out_unreliable_columns(
                    block,
                    mask_unreliable_filters_indices=idx
                )

            elif step == "nan":
                keep_partial = cfg.get("keep_partial", False)
                block = remove_NaNs(
                    block,
                    keep_rows_partially_filled_with_NaNs=keep_partial
                )

            elif step == "magic":
                vals = cfg.get("values", (-99, 99))
                keep_partial = cfg.get("keep_partial", False)
                block = remove_magic_rows(
                    block,
                    magic_numbers=vals,
                    keep_rows_partially_filled_with_magic=keep_partial
                )

            elif step == "selection":
                block = apply_selection_cuts(
                    block,
                    i_band_sn_threshold=cfg.get("i_band_sn_threshold", -999999),
                    magnitude_flux_key=cfg.get("magnitude_flux_key", "DESI_FLUX_R"),
                    magnitude_threshold=cfg.get("magnitude_threshold", 999999),
                )

            elif step == "neg_errors":
                block = fix_and_mask_negative_errors(block)

            elif step == "qso_split":
                block = split_qso_by_redshift(
                    block,
                    z_lim_QSO_cut=cfg.get("z_lim_QSO_cut", 2.1),
                    spectype_key=cfg.get("spectype_key", "SPECTYPE"),
                    redshift_key=cfg.get("redshift_key", "REDSHIFT"),
                    redshift_key_fallback=cfg.get("redshift_key_fallback", "Z_DESI"),
                    qso_label=cfg.get("qso_label", "QSO"),
                    low_z_QSO_label=cfg.get("low_label", "QSO_low"),
                    high_z_QSO_label=cfg.get("high_label", "QSO_high"),
                )

            elif step == "collapse":
                block = collapse_spectype_one_vs_rest(
                    block,
                    positive_collapse_label=cfg.get("positive_collapse_label", None),
                    spectype_key=cfg.get("spectype_key", "SPECTYPE"),
                    negative_prefix=cfg.get("negative_prefix", "no_"),
                )

            elif step == "encoding":
                block = encode_keys_strings_to_integers(
                    block,
                    shared_mappings=cfg.get("shared_mappings", {})
                )

            elif step == "normalization":
                file_path = cfg.get("file_path", None)
                if not file_path:
                    logging.warning("[clean_data_pipeline] 'normalization' enabled but no 'file_path' provided; skipping.")
                else:
                    block = apply_normalization_from_file(
                        block,
                        file_path=file_path,
                        obs_key=cfg.get("obs_key", "all_observations"),
                        err_key=cfg.get("err_key", "all_errors"),
                        pd_key=cfg.get("pd_key", "all_pd"),
                        suffix=cfg.get("suffix", "_normalized"),
                    )

            else:
                logging.warning(f"[clean_data_pipeline] Unknown step '{step}' — skipping.")

        out[dset_key] = block

    return out