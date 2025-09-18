import os
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import fitsio
import pickle


def load_JPAS_x_DESI_Raul(
    data: Dict[str, Any],
    root_path: str,
    datasets: List[Dict[str, Any]],
    random_seed: int = None
) -> Dict[str, Any]:
    """
    Loads JPAS_x_DESI_Raul datasets into a structured dictionary, with optional percentage-based sampling.

    Parameters:
    - data (dict): Dictionary to populate.
    - root_path (str): Path where the data files are located.
    - datasets (list of dict): Each dict must contain:
        - 'name': identifier string
        - 'npy': filename of structured NumPy array
        - 'csv': filename of CSV file
        - 'pickle': filename of pickle file (optional)
        - 'sample_percentage': optional float in (0, 1], default 1.0
    - random_seed (int, optional): Seed for reproducible sampling

    Returns:
    - dict: Updated `data` dictionary
    """
    logging.info("├─── 📥 Starting JPAS_x_DESI_Raul dataset loading...")
    rng = np.random.default_rng(seed=random_seed)

    for dset in datasets:
        name = dset["name"]
        npy_file = os.path.join(root_path, dset["npy"])
        csv_file = os.path.join(root_path, dset["csv"])
        pickle_file = os.path.join(root_path, dset["pickle"])
        pct = float(dset.get("sample_percentage", 1.0))

        logging.info(f"|    ├─── 🔹 Dataset: {name} (sample {pct:.0%})")

        # Load CSV
        try:
            df_full = pd.read_csv(csv_file)
            total_rows = len(df_full)
            if pct < 1.0:
                n_sample = int(pct * total_rows)
                selected_indices = np.sort(rng.choice(total_rows, size=n_sample, replace=False))
                df = df_full.iloc[selected_indices]
            else:
                selected_indices = np.arange(total_rows)
                df = df_full

            data[f"{name}_pd"] = {}
            for col in df.columns:
                dtype = df[col].dtype
                if np.issubdtype(dtype, np.integer):
                    data[f"{name}_pd"][col] = df[col].astype(np.int64).to_numpy()
                elif np.issubdtype(dtype, np.floating):
                    data[f"{name}_pd"][col] = df[col].astype(np.float64).to_numpy()
                else:
                    data[f"{name}_pd"][col] = df[col].astype(str).tolist()

            logging.info(f"|    |    ✔ CSV loaded: {dset['csv']} (shape: {df.shape})")
        except Exception as e:
            logging.error(f"|    |    ❌ Failed to load CSV '{dset['csv']}': {e}")
            continue

        # Load NPY
        try:
            np_data = np.load(npy_file)
            field_names = np_data.dtype.names
            if len(field_names) < 2:
                raise ValueError(f"Expected at least 2 fields in npy structured array, got {field_names}")

            key_obs, key_err = field_names[:2]

            np_sample = np_data[selected_indices]
            data[f"{name}_observations"] = np_sample[key_obs]
            data[f"{name}_errors"] = np_sample[key_err] * np.abs(np_sample[key_obs])

            logging.info(f"|    |    ✔ NPY loaded: {dset['npy']} (obs shape: {data[f'{name}_observations'].shape})")
        except Exception as e:
            logging.error(f"|    |    ❌ Failed to load NPY '{dset['npy']}': {e}")
            continue

        # Load Pickle and integrate its contents into the CSV-derived dict
        if pickle_file and os.path.exists(pickle_file):
            try:
                with open(pickle_file, "rb") as f:
                    aux_dict = pickle.load(f)

                for k, v in aux_dict.items():
                    v = np.asarray(v)
                    if v.shape[0] == len(selected_indices):
                        data[f"{name}_pd"][k] = v[selected_indices]
                    elif v.shape[0] == total_rows:
                        data[f"{name}_pd"][k] = v[selected_indices]
                    else:
                        raise ValueError(f"Shape mismatch for key '{k}' in pickle: expected {total_rows}, got {v.shape[0]}")

                logging.info(f"|    |    ✔ Pickle merged: {os.path.basename(pickle_file)} (keys: {list(aux_dict.keys())})")
            except Exception as e:
                logging.error(f"|    |    ❌ Failed to load or merge pickle '{pickle_file}': {e}")
        else:
            logging.warning(f"|    |    ⚠ No pickle file provided or file does not exist: {pickle_file}")

    logging.info("├─── ✅ Finished loading all JPAS datasets.")
    return data


def load_DESI_mocks_Raul(
    data: Dict[str, Any],
    root_path: str,
    datasets: List[Dict[str, object]],
    random_seed: int = None
) -> Dict[str, Any]:
    """
    Loads DESI datasets into a structured dictionary
    """
    
    def _load_DESI_mocks_Raul_splits(
        data: Dict,
        root_path: str,
        datasets: List[Dict[str, object]],
        random_seed: int = None
    ) -> Dict:
        """
        Loads DESI datasets into a structured dictionary with optional sampling.

        Parameters:
        - data (dict): Dictionary to populate.
        - root_path (str): Path to data files.
        - datasets (list of dict): Each dict contains:
            - 'name' (str)
            - 'npy' (str)
            - 'csv' (str)
            - 'sample_percentage' (float)
        - random_seed (int): Optional seed for reproducible sampling.

        Returns:
        - dict: Updated `data` dictionary.
        """
        logging.info("├─── 📥 Starting DESI dataset loading...")
        rng = np.random.default_rng(seed=random_seed)

        for dset in datasets:
            name = dset["name"]
            csv_file = os.path.join(root_path, dset["csv"])
            npy_file = os.path.join(root_path, dset["npy"])
            pct = float(dset["sample_percentage"])

            logging.info(f"|    ├─── 🔹 Dataset: {name}")

            try:
                df_full = pd.read_csv(csv_file)
                total_rows = len(df_full)
                logging.info(f"|    |    ✔ CSV loaded ({df_full.shape}), Size: {df_full.memory_usage(deep=True).sum()*1e-6:.2f} MB")
            except Exception as e:
                logging.error(f"|    |    ❌ CSV load failed for {csv_file}: {e}")
                continue

            try:
                mmap_data = np.load(npy_file, mmap_mode='r')
                logging.info(f"|    |    ✔ NPY loaded ({mmap_data.shape}), Size: {mmap_data.nbytes*1e-6:.2f} MB")
            except Exception as e:
                logging.error(f"|    |    ❌ NPY load failed for {npy_file}: {e}")
                continue

            if pct < 1.0:
                n_sample = int(pct * total_rows)
                selected_idxs = np.sort(rng.choice(total_rows, size=n_sample, replace=False))
                logging.info(f"|    |    📉 Sampling {n_sample}/{total_rows} rows ({pct:.0%})")
            else:
                selected_idxs = np.arange(total_rows)

            # Load and process sampled CSV
            df_sampled = df_full.iloc[selected_idxs]
            data[f"{name}_pd"] = {}
            for col in df_sampled.columns:
                if np.issubdtype(df_sampled[col].dtype, np.integer):
                    data[f"{name}_pd"][col] = df_sampled[col].astype(np.int64).to_numpy()
                elif np.issubdtype(df_sampled[col].dtype, np.floating):
                    data[f"{name}_pd"][col] = df_sampled[col].astype(np.float64).to_numpy()
                else:
                    data[f"{name}_pd"][col] = df_sampled[col].astype(str).tolist()

            # Load and process sampled NPY
            data[f"{name}_np"] = mmap_data[selected_idxs]
            data[f"{name}_np"][..., 2] *= np.abs(data[f"{name}_np"][..., 0])

            logging.info(f"|    |    ✔ Sample loaded. Final shape: {data[f'{name}_np'].shape}")

        logging.info("├─── ✅ Finished loading all DESI datasets.")
        return data


    # Load DESI datasets (split format)
    logging.info("├─── 📥 Loading DESI datasets (splitted)...")
    desi_split = _load_DESI_mocks_Raul_splits({}, root_path, datasets, random_seed=random_seed)


    def _concatenate_DESI_mocks_Raul_splits(
        desi_split: Dict[str, Any],
        merged_pd_key: str = "all_pd",
        merged_np_key: str = "all_np"
    ) -> Dict[str, Any]:
        """
        Concatenates *_pd and *_np entries in a DESI split dictionary into merged keys.

        Parameters:
        - desi_split (dict): Dictionary with keys like 'train_pd', 'val_np', etc.
        - merged_pd_key (str): Key under which to store concatenated pd-like data.
        - merged_np_key (str): Key under which to store concatenated np arrays.

        Returns:
        - dict: Dictionary containing merged keys with concatenated data.
        """

        logging.info("├─── 🔄 Concatenating DESI dataset splits...")

        # Identify base split names from *_pd keys
        pd_keys = [k for k in desi_split if k.endswith('_pd')]
        base_names = [k[:-3] for k in pd_keys]
        logging.info(f"|    |    Identified split names: {base_names}")

        # Ensure *_np exists for each *_pd
        for name in base_names:
            np_key = f"{name}_np"
            if np_key not in desi_split:
                raise KeyError(f"|    |    ❌ Missing expected key '{np_key}' for base '{name}'")

        merged = {merged_pd_key: {}}

        # Reference structure from the first pd split
        first_pd = desi_split[f"{base_names[0]}_pd"]
        pd_keys = list(first_pd.keys())

        for key in pd_keys:
            entries = []
            for name in base_names:
                value = desi_split[f"{name}_pd"].get(key)
                if value is None:
                    raise KeyError(f"|    |    ❌ Key '{key}' missing in split '{name}_pd'")
                entries.append(value)

            # Type check consistency
            first_type = type(entries[0])
            if not all(isinstance(e, first_type) for e in entries):
                raise TypeError(f"|    |    ❌ Inconsistent types for field '{key}': {[type(e) for e in entries]}")

            # Concatenate
            if isinstance(entries[0], np.ndarray):
                merged[merged_pd_key][key] = np.concatenate(entries, axis=0)
            elif isinstance(entries[0], list):
                merged[merged_pd_key][key] = sum(entries, [])
            else:
                raise TypeError(f"|    |    ❌ Unsupported type '{first_type}' for key '{key}'")

            logging.debug(f"|    |    ✅ Merged field '{key}'")

        # Concatenate *_np arrays
        try:
            np_arrays = [desi_split[f"{name}_np"] for name in base_names]
            merged[merged_np_key] = np.concatenate(np_arrays, axis=0)
            logging.info(f"|    |    Merged NPY arrays into '{merged_np_key}' with shape {merged[merged_np_key].shape}")
        except Exception as e:
            logging.warning(f"|    |    ⚠️ Skipping '{merged_np_key}' due to: {e}")

        logging.info("├─── ✅ DESI split concatenation complete.")
        return merged


    # Concatenate DESI splits into unified arrays
    logging.info("├─── 🔗 Concatenating DESI datasets...")
    out = _concatenate_DESI_mocks_Raul_splits(desi_split)  # expects keys: 'all_pd', 'all_np'

    # Validate expected keys
    if "all_pd" not in out or "all_np" not in out:
        raise KeyError("Expected keys 'all_pd' and 'all_np' in concatenated DESI data.")

    all_np = np.asarray(out["all_np"])
    if all_np.ndim != 3 or all_np.shape[2] < 3:
        raise ValueError(f"'all_np' must have shape (N_objects, N_filters, 3). Got {all_np.shape}.")

    # Slice the packed array:
    #   0 -> mean mock fluxes
    #   1 -> mock observations
    #   2 -> mock errors
    mean_fluxes   = np.asarray(all_np[:, :, 0], dtype=np.float64)
    observations  = np.asarray(all_np[:, :, 1], dtype=np.float64)
    errors        = np.asarray(all_np[:, :, 2], dtype=np.float64)

    # Update all_pd with the new mean fluxes entry
    all_pd = dict(out["all_pd"])  # shallow copy to avoid aliasing
    all_pd["mean_mock_observations_from_DESI_Raul"] = mean_fluxes

    # Assemble standardized return dict
    out["all_pd"] = all_pd
    out["all_observations"] = observations
    out["all_errors"] = errors

    # Remove the packed array key to match the common interface
    out.pop("all_np", None)

    logging.info(
        f"├─── ✅ DESI mocks (Raul) ready: "
        f"all_observations {observations.shape}, all_errors {errors.shape}, "
        f"all_pd keys: {len(all_pd)}"
    )

    return out


def _load_JPAS_x_DESI_Ignasi(
    data: Dict[str, Any],
    root_path: str,
    fits_filename: str,
    *,
    name: str = "all",
) -> Dict[str, Any]:
    """
    Minimal JPAS-Ignasi FITS loader that mimics the '..._pd' output style:

    - int/uint/float -> NumPy arrays (int64/float64), byteswapped if needed
    - bool (native FITS '?') -> NumPy bool array
    - 'U'/'S' columns:
        * if values look like booleans ("true/false/1/0/t/f/yes/no"), convert to NumPy bool array
        * otherwise -> list[str] (matching original CSV loader's 'else' branch)
    """
    logging.info("├─── 📥 Starting JPAS-Ignasi FITS loading...")

    fpath = os.path.join(root_path, fits_filename)
    try:
        rec = fitsio.read(fpath)  # structured ndarray
        total_rows = rec.shape[0]
        logging.info(f"|    ├─── 🔹 FITS loaded: {os.path.basename(fpath)} (rows: {total_rows})")
    except Exception as e:
        logging.error(f"|    └─── ❌ Failed to read FITS '{fits_filename}': {e}")
        return data

    pd_key = f"{name}_pd"
    data[pd_key] = {}

    # helper: try to convert a string/bytes column into booleans if all non-empty tokens are recognized
    def _try_string_bools_to_bool_array(str_arr: np.ndarray) -> np.ndarray:
        s = np.char.strip(str_arr.astype(str))
        low = np.char.lower(s)
        # recognized tokens (case-insensitive)
        true_tokens  = ("true", "1", "t", "y", "yes")
        false_tokens = ("false", "0", "f", "n", "no")
        known = true_tokens + false_tokens

        nonempty = s != ""
        if np.all(np.isin(low[nonempty], known)):
            # empties map to False by default
            bools = (
                (low == "true") | (low == "1") | (low == "t") | (low == "y") | (low == "yes")
            )
            return bools.astype(bool)
        # signal: do not convert
        return None

    for col in rec.dtype.names:
        try:
            arr = rec[col]
            kind = arr.dtype.kind  # 'i','u','f','U','S','b','O', etc.

            # Numeric → byteswap only numerics; then cast to canonical dtypes
            if kind in ("i", "u", "f"):
                if arr.dtype.byteorder in (">", "<"):
                    arr = arr.byteswap().newbyteorder()
                if kind in ("i", "u"):
                    data[pd_key][col] = arr.astype(np.int64, copy=False)
                else:
                    data[pd_key][col] = arr.astype(np.float64, copy=False)
                continue

            # Native boolean ('?') → NumPy bool
            if kind == "b":
                data[pd_key][col] = arr.astype(bool, copy=False)
                continue

            # Unicode strings ('U'): try boolean conversion first; otherwise list[str]
            if kind == "U":
                maybe_bool = _try_string_bools_to_bool_array(arr)
                if isinstance(maybe_bool, np.ndarray):
                    data[pd_key][col] = maybe_bool
                else:
                    data[pd_key][col] = np.char.strip(arr.astype(str)).tolist()
                continue

            # Byte strings ('S'): decode → try boolean conversion; otherwise list[str]
            if kind == "S":
                decoded = np.char.decode(arr, "utf-8", errors="ignore")
                maybe_bool = _try_string_bools_to_bool_array(decoded)
                if isinstance(maybe_bool, np.ndarray):
                    data[pd_key][col] = maybe_bool
                else:
                    data[pd_key][col] = np.char.strip(decoded.astype(str)).tolist()
                continue

            # Fallback (object/other): keep stringified list to mirror original behavior
            data[pd_key][col] = [
                (x.decode("utf-8", errors="ignore").strip() if isinstance(x, (bytes, bytearray))
                 else str(x).strip())
                for x in arr
            ]

        except Exception as e:
            logging.error(f"|    |    ❌ Failed to convert column '{col}': {e}")

    logging.info(f"├─── ✅ Finished loading JPAS-Ignasi dataset into data['{pd_key}'].")
    return data


def _load_JPAS_IDR_Ignasi(
    data: Dict[str, Any],
    root_path: str,
    fits_filename: str,
    *,
    dataset_key: str = "all",
    n_filters: int = 57,
    flux_prefix: str = "f",
    err_prefix: str = "err_f",
) -> Dict[str, Any]:
    """
    Load a JPAS IDR (Ignasi) FITS table and populate:
        data[name][f"{dataset_key}_pd"]          <- dict (via _fits_struct_to_dict)
        data[name][f"{dataset_key}_observations"]<- (N_rows, n_filters) float64
        data[name][f"{dataset_key}_errors"]      <- (N_rows, n_filters) float64

    Notes:
      - Relies on an already-defined `_fits_struct_to_dict(recarray)`.
      - Expects flux columns named f0..f{n_filters-1} and error columns err_f0..err_f{n_filters-1}.
      - Raises a clear error if any required column is missing.
    """

    def _fits_struct_to_dict(recarray):
        """
        Convert a NumPy structured array (e.g., from fitsio.read)
        into a dict: numeric fields -> np.ndarray; string/bytes -> list[str].
        """
        out = {}
        for name in recarray.dtype.names:
            col = recarray[name]
            kind = col.dtype.kind  # 'i' int, 'u' uint, 'f' float, 'S' bytes, 'U' unicode, 'O' object

            if kind in ('i', 'u', 'f'):
                # Already numeric; ensure a contiguous array
                out[name] = np.asarray(col)
            elif kind == 'U':
                # Unicode already; make a list[str]
                out[name] = col.astype(str).tolist()
            elif kind == 'S':
                # FITS often stores fixed-length bytes; decode to str
                # astype('U') converts bytes to unicode safely
                out[name] = col.astype('U').tolist()
            else:
                # Fallback: try to coerce to str list
                try:
                    out[name] = col.astype('U').tolist()
                except Exception:
                    out[name] = [str(x) for x in col]
        return out


    logging.info("├─── 📥 Starting JPAS-IDR-Ignasi FITS loading...")

    fpath = os.path.join(root_path, fits_filename)
    try:
        rec = fitsio.read(fpath)
        n_rows = rec.shape[0]
        logging.info(f"|    ├─── 🔹 FITS loaded: {os.path.basename(fpath)} (rows: {n_rows})")
    except Exception as e:
        logging.error(f"|    └─── ❌ Failed to read FITS '{fits_filename}': {e}")
        return data

    # Convert to pd-like dict using your helper
    try:
        tmp_dict = _fits_struct_to_dict(rec)  # <- uses your previously defined function
        logging.info(f"|    |    ✔ Converted FITS to dict with {len(tmp_dict)} columns")
    except Exception as e:
        logging.error(f"|    |    ❌ _fits_struct_to_dict failed: {e}")
        return data

    # Store pd-like dict
    pd_key = f"{dataset_key}_pd"
    data[pd_key] = tmp_dict

    # Build observations/errors matrices
    flux_cols = [f"{flux_prefix}{i}" for i in range(n_filters)]
    err_cols  = [f"{err_prefix}{i}" for i in range(n_filters)]

    missing_flux = [c for c in flux_cols if c not in tmp_dict]
    missing_err  = [c for c in err_cols  if c not in tmp_dict]

    if missing_flux:
        raise KeyError(f"Missing flux columns in FITS dict: {missing_flux[:5]}{' ...' if len(missing_flux)>5 else ''}")
    if missing_err:
        raise KeyError(f"Missing error columns in FITS dict: {missing_err[:5]}{' ...' if len(missing_err)>5 else ''}")

    try:
        obs = np.vstack([np.asarray(tmp_dict[c], dtype=np.float64) for c in flux_cols]).T
        err = np.vstack([np.asarray(tmp_dict[c], dtype=np.float64) for c in err_cols]).T
        if obs.shape != (n_rows, n_filters) or err.shape != (n_rows, n_filters):
            raise ValueError(f"Unexpected shapes: obs {obs.shape}, err {err.shape}, expected ({n_rows}, {n_filters})")
        data[f"{dataset_key}_observations"] = obs
        data[f"{dataset_key}_errors"] = err
        logging.info(f"|    |    ✔ Built observations/errors with shape {obs.shape}")
    except Exception as e:
        logging.error(f"|    |    ❌ Failed to build observations/errors: {e}")
        return data

    # --- NEW: prune flux/error columns from all_pd after saving them above ---
    try:
        remove_keys = [k for k in list(data[pd_key].keys())
                       if k.startswith(flux_prefix) or k.startswith(err_prefix)]
        for k in remove_keys:
            data[pd_key].pop(k, None)
        logging.info(f"|    |    🧹 Removed {len(remove_keys)} flux/error columns from data['{pd_key}']")
    except Exception as e:
        logging.error(f"|    |    ❌ Failed to prune flux/error columns from '{pd_key}': {e}")

    logging.info(f"├─── ✅ Finished loading JPAS-IDR-Ignasi.")
    return data


def _infer_len_from_pd(pd: Dict[str, Any]) -> int:
    for v in pd.values():
        if isinstance(v, np.ndarray):
            return v.shape[0]
        return len(v)
    raise ValueError("Empty pd dict; cannot infer length.")

def _col_meta(x: Any) -> str:
    if isinstance(x, np.ndarray):
        return f"ndarray[{x.dtype.name}] shape={x.shape}"
    if isinstance(x, list):
        return f"list[{type(x[0]).__name__ if x else '∅'}] len={len(x)}"
    return f"{type(x).__name__}"

def _as_arraylike(x: Any) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else np.asarray(x, dtype=object)

def _compare_columns(a: Any, b: Any, *, preview_limit: int = 5) -> Tuple[bool, int, List[int], List[Tuple[Any, Any]]]:
    """
    Exact-equality check (NaNs count as equal for numeric values).
    Returns:
      equal_flag,
      mismatch_count,
      example_indices (up to preview_limit),
      example_pairs   ([(a[i], b[i]), ...])
    """
    aa = _as_arraylike(a)
    bb = _as_arraylike(b)

    # shape mismatch → immediate diff
    if aa.shape != bb.shape:
        return False, -1, [], []

    # elementwise equality
    if aa.dtype.kind in "iufb" or bb.dtype.kind in "iufb":
        # numeric path; treat NaN==NaN
        with np.errstate(invalid="ignore"):
            eq = (aa == bb)
            both_nan = np.isnan(aa) & np.isnan(bb)
            eq = np.where(both_nan, True, eq)
    else:
        # non-numeric (strings/objects); plain equality + treat float-NaNs equal when present
        eq = (aa == bb)
        # for object arrays, guard: NaN comparisons
        def _isnan(v):
            try:
                return bool(np.isnan(v))
            except Exception:
                return False
        if aa.dtype == object or bb.dtype == object:
            # vectorize isnan on objects
            v_isnan = np.vectorize(_isnan)
            both_nan = v_isnan(aa) & v_isnan(bb)
            eq = np.where(both_nan, True, eq)

    if np.all(eq):
        return True, 0, [], []

    # collect up to preview_limit mismatches (flatten for 1D columns)
    idxs = np.flatnonzero(~eq)[:preview_limit]
    pairs = [(aa[i], bb[i]) for i in idxs]
    return False, int((~eq).sum()), idxs.tolist(), pairs

def merge_all_pd_from_blocks(
    block_a: Dict[str, Any],
    block_b: Dict[str, Any],
    *,
    name_a: str = "JPAS_x_DESI_Ignasi",
    name_b: str = "JPAS_IDR_Ignasi",
    on_conflict: str = "error",         # 'error' | 'prefer_a' | 'prefer_b' | 'suffix'
    suffixes: Tuple[str, str] = ("_A", "_B"),
    log_common_limit: int = 5,         # how many repeated keys to list explicitly
    log_preview_limit: int = 5,         # how many mismatches to preview
) -> Dict[str, Any]:
    """
    Merge two `all_pd` dictionaries with identical row counts.
    - Logs repeated keys and verifies each repeated key (dtype/shape + equality).
    - Keeps non-common keys from both.
    - For repeated keys:
        * if identical → keep one
        * if different → resolve per `on_conflict` (default: raise).
    """
    pd_a = block_a.get("all_pd", {})
    pd_b = block_b.get("all_pd", {})
    if not pd_a or not pd_b:
        raise ValueError("Both blocks must have a non-empty 'all_pd'.")

    # Validate row counts
    len_a = _infer_len_from_pd(pd_a)
    len_b = _infer_len_from_pd(pd_b)
    if len_a != len_b:
        raise ValueError(f"Row count mismatch: {name_a}={len_a}, {name_b}={len_b}")

    keys_a = set(pd_a.keys())
    keys_b = set(pd_b.keys())
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    logging.info(f"├── Merging all_pd: {name_a} + {name_b}  (rows={len_a})")
    logging.info(f"|   ├── only in {name_a}: {len(only_a)}"
                 + (f" → {only_a[:log_common_limit]}{' ...' if len(only_a)>log_common_limit else ''}" if only_a else ""))
    logging.info(f"|   ├── only in {name_b}: {len(only_b)}"
                 + (f" → {only_b[:log_common_limit]}{' ...' if len(only_b)>log_common_limit else ''}" if only_b else ""))
    logging.info(f"|   ├── repeated keys: {len(common)}"
                 + (f" → {common[:log_common_limit]}{' ...' if len(common)>log_common_limit else ''}" if common else ""))

    merged: Dict[str, Any] = {}

    # copy unique keys first
    for k in only_a:
        merged[k] = pd_a[k]
    for k in only_b:
        merged[k] = pd_b[k]

    differing = []
    identical = []

    # verify common keys
    for k in common:
        a_col = pd_a[k]
        b_col = pd_b[k]
        equal, mismatch_count, idxs, pairs = _compare_columns(a_col, b_col, preview_limit=log_preview_limit)

        meta_a = _col_meta(a_col)
        meta_b = _col_meta(b_col)

        if equal:
            logging.info(f"|   |   ✓ '{k}' identical  [{meta_a}] vs [{meta_b}]")
            merged[k] = a_col
            identical.append(k)
        else:
            differing.append(k)
            # detailed diff preview
            if mismatch_count == -1:
                logging.warning(f"|   |   ✗ '{k}' differs: shape mismatch [{meta_a}] vs [{meta_b}]")
            else:
                preview = ", ".join([f"i={i}: {repr(pa)} ≠ {repr(pb)}" for i,(pa,pb) in zip(idxs, pairs)])
                logging.warning(f"|   |   ✗ '{k}' differs: {mismatch_count} element(s) mismatch "
                                f"({preview}{'...' if mismatch_count > len(idxs) else ''})")

            # conflict resolution
            if on_conflict == "error":
                # Note: we continue checking other keys to log all diffs; raise after loop
                continue
            elif on_conflict == "prefer_a":
                logging.warning(f"|   |     keeping {name_a}'s '{k}'")
                merged[k] = a_col
            elif on_conflict == "prefer_b":
                logging.warning(f"|   |     keeping {name_b}'s '{k}'")
                merged[k] = b_col
            elif on_conflict == "suffix":
                ka, kb = f"{k}{suffixes[0]}", f"{k}{suffixes[1]}"
                logging.warning(f"|   |     storing as '{ka}' and '{kb}'")
                merged[ka] = a_col
                merged[kb] = b_col
            else:
                raise ValueError(f"Unknown on_conflict='{on_conflict}'")

    if on_conflict == "error" and differing:
        preview = ", ".join(differing[:min(10, len(differing))])
        more = "" if len(differing) <= 10 else f" ... (+{len(differing)-10} more)"
        raise ValueError(
            f"Found {len(differing)} repeated key(s) with differing contents between "
            f"{name_a} and {name_b}: {preview}{more}. "
            f"Use on_conflict='prefer_a'|'prefer_b'|'suffix' to resolve."
        )

    logging.info(f"|   ├── identical repeated keys kept: {len(identical)}")
    logging.info(f"|   └── differing repeated keys handled: {len(differing)}")
    logging.info(f"└── Merge complete. Final columns: {len(merged)}")

    return merged

def load_Ignasi_merged(
    data: Dict[str, Any],
    root_path: str,
    *,
    datasets: Union[Dict[str, str], Tuple[str, str], List[str]],
    random_seed: int = None,              # kept for signature parity; not used
    on_conflict: str = "error",           # 'error' | 'prefer_a' | 'prefer_b' | 'suffix'
    suffixes: Tuple[str, str] = ("_A", "_B"),
) -> Dict[str, Any]:
    """
    Load the two Ignasi datasets and return a merged block:
      - loads JPAS_x_DESI_Ignasi (pd-only) and JPAS_IDR_Ignasi (pd + obs/err)
      - merges their all_pd (verifies repeated keys; see on_conflict)
      - keeps all_observations / all_errors from JPAS_IDR_Ignasi (if present)

    Parameters
    ----------
    data : dict
        Ignored for content (returned block is independent), kept for call symmetry.
    root_path : str
        Base path for files.
    datasets : dict | tuple/list
        Either:
          - dict with keys {"fits_x_desi", "fits_idr"} (preferred), or
          - (fits_x_desi, fits_idr)
    random_seed : int, optional
        Unused (kept for API symmetry with other loaders).
    on_conflict : str
        Passed to merge_all_pd_from_blocks.
    suffixes : (str, str)
        Suffixes used if on_conflict='suffix'.

    Returns
    -------
    block : dict
        {"all_pd": merged_pd, "all_observations": ..., "all_errors": ...} (when available)
    """
    # ---- parse inputs
    if isinstance(datasets, dict):
        fits_x_desi = datasets.get("fits_x_desi") or datasets.get("JPAS_x_DESI_Ignasi")
        fits_idr    = datasets.get("fits_idr")    or datasets.get("JPAS_IDR_Ignasi")
    else:
        if len(datasets) != 2:
            raise ValueError("`datasets` must be a dict with keys "
                             "('fits_x_desi','fits_idr') or a 2-tuple/list (x_desi, idr).")
        fits_x_desi, fits_idr = datasets

    if not fits_x_desi or not fits_idr:
        raise ValueError("Both FITS filenames must be provided: fits_x_desi and fits_idr.")

    logging.info("📥 Loading Ignasi sources for merged PD...")
    logging.info("├ Loading JPAS_x_DESI_Ignasi (pd-only)...")
    block_x = _load_JPAS_x_DESI_Ignasi({}, root_path=root_path, fits_filename=fits_x_desi)

    logging.info("├ Loading JPAS_IDR_Ignasi (pd + obs/err)...")
    block_idr = _load_JPAS_IDR_Ignasi({}, root_path=root_path, fits_filename=fits_idr)

    # ---- merge PDs
    logging.info("├ Merging all_pd from both Ignasi sources...")
    merged_pd = merge_all_pd_from_blocks(
        block_x, block_idr,
        name_a="JPAS_x_DESI_Ignasi",
        name_b="JPAS_IDR_Ignasi",
        on_conflict=on_conflict,
        suffixes=suffixes,
    )

    # ---- assemble output block
    out = dict(block_idr)  # keep obs/err and any extras from IDR
    out["all_pd"] = merged_pd

    logging.info("✅ Ignasi merged block ready.")
    return out


def load_data_bundle(
    root_path: str,
    *,
    include: Optional[List[str]] = None,   # e.g. ["JPAS_x_DESI_Raul", "Ignasi"], or None
    JPAS_x_DESI_Raul: Optional[Dict[str, Any]] = None,  # {"datasets": ..., "random_seed": ...}
    DESI_mocks_Raul: Optional[Dict[str, Any]] = None,   # {"datasets": ..., "random_seed": ...}
    Ignasi: Optional[Dict[str, Any]] = None,            # {"datasets": {...}} for load_Ignasi_merged
    random_seed: Optional[int] = None,                  # optional default seed for Raul loaders
    continue_on_error: bool = True,                     # continue even if one loader fails
) -> Dict[str, Any]:
    """
    Load any subset (or all) of the available datasets into a single dict:
      - "JPAS_x_DESI_Raul"  -> load_JPAS_x_DESI_Raul(...)
      - "DESI_mocks_Raul"   -> load_DESI_mocks_Raul(...)
      - "Ignasi"            -> load_Ignasi_merged(...)

    Parameters
    ----------
    root_path : str
        Base path for files.
    include : list[str] or None
        Which datasets to load. If None, load only those for which kwargs are provided.
        Valid names: ["JPAS_x_DESI_Raul", "DESI_mocks_Raul", "Ignasi"].
    JPAS_x_DESI_Raul, DESI_mocks_Raul, Ignasi : dict or None
        Per-dataset kwargs passed straight to the underlying loader.
        - Raul loaders require {"datasets": ...}; "random_seed" is optional.
        - Ignasi requires {"datasets": ...} for load_Ignasi_merged (fits filenames).
    random_seed : int or None
        If provided, used as a default for Raul loaders when their kwargs omit it.
    continue_on_error : bool
        If True, logs and skips a failing dataset; if False, re-raises the exception.

    Returns
    -------
    DATA : dict
        Dict with loaded blocks keyed by dataset name.
    """
    # Registry of loaders
    registry = {
        "JPAS_x_DESI_Raul": {
            "fn": load_JPAS_x_DESI_Raul,
            "kwargs": JPAS_x_DESI_Raul or {},
            "requires": {"datasets"},
            "accepts_seed": True,
        },
        "DESI_mocks_Raul": {
            "fn": load_DESI_mocks_Raul,
            "kwargs": DESI_mocks_Raul or {},
            "requires": {"datasets"},
            "accepts_seed": True,
        },
        "Ignasi": {
            "fn": load_Ignasi_merged,
            "kwargs": Ignasi or {},
            "requires": {"datasets"},
            "accepts_seed": False,
        },
    }

    # Decide which ones to load
    if include is None:
        to_load = [k for k, meta in registry.items() if meta["kwargs"]]
    else:
        # Validate names; ignore unknown with a warning
        valid = set(registry.keys())
        to_load = []
        for name in include:
            if name in valid:
                to_load.append(name)
            else:
                logging.warning(f"[load_data_bundle] Unknown dataset '{name}' — skipping.")

    DATA: Dict[str, Any] = {}
    logging.info("📥 Starting modular dataset loading (load_data_bundle)")

    for name in to_load:
        meta = registry[name]
        fn = meta["fn"]
        kwargs = dict(meta["kwargs"])  # shallow copy

        # Fill default random_seed for Raul loaders if not provided
        if meta["accepts_seed"] and (random_seed is not None) and ("random_seed" not in kwargs):
            kwargs["random_seed"] = random_seed

        # Check required keys
        missing = [k for k in meta["requires"] if k not in kwargs]
        if missing:
            logging.error(f"├── ❌ '{name}': missing required kwarg(s) {missing} — skipping.")
            if not continue_on_error:
                raise ValueError(f"{name}: missing required kwarg(s) {missing}")
            continue

        try:
            logging.info(f"├── Loading {name} ...")
            DATA[name] = fn({}, root_path=root_path, **kwargs)
            logging.info(f"│   ✔ Loaded {name}")
        except Exception as e:
            logging.exception(f"│   ❌ Failed to load {name}: {e}")
            if not continue_on_error:
                raise

    logging.info("✅ Finished modular dataset loading.")
    return DATA
