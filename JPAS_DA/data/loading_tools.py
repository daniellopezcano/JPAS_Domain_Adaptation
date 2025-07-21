import os
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import logging
from typing import Dict, List, Any


import pickle

def load_JPAS_dsets(
    data: Dict[str, Any],
    root_path: str,
    datasets: List[Dict[str, Any]],
    random_seed: int = None
) -> Dict[str, Any]:
    """
    Loads JPAS datasets into a structured dictionary, with optional percentage-based sampling.

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
    logging.info("├─── 📥 Starting JPAS dataset loading...")
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



def load_JPAS_Ignasi_dsets(path):

    logging.info("├─── 📥 Starting JPAS-Ignasi dataframe loading...")

    with fits.open(Path(path)) as hdul:
        # hdul.info()
        data = hdul[1].data  # usar extensión 1

    # Convertir a ndarray con endianess correcto (compatible con NumPy 2.0+)
    data_array = np.array(data).byteswap().view(data.dtype.newbyteorder('='))

    # Convertir a DataFrame
    df = pd.DataFrame(data_array)

    logging.info("├─── ✅ Finished loading JPAS-Ignasi dataframe.")

    return df


def load_DESI_dsets(
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


def load_DESI_Lilianne_dsets(
    data: Dict,
    root_path: str,
    datasets: List[Dict[str, object]],
    random_seed: int = None,
    pd_keys: List[str] = None
) -> Dict:
    
    logging.info("├─── 📥 Starting DESI-Lilianne dataset loading...")

    for dset in datasets:
        name = dset["name"]
        csv_file = os.path.join(root_path, dset["csv"])
        pct = float(dset["sample_percentage"])

        logging.info(f"|    ├─── 🔹 Dataset: {name}")

        try:
            df_full = pd.read_csv(csv_file)
            total_rows = len(df_full)
            logging.info(f"|    |    ✔ CSV loaded ({df_full.shape}), Size: {df_full.memory_usage(deep=True).sum()*1e-6:.2f} MB")
        except Exception as e:
            logging.error(f"|    |    ❌ CSV load failed for {csv_file}: {e}")
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

    loaded_Lilianne_keys = list(data[datasets[0]["name"]+"_pd"].keys()) 
    for ii_dset in range(len(datasets)):
        dset_key = datasets[ii_dset]['name']
        data[f"{dset_key}_np"] = []
        for ii, key in enumerate(loaded_Lilianne_keys):
            if key not in pd_keys:
                tmp_pop = data[f"{dset_key}_pd"].pop(key)
                data[f"{dset_key}_np"].append(tmp_pop)
        data[f"{dset_key}_np"] = np.array(data[f"{dset_key}_np"]).T[..., None]

    logging.info("├─── ✅ Finished loading all DESI-Lilianne datasets.")

    return data


def concatenate_DESI_splits(
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


def load_dsets(
    root_path: str,
    datasets_jpas: List[Dict[str, str]],
    datasets_desi: List[Dict[str, Any]],
    random_seed: int = None
) -> Dict[str, Any]:
    """
    Loads JPAS and DESI datasets and structures them into a unified dictionary.

    Parameters:
    - root_path (str): Path to base data directory.
    - datasets_jpas (list of dict): Each dict must contain 'name', 'npy', 'csv'.
    - datasets_desi (list of dict): Each dict must contain 'name', 'npy', 'csv', 'sample_percentage'.
    - random_seed (int, optional): Seed for reproducibility in DESI sampling.

    Returns:
    - dict: A dictionary with keys:
        - 'JPAS': Loaded JPAS data
        - 'DESI_splitted': Per-split DESI data
        - 'DESI': Concatenated DESI data
    """

    logging.info("📥 Starting full dataset loading with `load_dsets()`")
    data = {}

    # Load JPAS datasets
    logging.info("├ Loading JPAS datasets...")
    data["JPAS"] = load_JPAS_dsets({}, root_path, datasets_jpas)

    # Load DESI datasets (split format)
    logging.info("├ Loading DESI datasets (splitted)...")
    desi_split = load_DESI_dsets({}, root_path, datasets_desi, random_seed=random_seed)
    data["DESI_splitted"] = desi_split

    # Concatenate DESI splits into unified arrays
    logging.info("├ Concatenating DESI datasets...")
    data["DESI"] = concatenate_DESI_splits(desi_split)

    logging.info("✅ Finished `load_dsets()`")
    return data

