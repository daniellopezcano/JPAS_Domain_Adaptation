import logging
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from JPAS_DA.data import loading_tools
from JPAS_DA.data import cleaning_tools
from JPAS_DA.data import crossmatch_tools
from JPAS_DA.data import process_dset_splits
from JPAS_DA.data import data_loaders


def wrapper_build_dataloaders(
    *,
    # --- Loading ---
    root_path: str,
    include: List[str],
    dataset_params: Dict[str, Dict[str, Any]],
    random_seed_load: int,

    # --- Cleaning ---
    cleaning_config: Dict[str, Any],

    # --- Crossmatch & splits ---
    crossmatch_pair: Tuple[str, str] = ("DESI_mocks_Raul", "JPAS_x_DESI_Raul"),
    id_key: str = "TARGETID",
    split_config: Dict[str, Any],

    # --- Features & labels ---
    keys_xx: List[str],
    keys_yy: List[str],

    # --- Output control ---
    return_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Build dataloaders with the new pipeline:
      load_data_bundle -> clean_data_pipeline -> crossmatch -> split -> extract & build DataLoader

    Parameters
    ----------
    root_path : str
        Base data path.
    include : list[str]
        Dataset names to load (e.g., ["JPAS_x_DESI_Raul", "DESI_mocks_Raul"]).
    dataset_params : dict[str, dict]
        Per-dataset kwargs forwarded to `loading_tools.load_data_bundle`, e.g.:
            {
              "JPAS_x_DESI_Raul": {"datasets": load_JPAS_x_DESI_Raul},
              "DESI_mocks_Raul":  {"datasets": load_DESI_mocks_Raul},
              "Ignasi":           {"datasets": load_Ignasi},
            }
    random_seed_load : int
        Seed for loaders that sample when loading.
    cleaning_config : dict
        Config for `cleaning_tools.clean_data_pipeline`.
    crossmatch_pair : (str, str)
        Datasets to crossmatch by `id_key`. First is treated as “outersection”,
        second as “intersection”, matching your current usage.
    id_key : str
        ID key inside each block’s all_pd (default "TARGETID").
    split_config : dict
        Must contain:
            train_ratio_intersection, val_ratio_intersection, test_ratio_intersection,
            random_seed_split_intersection,
            train_ratio_outersection,  val_ratio_outersection,  test_ratio_outersection,
            random_seed_split_outersection
    keys_xx, keys_yy : list[str]
        Feature/label keys expected to exist in the block (top-level or block['all_pd']).
    return_artifacts : bool
        If True, also returns DATA, Dict_LoA, Dict_LoA_split.

    Returns
    -------
    dict
        {
          "dataloaders": {
             "train": { <dsetA>: DataLoader, <dsetB>: DataLoader },
             "val":   { ... },
             "test":  { ... },
          },
          # If return_artifacts=True:
          "DATA": <cleaned DATA>,
          "Dict_LoA": <intersection/outersection lists>,
          "Dict_LoA_split": <train/val/test splits of those lists>,
        }
    """
    A, B = crossmatch_pair  # e.g., ("DESI_mocks_Raul", "JPAS_x_DESI_Raul")

    # 1) Load
    logging.info("📦 Loading datasets with load_data_bundle()")
    DATA = loading_tools.load_data_bundle(
        root_path=root_path,
        include=include,
        random_seed=random_seed_load,
        **dataset_params,  # forwards JPAS_x_DESI_Raul={"datasets":...}, etc.
    )

    # 2) Clean
    logging.info("🧹 Cleaning datasets with clean_data_pipeline()")
    DATA = cleaning_tools.clean_data_pipeline(DATA, config=cleaning_config, in_place=True)

    # Validate presence of crossmatch datasets + IDs
    for ds in (A, B):
        if ds not in DATA:
            raise KeyError(f"[wrapper] Dataset '{ds}' not found in DATA (available: {list(DATA.keys())}).")
        if "all_pd" not in DATA[ds] or id_key not in DATA[ds]["all_pd"]:
            raise KeyError(f"[wrapper] DATA['{ds}']['all_pd']['{id_key}'] is required for crossmatch.")

    # 3) Crossmatch
    logging.info(f"🔗 Crossmatching '{A}' vs '{B}' on all_pd['{id_key}']")
    Dict_LoA = {"intersection": {}, "outersection": {}}
    _, _, _, \
    Dict_LoA["outersection"][A], Dict_LoA["outersection"][B], \
    Dict_LoA["intersection"][A], Dict_LoA["intersection"][B] = crossmatch_tools.crossmatch_IDs_two_datasets(
        DATA[A]["all_pd"][id_key], DATA[B]["all_pd"][id_key]
    )

    # 4) Split LoAs
    logging.info("✂️ Splitting LoAs into train/val/test")
    req_keys = [
        "train_ratio_intersection", "val_ratio_intersection", "test_ratio_intersection",
        "random_seed_split_intersection",
        "train_ratio_outersection", "val_ratio_outersection", "test_ratio_outersection",
        "random_seed_split_outersection",
    ]
    missing = [k for k in req_keys if k not in split_config]
    if missing:
        raise KeyError(f"[wrapper] split_config missing keys: {missing}")

    Dict_LoA_split = {"intersection": {}, "outersection": {}}

    # intersection → for B (e.g., JPAS_x_DESI_Raul)
    Dict_LoA_split["intersection"][B] = process_dset_splits.split_LoA(
        Dict_LoA["intersection"][B],
        train_ratio=split_config["train_ratio_intersection"],
        val_ratio=split_config["val_ratio_intersection"],
        test_ratio=split_config["test_ratio_intersection"],
        seed=split_config["random_seed_split_intersection"],
    )
    # outersection → for A (e.g., DESI_mocks_Raul)
    Dict_LoA_split["outersection"][A] = process_dset_splits.split_LoA(
        Dict_LoA["outersection"][A],
        train_ratio=split_config["train_ratio_outersection"],
        val_ratio=split_config["val_ratio_outersection"],
        test_ratio=split_config["test_ratio_outersection"],
        seed=split_config["random_seed_split_outersection"],
    )

    # 5) Build DataLoaders  (dataset-first structure)
    logging.info("🧰 Building DataLoader objects (dataset-first)")

    # Which dataset pulls from which crossmatch bucket
    extract_dsets = [
        (A, "outersection"),   # e.g., ("DESI_mocks_Raul", "outersection")
        (B, "intersection"),   # e.g., ("JPAS_x_DESI_Raul", "intersection")
    ]

    dset_loaders: Dict[str, Dict[str, data_loaders.DataLoader]] = {}

    for key_dset, key_xmatch in extract_dsets:
        dset_loaders.setdefault(key_dset, {})
        for split_name in ["train", "val", "test"]:
            LoA_groups = Dict_LoA_split[key_xmatch][key_dset].get(split_name, [])
            _, xx, yy = process_dset_splits.extract_from_block_by_LoA(
                block=DATA[key_dset],
                LoA=LoA_groups,
                keys_xx=keys_xx,
                keys_yy=keys_yy,
            )
            dset_loaders[key_dset][split_name] = data_loaders.DataLoader(xx, yy)

    result = {"dataloaders": dset_loaders}
    if return_artifacts:
        result.update({"DATA": DATA, "Dict_LoA": Dict_LoA, "Dict_LoA_split": Dict_LoA_split})
    logging.info("✅ Dataloader bundle ready.")
    return result
