# =============================================================================
# JPAS Domain Adaptation — Central Configuration
# -----------------------------------------------------------------------------
# Purpose:
#   Centralized variables for paths, dataset manifests, cleaning/splitting
#   options, feature/label selections, and class names used across the project.
# =============================================================================

# ----------------------------
# Standard library imports
# ----------------------------
import os

# ----------------------------
# Versioning
# ----------------------------
version = "0.1.0"

# ----------------------------
# Project paths
#   - main_path: project root
#   - DATA_path: data directory
#   - path_models: directory to save trained models
#   - path_configs: directory with config files
# ----------------------------
main_path = os.path.join("/home/dlopez/Documentos/0.profesional/Postdoc/USP/Projects/JPAS_Domain_Adaptation")
DATA_path = os.path.join(main_path, "DATA")
path_models = os.path.join(main_path, "SAVED_models")
path_configs = os.path.join(main_path, "configs")
path_saved_figures = os.path.join(main_path, "SAVED_FIGURES")

# ----------------------------
# Reproducibility
# ----------------------------
default_seed = 137

# ----------------------------
# JPAS dataset manifest (arrays + metadata)
#   Structure (per item):
#     - name: logical subset name
#     - npy: path to FLUX+NOISE numpy array
#     - csv: path to properties table
#     - pickle: auxiliary properties cache
#     - sample_percentage: fraction in (0, 1] to downsample
# ----------------------------
load_JPAS_x_DESI_Raul = [{
    "name": "all",
    "npy": "JPAS_DATA_Aper_Cor_3_FLUX+NOISE.npy",
    "csv": "JPAS_DATA_PROPERTIES.csv",
    "pickle": "JPAS_DATA_PROPERTIES_AUX.pkl",
    "sample_percentage": 1.0
}]

# ----------------------------
# DESI mock dataset manifest (train/val/test)
#   Structure (per split):
#     - name: "train" | "val" | "test"
#     - npy: FLUX+NOISE numpy array file
#     - csv: properties table
#     - sample_percentage: fraction in (0, 1]
# ----------------------------
load_DESI_mocks_Raul = [
{
    "name": "train",
    "npy": "Mock_train_FLUX+NOISE.npy",
    "csv": "Mock_train_PROPS.csv",
    "sample_percentage": 1.0
},
{
    "name": "val",
    "npy": "Mock_valid_FLUX+NOISE.npy",
    "csv": "Mock_valid_PROPS.csv",
    "sample_percentage": 1.0
},
{
    "name": "test",
    "npy": "Mock_test_FLUX+NOISE.npy",
    "csv": "Mock_test_PROPS.csv",
    "sample_percentage": 1.0
}
]

# ----------------------------
# JPAS dataset (Ignasi FITS sources)
# ----------------------------
load_Ignasi = {
    "fits_x_desi": "jpas_idr_classification_xmatch_desi_dr1.fits.gz",
    "fits_idr":    "jpas_mag_23_5_flag_0_40_filters.fits"
}

# ----------------------------
# Data cleaning options
# ----------------------------
config_dict_cleaning = {
    "mask_unreliable": {
        "enabled": True,
        "indices": [0, -2]
    },
    "nan": {
        "enabled": True,
        "keep_partial": True
    },
    "magic": {
        "enabled": True,
        "values": (-99, 99),
        "keep_partial": True
    },
    "selection": {
        "enabled": True,
        "i_band_sn_threshold": 0.0,
        "magnitude_flux_key": "DESI_FLUX_R",
        "magnitude_threshold": 25.
    },
    "neg_errors": {
        "enabled": True
    },
    "qso_split": {
        "enabled": True,
        "z_lim_QSO_cut": 2.1,
        "spectype_key": "SPECTYPE",
        "redshift_key": "REDSHIFT",
        "redshift_key_fallback": "Z_DESI",
        "qso_label": "QSO",
        "low_label": "QSO_low",
        "high_label": "QSO_high"
    },
    "collapse": {
        "enabled": True,
        "positive_collapse_label": None,
        "spectype_key": "SPECTYPE",
        "negative_prefix": "no_"
    },
    "encoding": {
        "enabled": True,
        "shared_mappings": {
            "SPECTYPE":  {
                "QSO_high": 0,
                "QSO_low": 1,
                'GALAXY': 2,               
                'STAR': 3
            },
            "MORPHTYPE": {
                'DEV': 0,
                'EXP': 1,
                'GGAL': 2,
                'GPSF': 3,
                'PSF': 4,
                'REX': 5,
                'SER': 6,
                'nan': 7
            },
        }
    },
    "normalization": {
        "enabled": True,
        "file_path": os.path.join(DATA_path, "norm_params.pkl"),
        "obs_key": "all_observations",
        "err_key": "all_errors",
        "pd_key": "all_pd",
        "suffix": "_normalized"
    }
}

# ----------------------------
# Train/Val/Test split options
# ----------------------------
splits = ["train", "val", "test"]

dict_split_data_options = {
    # Splitting ratios for intersecting datasets
    "train_ratio_intersection"            : 0.3,
    "val_ratio_intersection"              : 0.3,
    "test_ratio_intersection"             : 0.4,
    "random_seed_split_intersection"      : default_seed,
    # Splitting ratios for outersecting dataset
    "train_ratio_outersection"       : 0.70,
    "val_ratio_outersection"         : 0.15,
    "test_ratio_outersection"        : 0.15,
    "random_seed_split_outersection": default_seed
}

# ----------------------------
# Feature/label configuration
# ----------------------------
keys_load_features = {
    "keys_xx": ['all_observations_normalized', 'all_errors_normalized', 'MORPHTYPE_int'],
    "keys_yy": ['SPECTYPE_int', 'TARGETID'],
}
