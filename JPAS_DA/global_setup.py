import os

version = "0.1.0"

main_path = os.path.join("/home/dlopez/Documentos/0.profesional/Postdoc/USP/Projects/JPAS_Domain_Adaptation")
DATA_path = os.path.join(main_path, "DATA")
path_models = os.path.join(main_path, "SAVED_models")
path_configs = os.path.join(main_path, "configs")

default_seed = 137

load_JPAS_data = [{
    "name": "all",
    "npy": "JPAS_DATA_Aper_Cor_3_FLUX+NOISE.npy",
    "csv": "JPAS_DATA_PROPERTIES.csv",
    "pickle": "JPAS_DATA_PROPERTIES_AUX.pkl",
    "sample_percentage": 1.0  # Optional, defaults to 1.0
}]

load_DESI_data = [
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

dict_clean_data_options = {
    "apply_masks"         : ["unreliable", "jpas_ignasi_dense", "magic_numbers", "negative_errors", "nan_values", "apply_additional_filters"],
    "mask_indices"        : [0, -2],
    "magic_numbers"       : [-99, 99],
    "i_band_sn_threshold" : 0,
    "magnitude_flux_key"  : "DESI_FLUX_R",
    "magnitude_threshold" : 22.5,
    "z_lim_QSO_cut"       : 2.1,
    "manually_select_one_SPECTYPE_vs_rest" : None # "QSO_high"
}

dict_split_data_options = {
    # Splitting ratios for matched (both) datasets
    "train_ratio_both"            : 0.4,
    "val_ratio_both"              : 0.2,
    "test_ratio_both"             : 0.4,
    "random_seed_split_both"      : default_seed,
    # Splitting ratios for DESI-only dataset
    "train_ratio_only_DESI"       : 0.70,
    "val_ratio_only_DESI"         : 0.15,
    "test_ratio_only_DESI"        : 0.15,
    "random_seed_split_only_DESI": default_seed
}

features_labels_options = {
    "define_dataset_loaders_keys": ["DESI_only", "JPAS_matched"], # ["DESI_combined", "DESI_only", "DESI_matched", "JPAS_matched"]
    "keys_xx": ['OBS', 'ERR', 'MORPHTYPE_int'], # ['OBS', 'ERR', 'MORPHTYPE_int']
    "keys_yy": ['SPECTYPE_int', 'TARGETID'], # 'SPECTYPE_int', 'TARGETID'
    "normalize": True,
    "normalization_source_key": "DESI_only" # "DESI_combined", "DESI_only", "DESI_matched", "JPAS_matched"
}

class_names = ['GALAXY', 'QSO_high', 'QSO_low', 'STAR']

load_DESI_data_Lilianne = [
{
    "name": "train",
    "csv": "train_simMB.csv",
    "sample_percentage": 1.0
},
{
    "name": "val",
    "csv": "val_simMB.csv",
    "sample_percentage": 1.0
},
{
    "name": "test",
    "csv": "test_simMB.csv",
    "sample_percentage": 1.0
}
]

load_JPAS_data_Ignasi = {
    "fits": "jpas_idr_classification_xmatch_desi_dr1.fits.gz",
    "fits_obs" : "jpas_mag_23_5_flag_0_40_filters.fits"
}