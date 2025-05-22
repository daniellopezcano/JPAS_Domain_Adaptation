DATA_path = "/home/dlopez/Documentos/0.profesional/Postdoc/USP/Projects/JPAS_Domain_Adaptation/DATA/Train-Validate-Test"

path_models = "/home/dlopez/Documentos/0.profesional/Postdoc/USP/Projects/JPAS_Domain_Adaptation/SAVED_models"

path_configs = "/home/dlopez/Documentos/0.profesional/Postdoc/USP/Projects/JPAS_Domain_Adaptation/configs"

default_seed = 42

load_JPAS_data = [{
    "name": "all",
    "npy": "JPAS_DATA_Aper_Cor_3_FLUX+NOISE.npy",
    "csv": "JPAS_DATA_PROPERTIES.csv",
    "sample_percentage": 1.0  # Optional, defaults to 1.0
}]

load_DESI_data = [
{
    "name": "train",
    "npy": "mock_3_train.npy",
    "csv": "props_training.csv",
    "sample_percentage": 1.0
},
{
    "name": "val",
    "npy": "mock_3_validate.npy",
    "csv": "props_validate.csv",
    "sample_percentage": 1.0
},
{
    "name": "test",
    "npy": "mock_3_test.npy",
    "csv": "props_test.csv",
    "sample_percentage": 1.0
}
]

dict_clean_data_options = {
    "apply_masks"         : ["unreliable", "magic_numbers", "negative_errors", "nan_values", "apply_additional_filters"],
    "mask_indices"        : [0, -2],
    "magic_numbers"       : [-99, 99],
    "i_band_sn_threshold" : 0,
    "z_lim_QSO_cut"       : 2.2
}