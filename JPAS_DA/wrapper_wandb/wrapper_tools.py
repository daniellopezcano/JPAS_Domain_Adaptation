from JPAS_DA import global_setup
from JPAS_DA.data import wrapper_data_loaders
from JPAS_DA.models import model_building_tools
from JPAS_DA.training import training_tools
from JPAS_DA.utils import aux_tools
from JPAS_DA.training import save_load_tools
import os
import torch
import numpy as np
from typing import Optional, Tuple, List
import yaml
from pathlib import Path
import logging
import copy


def load_config_file(config_path: str) -> dict:
    """
    Loads a YAML configuration file with detailed logging.
    
    Parameters:
    - config_path (str): Path to the YAML configuration file.
    
    Returns:
    - config (dict): Parsed YAML configuration, or None if an error occurs.
    """
    
    logging.info(f"🔍 Checking for config file at: {config_path}")

    config_path = Path(config_path)
    if not config_path.exists():
        raise ValueError("❌ ERROR: Config file does not exist at config_path")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.info(f"✅ Successfully loaded config file: {config_path}")
        if isinstance(config, dict):
            sample_keys = list(config.keys())[:3]
            logging.debug(f"📄 Parsed Config Data (Sample Keys): {sample_keys}")
        return config
    except yaml.YAMLError as e:
        logging.error(f"❌ YAML Parsing Error in {config_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ ERROR: Unexpected issue reading {config_path}: {e}")
        return None

def load_and_massage_config_file(config_path, run_name):
    """
    """

    config = load_config_file(config_path)
    config_0 = copy.deepcopy(config)

    # === Data === #
    path_load = config["models"]["path_load"]

    if path_load is None:
        config["data"]["provided_normalization"] = None
    else:
        path_load = os.path.join(global_setup.path_models, path_load)
        means, stds = save_load_tools.load_means_stds(path_load)
        config["data"]["provided_normalization"] = (means, stds)

    if config["data"]["data_paths"] == "default_global_setup":
        config["data"]["data_paths"] = {
            "root_path": global_setup.DATA_path,
            "load_JPAS_data": global_setup.load_JPAS_data,
            "load_DESI_data": global_setup.load_DESI_data,
            "random_seed_load": global_setup.default_seed
        }

    if config["data"]["dict_clean_data_options"] == "default_global_setup":
        config["data"]["dict_clean_data_options"] = global_setup.dict_clean_data_options
    
    if config["data"]["dict_split_data_options"] == "default_global_setup":
        config["data"]["dict_split_data_options"] = global_setup.dict_split_data_options

    # === Training === #
    if config["training"]["path_save"] == "default_global_setup":
        config["training"]["path_save"] = os.path.join(global_setup.path_models, run_name)

    return config_0, config

def wrapper_data_loaders_from_config(dict_data):
    """
    Wrapper function to initialize training and validation DataLoaders from a structured config dictionary.

    Parameters
    ----------
    dict_data : dict
        Dictionary containing all required parameters for loading, cleaning, splitting, and formatting data.

    Returns
    -------
    dset_train : DataLoader
        Training DataLoader from the selected survey.
    dset_val : DataLoader
        Validation DataLoader from the selected survey.
    """
    logging.info("📦 Parsing configuration to prepare DataLoaders...")

    # Extract configuration sub-dictionaries
    data_paths             = dict_data['data_paths']
    clean_opts             = dict_data['dict_clean_data_options']
    split_opts             = dict_data['dict_split_data_options']
    feature_opts           = dict_data['features_labels_options']
    provided_normalization = dict_data.get('provided_normalization', None)
    
    logging.info("🔧 Launching wrapper_data_loaders with loaded configuration...")
    dset_loaders = wrapper_data_loaders.wrapper_data_loaders(
        root_path                   = data_paths['root_path'],
        load_JPAS_data              = data_paths['load_JPAS_data'],
        load_DESI_data              = data_paths['load_DESI_data'],
        random_seed_load            = data_paths['random_seed_load'],

        apply_masks                 = clean_opts['apply_masks'],
        mask_indices                = clean_opts['mask_indices'],
        magic_numbers               = clean_opts['magic_numbers'],
        i_band_sn_threshold         = clean_opts['i_band_sn_threshold'],
        magnitude_flux_key          = clean_opts['magnitude_flux_key'],
        magnitude_threshold         = clean_opts['magnitude_threshold'],
        z_lim_QSO_cut               = clean_opts['z_lim_QSO_cut'],
        manually_select_one_SPECTYPE_vs_rest = clean_opts['manually_select_one_SPECTYPE_vs_rest'],

        train_ratio_both            = split_opts['train_ratio_both'],
        val_ratio_both              = split_opts['val_ratio_both'],
        test_ratio_both             = split_opts['test_ratio_both'],
        random_seed_split_both      = split_opts['random_seed_split_both'],

        train_ratio_only_DESI       = split_opts['train_ratio_only_DESI'],
        val_ratio_only_DESI         = split_opts['val_ratio_only_DESI'],
        test_ratio_only_DESI        = split_opts['test_ratio_only_DESI'],
        random_seed_split_only_DESI = split_opts['random_seed_split_only_DESI'],

        define_dataset_loaders_keys = feature_opts['define_dataset_loaders_keys'],
        keys_xx                     = feature_opts['keys_xx'],
        keys_yy                     = feature_opts['keys_yy'],
        normalization_source_key    = feature_opts['normalization_source_key'],
        normalize                   = feature_opts['normalize'],
        provided_normalization      = provided_normalization
    )

    key_survey_training = feature_opts["key_survey_training"]
    logging.info(f"✅ Returning DataLoaders from training survey key: {key_survey_training}")
    return dset_loaders[key_survey_training]["train"], dset_loaders[key_survey_training]["val"]

def wrapper_define_or_load_models_from_config(input_dim, n_classes, config_models):
    """
    Loads or builds encoder and downstream models based on configuration settings.

    Parameters
    ----------
    input_dim : int
        Input dimensionality for the encoder.
    n_classes : int
        Number of output classes for the downstream model.
    config_models : dict
        Configuration dictionary specifying the architecture or checkpoint paths.

    Returns
    -------
    model_encoder : nn.Module
        Encoder model instance.
    model_downstream : nn.Module
        Downstream classifier model instance.
    """
    path_load_models = config_models["path_load"]
    aux_tools.set_seed(0)

    # ───── Load or build encoder ─────
    if path_load_models:
        path_load_encoder = os.path.join(global_setup.path_models, path_load_models, "model_encoder.pt")
        assert os.path.isfile(path_load_encoder), f"❌ Encoder checkpoint not found: {path_load_encoder}"
        logging.info(f"📥 Loading encoder from checkpoint: {path_load_encoder}")
        config_encoder, model_encoder = save_load_tools.load_model_from_checkpoint(path_load_encoder, model_building_tools.create_mlp)
    else:
        logging.info("🛠️ Building encoder model from configuration...")
        config_encoder = {
            'input_dim': input_dim,
            'hidden_layers': config_models["encoder"]["hidden_layers"],
            'dropout_rates': config_models["encoder"]["dropout_rates"],
            'output_dim': config_models["encoder"]["output_dim"],
            'use_batchnorm': False,
            'use_layernorm_at_output': False,
            'init_method': 'xavier'
        }
        model_encoder = model_building_tools.create_mlp(**config_encoder)

    # ───── Load or build downstream model ─────
    if path_load_models:
        path_load_downstream = os.path.join(global_setup.path_models, path_load_models, "model_downstream.pt")
        assert os.path.isfile(path_load_downstream), f"❌ Downstream checkpoint not found: {path_load_downstream}"
        logging.info(f"📥 Loading downstream model from checkpoint: {path_load_downstream}")
        config_downstream, model_downstream = save_load_tools.load_model_from_checkpoint(path_load_downstream, model_building_tools.create_mlp)
    else:
        logging.info("🛠️ Building downstream model from configuration...")
        config_downstream = {
            'input_dim': config_models["encoder"]["output_dim"],
            'hidden_layers': config_models["downstream"]["hidden_layers"],
            'dropout_rates': config_models["downstream"]["dropout_rates"],
            'output_dim': n_classes,
            'use_batchnorm': False,
            'use_layernorm_at_output': False,
            'init_method': 'xavier'
        }
        model_downstream = model_building_tools.create_mlp(**config_downstream)

    logging.info("✅ Models ready for training or evaluation.")
    return config_encoder, model_encoder, config_downstream, model_downstream

def wrapper_train_routine_from_config(dset_train, dset_val, model_encoder, config_encoder, model_downstream, config_downstream, config_training):

    n_classes = len(dset_train.class_labels)
    
    loss_function_dict = {
        "type": "CrossEntropyLoss",
        "sampling_strategy": config_training["sampling_strategy"]
    }
    if config_training["sampling_strategy"] == "true_random":
        counts = dset_train.class_counts
        total_samples = np.sum(counts)
        weights = total_samples / (n_classes * counts)
        loss_function_dict["class_weights"] = torch.tensor(weights, dtype=torch.float32)
    elif config_training["sampling_strategy"] == "class_random":
        loss_function_dict["class_weights"] = torch.tensor(np.ones(n_classes), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported sampling strategy")

    min_val_loss = training_tools.train_model(
        dset_train=dset_train,
        model_encoder=model_encoder,
        model_downstream=model_downstream,
        loss_function_dict=loss_function_dict,
        freeze_downstream_model=config_training["freeze_downstream_model"],
        dset_val=dset_val,
        NN_epochs=config_training["NN_epochs"],
        NN_batches_per_epoch=config_training["NN_batches_per_epoch"],
        batch_size=config_training["batch_size"],
        batch_size_val=config_training["batch_size_val"],
        lr=config_training["lr"],
        weight_decay=config_training["weight_decay"],
        clip_grad_norm=config_training["clip_grad_norm"],
        seed_mode=config_training["seed_mode"],
        seed=config_training["seed"],
        path_save=config_training["path_save"],
        config_encoder=config_encoder,
        config_downstream=config_downstream,
        device=config_training["device"],
        default_overwrite=config_training["default_overwrite"]
    )

    return min_val_loss

def wrapper_train_from_config(config_path, run_name):

    # 1. Load and process config file
    config_0, config = load_and_massage_config_file(config_path, run_name)
    _ = aux_tools.set_N_threads_(N_threads=config['global']['N_threads'])

    # 2. Load data and define data loaders
    dset_train, dset_val = wrapper_data_loaders_from_config(config["data"])

    # 3. Define or reload models
    xx, _ = dset_train(batch_size=1)
    config_encoder, model_encoder, config_downstream, model_downstream = wrapper_define_or_load_models_from_config(
        input_dim=xx.shape[-1], n_classes=len(dset_train.class_labels), config_models=config["models"]
    )

    # 4. Taining
    min_val_loss = wrapper_train_routine_from_config(
        dset_train, dset_val, model_encoder, config_encoder, model_downstream, config_downstream, config["training"]
    )

    # 5. Save config file
    with open(os.path.join(config["training"]["path_save"], "config.yaml"), "w") as f:
        yaml.dump(config_0, f, sort_keys=False)

    return config_0, config, dset_train, dset_val, config_encoder, model_encoder, config_downstream, model_downstream, min_val_loss

