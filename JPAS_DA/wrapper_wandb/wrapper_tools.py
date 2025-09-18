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
    if config["data"]["cleaning_config"] == "default_global_setup":
        config["data"]["cleaning_config"] = global_setup.config_dict_cleaning

    # === Training === #
    if config["training"]["path_save"] == "default_global_setup":
        config["training"]["path_save"] = os.path.join(global_setup.path_models, run_name)

    return config_0, config

def wrapper_data_loaders_from_config(dict_data):
    
    logging.info("📦 Parsing configuration to prepare DataLoaders...")

    bundle = wrapper_data_loaders.wrapper_build_dataloaders_current(
        root_path=global_setup.DATA_path,
        include=["JPAS_x_DESI_Raul", "DESI_mocks_Raul"],
        dataset_params={
            "JPAS_x_DESI_Raul": {"datasets": global_setup.load_JPAS_x_DESI_Raul},
            "DESI_mocks_Raul":  {"datasets": global_setup.load_DESI_mocks_Raul},
            "Ignasi":           {"datasets": global_setup.load_Ignasi},  # optional; won't be used in loaders unless you include it in "include"
        },
        random_seed_load=global_setup.default_seed,
        cleaning_config=dict_data["cleaning_config"],
        crossmatch_pair=("DESI_mocks_Raul", "JPAS_x_DESI_Raul"),
        id_key="TARGETID",
        split_config=global_setup.dict_split_data_options,
        keys_xx=dict_data["keys_xx"],
        keys_yy=dict_data["keys_yy"],
        return_artifacts=True,
    )
    dset_loaders = bundle["dataloaders"]

    key_survey_training = dict_data["key_survey_training"]
    logging.info(f"✅ Returning DataLoaders from training survey key: {key_survey_training}")
    return dset_loaders["train"][key_survey_training], dset_loaders["val"][key_survey_training]

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

