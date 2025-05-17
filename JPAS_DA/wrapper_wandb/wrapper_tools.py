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

def wrapper_train_models(
    dict_data: dict,
    path_load: Optional[str],
    hidden_layers_encoder: List[int],
    dropout_rates_encoder: List[float],
    output_dim_encoder: int,
    hidden_layers_downstream: List[int],
    dropout_rates_downstream: List[float],
    sampling_strategy: str,
    freeze_downstream_model: bool,
    NN_epochs: int,
    NN_batches_per_epoch: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    clip_grad_norm: Optional[float],
    seed_mode: str,
    seed: int,
    path_save: str,
    batch_size_val: Optional[int] = None,
    device: Optional[str] = "cuda",
    default_overwrite: Optional[bool] = False
):
    """
    Full training wrapper for encoder and downstream models.

    Handles normalization, model construction, loss setup, and training loop execution.

    Returns:
        dset_train, dset_val, model_encoder, model_downstream, min_val_loss
    """

    # ────────────────────────────────────────────────
    # 1. Load normalization if checkpoint provided
    # ────────────────────────────────────────────────
    if path_load is None:
        dict_data["provided_normalization"] = None
    else:
        means, stds = save_load_tools.load_means_stds(path_load)
        dict_data["provided_normalization"] = (means, stds)

    key_survey_training = dict_data.pop("key_survey_training")
    dset_loaders = wrapper_data_loaders.wrapper_data_loaders(**dict_data)
    dset_train = dset_loaders[key_survey_training]["train"]
    dset_val = dset_loaders[key_survey_training]["val"]
    
    # Return to default, makes my life far easier when reading back stored config files :D
    dict_data["provided_normalization"] = None

    # ────────────────────────────────────────────────
    # 2. Infer dimensions and initialize models
    # ────────────────────────────────────────────────
    n_classes = len(dset_train.class_labels)

    path_load_encoder = os.path.join(path_load, "model_encoder.pt") if path_load else None
    path_load_downstream = os.path.join(path_load, "model_downstream.pt") if path_load else None

    aux_tools.set_seed(seed)

    if path_load_encoder:
        assert os.path.isfile(path_load_encoder), f"File does not exist: {path_load_encoder}"
        config_encoder, model_encoder = save_load_tools.load_model_from_checkpoint(
            path_load_encoder, model_building_tools.create_mlp
        )
    else:
        xx, _ = dset_train(batch_size=1)
        config_encoder = {
            'input_dim': xx.shape[-1],
            'hidden_layers': hidden_layers_encoder,
            'dropout_rates': dropout_rates_encoder,
            'output_dim': output_dim_encoder,
            'use_batchnorm': False,
            'use_layernorm_at_output': False,
            'init_method': 'xavier'
        }
        model_encoder = model_building_tools.create_mlp(**config_encoder)

    if path_load_downstream:
        assert os.path.isfile(path_load_downstream), f"File does not exist: {path_load_downstream}"
        config_downstream, model_downstream = save_load_tools.load_model_from_checkpoint(
            path_load_downstream, model_building_tools.create_mlp
        )
    else:
        config_downstream = {
            'input_dim': output_dim_encoder,
            'hidden_layers': hidden_layers_downstream,
            'dropout_rates': dropout_rates_downstream,
            'output_dim': n_classes,
            'use_batchnorm': False,
            'use_layernorm_at_output': False,
            'init_method': 'xavier'
        }
        model_downstream = model_building_tools.create_mlp(**config_downstream)

    # ────────────────────────────────────────────────
    # 3. Build loss dictionary with class weights
    # ────────────────────────────────────────────────
    if sampling_strategy == "true_random":
        counts = dset_train.class_counts
        total_samples = np.sum(counts)
        weights = total_samples / (n_classes * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32)
    elif sampling_strategy == "class_random":
        class_weights = torch.tensor(np.ones(n_classes), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")

    loss_function_dict = {
        "type": "CrossEntropyLoss",
        "sampling_strategy": sampling_strategy,
        "class_weights": class_weights
    }

    # ────────────────────────────────────────────────
    # 4. Run training
    # ────────────────────────────────────────────────
    if batch_size_val is None:
        batch_size_val = len(dset_val.yy[list(dset_val.yy.keys())[0]])

    min_val_loss = training_tools.train_model(
        dset_train=dset_train,
        model_encoder=model_encoder,
        model_downstream=model_downstream,
        loss_function_dict=loss_function_dict,
        freeze_downstream_model=freeze_downstream_model,
        dset_val=dset_val,
        NN_epochs=NN_epochs,
        NN_batches_per_epoch=NN_batches_per_epoch,
        batch_size=batch_size,
        batch_size_val=batch_size_val,
        lr=lr,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        seed_mode=seed_mode,
        seed=seed,
        path_save=path_save,
        config_encoder=config_encoder,
        config_downstream=config_downstream,
        device=device,
        default_overwrite=default_overwrite
    )

    return dset_train, dset_val, model_encoder, model_downstream, min_val_loss


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

def wrapper_train_models_from_config(config_path, run_name, default_overwrite=True):
    """
    Wrapper to train models from config file
    """
    
    config = load_config_file(config_path)

    # === Global === #
    dict_global = config["global"]
    N_threads = dict_global["N_threads"]
    N_threads = aux_tools.set_N_threads_(N_threads=N_threads)

    # === Data === #
    dict_data = config["data"]

    # === Models === #
    dict_models = config["models"]

    path_load = dict_models["path_load"]

    if path_load is None:
        dict_encoder = dict_models["encoder"]
        hidden_layers_encoder = dict_encoder["hidden_layers"]
        dropout_rates_encoder = dict_encoder["dropout_rates"]
        output_dim_encoder = dict_encoder["output_dim"]

        dict_downstream = dict_models["downstream"]
        hidden_layers_downstream = dict_downstream["hidden_layers"]
        dropout_rates_downstream = dict_downstream["dropout_rates"]

    else:
        dict_encoder = {}
        hidden_layers_encoder = []
        dropout_rates_encoder = []
        output_dim_encoder = 0

        dict_downstream = {}
        hidden_layers_downstream = []
        dropout_rates_downstream = []

    # === Training === #
    dict_training = config["training"]

    path_save = dict_training["path_save"]
    path_save = os.path.join(path_save, run_name)
    config["training"]["path_save"] = path_save

    sampling_strategy = dict_training["sampling_strategy"]
    freeze_downstream_model = dict_training["freeze_downstream_model"]
    NN_epochs = dict_training["NN_epochs"]
    NN_batches_per_epoch = dict_training["NN_batches_per_epoch"]
    batch_size = dict_training["batch_size"]
    batch_size_val = dict_training["batch_size_val"]
    lr = dict_training["lr"]
    weight_decay = dict_training["weight_decay"]
    clip_grad_norm = dict_training["clip_grad_norm"]
    seed_mode = dict_training["seed_mode"]
    seed = dict_training["seed"]
    device = dict_training["device"]

    dset_train, dset_val, model_encoder, model_downstream, min_val_loss = wrapper_train_models(
        dict_data, path_load, hidden_layers_encoder, dropout_rates_encoder, output_dim_encoder,
        hidden_layers_downstream, dropout_rates_downstream, sampling_strategy, freeze_downstream_model,
        NN_epochs, NN_batches_per_epoch, batch_size, lr, weight_decay, clip_grad_norm, seed_mode, seed, path_save,
        batch_size_val=batch_size_val, device=device, default_overwrite=default_overwrite
    )

    os.makedirs(path_save, exist_ok=True)
    config_path = os.path.join(path_save, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    return config, dset_train, dset_val, model_encoder, model_downstream, min_val_loss
