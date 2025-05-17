from JPAS_DA.wrapper_wandb import wrapper_tools
import os
import torch
import numpy as np
from typing import Optional, Tuple, List
import yaml
import wandb
from pathlib import Path
import logging

def wandb_sweep(wandb_config_name, wandb_project_name, N_samples_hyperparameters=5, path_to_config="../configs_SBI/"):

    logging.info("🚀 Starting wandb sweep...")
    logging.info(f"📂 Loading configuration from: {os.path.join(path_to_config, wandb_config_name)}")
  
    # Load sweep configuration
    sweep_config = wrapper_tools.load_config_file(os.path.join(path_to_config, wandb_config_name))
  
    # Log in to wandb
    logging.info("🔑 Logging into wandb...")
    wandb.login()
  
    # Initialize sweep
    logging.info(f"📊 Creating wandb sweep for project: {wandb_project_name}")
    print("sweep_config", sweep_config)
    sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)
  
    # Launch sweep agent
    logging.info(f"🎯 Running {N_samples_hyperparameters} hyperparameter optimization trials...")
    wandb.agent(sweep_id, wandb_train, count=N_samples_hyperparameters)
  
    logging.info("✅ Wandb sweep completed.")

def wandb_train(config=None):
    """
    Train a model using Weights & Biases (wandb) experiment tracking.

    Parameters:
    - config (dict, optional): Configuration dictionary containing training parameters. Defaults to None.
    Example:
    >>> wandb_train(config={"run_name": "test_run", "path_save": "./models", "lr": 0.001})
    """
    logging.info("Starting wandb training...")

    with wandb.init(config=config) as run:

        config = wandb.config
        run_name = run.name
        logging.info(f"Running sweep: {run_name}")

        # Load the fixed + model configuration template
        aux_config_path = config["aux_config_path"]
        with open(aux_config_path, "r") as f:
            full_config = yaml.safe_load(f)

        if full_config["models"]["path_load"] is None:
            # Extract and remove model options
            encoders = full_config.pop("encoders", None)
            downstreams = full_config.pop("downstreams", None)

            if encoders is None or downstreams is None:
                raise ValueError("Model options not found in the auxiliary config file.")

            # Set encoder and downstream from indexed choices
            try:
                encoder_cfg = encoders[config.encoder_id]
                encoder_cfg = encoder_cfg.copy()
                encoder_cfg["output_dim"] = config.output_dim
            except IndexError:
                raise ValueError(f"Invalid encoder_id {config.encoder_id} in wandb config.")

            try:
                downstream_cfg = downstreams[config.downstream_id]
                downstream_cfg = downstream_cfg.copy()
            except IndexError:
                raise ValueError(f"Invalid downstream_id {config.downstream_id} in wandb config.")

        # Clone the fixed parameters
        merged_config = full_config.copy()

        if full_config["models"]["path_load"] is None:
            merged_config["models"]["encoder"] = encoder_cfg
            merged_config["models"]["downstream"] = downstream_cfg

        # Inject sweep-controlled training hyperparameters
        sweep_keys = [
            "NN_epochs", "NN_batches_per_epoch", "batch_size", "batch_size_val",
            "lr", "weight_decay", "clip_grad_norm"
        ]
        for key in sweep_keys:
            if key in config:
                merged_config["training"][key] = getattr(config, key)
            else:
                logging.warning(f"Parameter {key} not found in wandb config.")

        # Save and dispatch
        path_save = os.path.join(merged_config["training"]["path_save"], run_name)
        os.makedirs(path_save, exist_ok=True)
        yaml_path = os.path.join(path_save, "wandb_config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(merged_config, f, sort_keys=False)

        logging.info(f"Configuration saved to {yaml_path}")

        # Call the training function
        config, dset_train, dset_val, model_encoder, model_downstream, min_val_loss = wrapper_tools.wrapper_train_models_from_config(
            config_path=yaml_path, run_name=run_name
        )

        # Store loss in wandb summary
        wandb.run.summary["loss"] = min_val_loss
        logging.info("📊 Loss recorded in wandb summary.")