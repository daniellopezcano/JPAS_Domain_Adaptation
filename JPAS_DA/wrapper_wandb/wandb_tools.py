from JPAS_DA import global_setup
from JPAS_DA.wrapper_wandb import wrapper_tools
import os
import torch
import numpy as np
from typing import Optional, Tuple, List
import yaml
import wandb
from pathlib import Path
import logging

def wandb_sweep(path_wandb_config, wandb_project_name, N_samples_hyperparameters=5):

    logging.info("🚀 Starting wandb sweep...")
  
    # Load sweep configuration
    wandb_config = wrapper_tools.load_config_file(path_wandb_config)
  
    # Log in to wandb
    logging.info("🔑 Logging into wandb...")
    wandb.login()
  
    # Initialize sweep
    logging.info(f"📊 Creating wandb sweep for project: {wandb_project_name}")
    sweep_id = wandb.sweep(wandb_config, project=wandb_project_name)
  
    # Define wandb training function
    wandb_train = make_wandb_train(wandb_project_name)
    
    # Launch sweep agent
    logging.info(f"🎯 Running {N_samples_hyperparameters} hyperparameter optimization trials...")
    wandb.agent(sweep_id, wandb_train, count=N_samples_hyperparameters)
  
    logging.info("✅ Wandb sweep completed.")


def make_wandb_train(wandb_project_name):
    def wandb_train(
            config=None,
            sweep_keys = [
                "NN_epochs", "NN_batches_per_epoch", "batch_size", "batch_size_val", "lr", "weight_decay", "clip_grad_norm", "seed"
            ]
        ):
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
            aux_config_path = os.path.join(global_setup.path_configs, config["aux_config_path"])
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

            for key in sweep_keys:
                if key in config:
                    merged_config["training"][key] = getattr(config, key)
                else:
                    logging.warning(f"Parameter {key} not found in wandb config.")

            # Save and dispatch
            tmp_config_path = os.path.join(global_setup.path_configs, "tmp_wandb_config.yaml")
            os.makedirs(os.path.dirname(tmp_config_path), exist_ok=True)
            with open(tmp_config_path, "w") as f:
                yaml.dump(merged_config, f, sort_keys=False)

            # Call the training function
            _, _, _, _, _, _, _, _, min_val_loss = wrapper_tools.wrapper_train_from_config(
                config_path=tmp_config_path, run_name=os.path.join(wandb_project_name,run_name)
            )

            # Store loss in wandb summary
            wandb.run.summary["loss"] = min_val_loss
            logging.info("📊 Loss recorded in wandb summary.")
    return wandb_train
