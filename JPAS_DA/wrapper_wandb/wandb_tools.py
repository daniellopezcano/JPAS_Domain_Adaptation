from JPAS_DA.wrapper_wandb import wrapper_tools
import os
import torch
import numpy as np
from typing import Optional, Tuple, List
import yaml
import wandb
from pathlib import Path
import logging


def wandb_train(config=None):
    """
    Train a model using Weights & Biases (wandb) experiment tracking.
  
    Parameters:
    - config (dict, optional): Configuration dictionary containing training parameters. Defaults to None.
    Example:
    >>> wandb_train(config={"run_name": "test_run", "path_save": "./models", "lr": 0.001})
    """
    logging.info("🚀 Starting wandb training...")
  
    with wandb.init(config=config) as run:
        config = wandb.config
        logging.info(f"🔍 Running training for experiment: {run.name}")
        
        # Ensure the path exists
        path_save = os.path.join(config["training"]["path_save"], run.name)
        os.makedirs(path_save, exist_ok=True)
        # Save as a YAML file
        yaml_path = os.path.join(path_save, "config.yaml")
        with open(yaml_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
        print(f"✅ Configuration saved to {yaml_path}")

        loss = 0

        # # Train model
        # loss = wrapper_tools.wrapper_train_models_from_config(run_name=run.name, **config)
        # logging.info(f"✅ Training completed. Final loss: {loss:.6f}")
      
        # # Save experiment results
        # if "path_save" in config.keys():
        #     log_path = os.path.join(config["path_save"], 'register_sweeps.txt')
        #     logging.info(f"📄 Logging results to: {log_path}")
        #     with open(log_path, 'a') as ff:
        #         ff.write(run.name + ' ' + str(loss) + '\n')
      
        # Store loss in wandb summary
        wandb.run.summary["loss"] = loss
        logging.info("📊 Loss recorded in wandb summary.")


def wandb_sweep(wandb_config_name, wandb_project_name, N_samples_hyperparameters=5, path_to_config="../configs_SBI/"):

    logging.info("🚀 Starting wandb sweep...")
    logging.info(f"📂 Loading configuration from: {os.path.join(path_to_config, wandb_config_name)}")
  
    # Load sweep configuration
    sweep_config = wrapper_tools.load_config_file(path_to_config=path_to_config, config_file_name=wandb_config_name)
  
    # Log in to wandb
    logging.info("🔑 Logging into wandb...")
    wandb.login()
  
    # Initialize sweep
    logging.info(f"📊 Creating wandb sweep for project: {wandb_project_name}")
    sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)
  
    # Launch sweep agent
    logging.info(f"🎯 Running {N_samples_hyperparameters} hyperparameter optimization trials...")
    wandb.agent(sweep_id, wandb_train, count=N_samples_hyperparameters)
  
    logging.info("✅ Wandb sweep completed.")
