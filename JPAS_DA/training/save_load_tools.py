import os
import torch
import logging
import numpy as np

def load_means_stds(
    path_dir: str,
    means_file_name: str = "means.npy",
    stds_file_name: str = "stds.npy",
    dtype=np.float32
) -> tuple:
    """
    Load saved means and stds (list of arrays) with enforced dtype.
    """
    means_path = os.path.join(path_dir, means_file_name)
    stds_path = os.path.join(path_dir, stds_file_name)

    logging.debug(f"📥 Loading means from: {means_path}")
    logging.debug(f"📥 Loading stds from: {stds_path}")

    means = [arr.astype(dtype) for arr in np.load(means_path, allow_pickle=True)]
    stds  = [arr.astype(dtype) for arr in np.load(stds_path, allow_pickle=True)]

    return means, stds

def load_model_from_checkpoint(
    path_checkpoint: str,
    model_builder: callable
) -> torch.nn.Module:
    """
    Load a PyTorch model from a checkpoint containing both config and state_dict.

    Parameters:
    ----------
    path_checkpoint : str
        Path to the saved .pt checkpoint file.
    model_builder : callable
        Function to create the model (e.g., model_building_tools.create_mlp)

    Returns:
    --------
    model : torch.nn.Module
        The loaded model.
    """

    checkpoint = torch.load(path_checkpoint, map_location='cpu')
    config = checkpoint['config']
    model = model_builder(**config)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    return config, model
