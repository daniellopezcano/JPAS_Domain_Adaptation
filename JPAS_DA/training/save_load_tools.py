import os
import torch
import torch.nn as nn
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

    logging.info(f"📥 Loading means from: {means_path}")
    logging.info(f"📥 Loading stds from: {stds_path}")

    means = [arr.astype(dtype) for arr in np.load(means_path, allow_pickle=True)]
    stds  = [arr.astype(dtype) for arr in np.load(stds_path, allow_pickle=True)]

    return means, stds

def apply_dropout_rates(model: nn.Module, rates):
    """Set dropout p layer-by-layer. Accepts a scalar or a list."""
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    if isinstance(rates, (float, int)):
        for d in drops: d.p = float(rates)
        return

    assert len(rates) == len(drops), (
        f"Provided {len(rates)} rates but model has {len(drops)} dropout layers."
    )
    for d, p in zip(drops, rates):
        d.p = float(p)

    logging.info(f"📥 Applied dropout rates: {rates}")

def apply_batchnorm_override(model: nn.Module, override):
    """
    Override BatchNorm behavior/hyperparams in-place.

    override can be:
      - True / 'train' : force BN to training mode
      - False / 'eval' : force BN to eval mode (use running stats)
      - 'freeze'       : eval mode + freeze affine params (no grads)
      - dict           : set BN attributes like momentum/eps; optional 'mode' in {'train','eval','freeze'}
    """
    bn_modules = [m for m in model.modules()
                  if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if not bn_modules:
        logging.info("ℹ️ No BatchNorm layers found; override skipped.")
        return

    # Normalize override into (mode, attrs)
    mode = None
    attrs = {}
    if isinstance(override, dict):
        mode = override.get("mode", None)
        attrs = {k: v for k, v in override.items() if k != "mode"}
    elif override is True or override == "train":
        mode = "train"
    elif override is False or override == "eval":
        mode = "eval"
    elif override == "freeze":
        mode = "freeze"
    else:
        raise ValueError(f"Unsupported use_batchnorm value: {override!r}")

    # Apply mode
    if mode == "train":
        for m in bn_modules:
            m.train(True)  # use batch stats & update running stats
        logging.info(f"🔧 Forced {len(bn_modules)} BatchNorm layers to TRAIN mode")
    elif mode == "eval":
        for m in bn_modules:
            m.eval()       # use running stats, no updates
        logging.info(f"🔧 Forced {len(bn_modules)} BatchNorm layers to EVAL mode")
    elif mode == "freeze":
        for m in bn_modules:
            m.eval()
            if m.affine:
                if m.weight is not None:
                    m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)
        logging.info(f"🧊 Froze {len(bn_modules)} BatchNorm layers (eval mode, no grads for affine params)")

    # Apply attribute tweaks (momentum/eps/etc.)
    if attrs:
        for m in bn_modules:
            for k, v in attrs.items():
                if hasattr(m, k):
                    setattr(m, k, v)
        logging.info(f"🔧 Updated BatchNorm attrs: {attrs}")

def load_model_from_checkpoint(path_checkpoint, model_builder, override_dropout=None, use_batchnorm=None):
    """
    Load model (config + state_dict), then optionally override dropout p and batchnorm behavior in-place.
    """
    checkpoint = torch.load(path_checkpoint, map_location='cpu')
    config = checkpoint['config']
    print("config", config)
    model = model_builder(**config)
    model.load_state_dict(checkpoint['state_dict'])

    # Optional overrides
    if override_dropout is not None:
        apply_dropout_rates(model, override_dropout)
        config = {**config, 'dropout_rates': override_dropout}  # keep returned config honest
    if use_batchnorm is not None:
        apply_batchnorm_override(model, use_batchnorm)
        # reflect "mode" only if user supplied a dict or a simple mode; we don't try to mutate use_batchnorm flag
        config = {**config, 'use_batchnorm': use_batchnorm}

    model.eval()
    return config, model
