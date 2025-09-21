import os
import shutil
import torch
import numpy as np
import logging
from typing import Union, Optional, Tuple, Dict
from datetime import datetime
from JPAS_DA.training import loss_functions

def train_model(
    dset_train,
    model_encoder: torch.nn.Module,
    model_downstream: torch.nn.Module,
    loss_function_dict: Dict,
    freeze_downstream_model: bool,
    dset_val: Optional[float] = None,
    NN_epochs: Optional[int] = 100,
    NN_batches_per_epoch: Optional[int] = 10,
    batch_size: Optional[int] = 64,
    batch_size_val: Optional[int] = None,
    lr: Optional[float] = 1e-3,
    weight_decay: Optional[float] = 0.0,
    clip_grad_norm: Optional[float] = None,
    seed_mode: Optional[str] = "random",  # 'random', 'deterministic', or 'overfit'
    seed: Optional[int] = 0,
    path_save: Optional[str] = None,
    config_encoder: Optional[Dict] = None,
    config_downstream: Optional[Dict] = None,
    device: Optional[str] = None,
    default_overwrite: Optional[bool] = False
) -> float:
    """
    Train a classifier using the provided encoder and downstream model.

    Parameters:
    - dset_train: Training dataset object.
    - model_encoder: Feature extractor.
    - model_downstream: Classification head.
    - loss_function_dict: Dict containing 'type', 'sampling_strategy', 'class_weights'.
    - freeze_downstream_model: Whether to freeze downstream model weights.
    - dset_val: Optional validation dataset.
    - NN_epochs: Number of epochs to train.
    - NN_batches_per_epoch: Number of training batches per epoch.
    - batch_size: Training batch size.
    - batch_size_val: Validation batch size.
    - lr: Learning rate.
    - weight_decay: Weight decay for optimizer.
    - clip_grad_norm: Optional gradient clipping value.
    - seed_mode: "random", "deterministic", or "overfit".
    - seed: Integer seed.
    - path_save: Where to save model weights and logs.
    - config_encoder: Optional dictionary with encoder config.
    - config_downstream: Optional dictionary with downstream model config.

    Returns:
    - min_val_loss: Best validation loss achieved.
    """

    assert seed_mode in {"random", "deterministic", "overfit"}
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create log directory if it exists already, or create a new one
    if path_save is not None:
        if os.path.exists(path_save):
            logging.info(f"Directory '{path_save}' already exists.")
            while True:
                if default_overwrite:
                    logging.info(f"Overwriting '{path_save}'...")
                    shutil.rmtree(path_save)
                    os.makedirs(path_save)
                    logging.info(f"Directory '{path_save}' has been overwritten.")
                    break
                else:
                    choice = input("Do you want to [O]verwrite, [N]ew path, or [C]ancel? ").strip().lower()
                    if choice == 'o':
                        logging.info(f"Overwriting '{path_save}'...")
                        shutil.rmtree(path_save)
                        os.makedirs(path_save)
                        logging.info(f"Directory '{path_save}' has been overwritten.")
                        break
                    elif choice == 'n':
                        path_save = input("Enter new directory path: ").strip()
                        if not os.path.exists(path_save):
                            os.makedirs(path_save)
                            logging.info(f"Created new directory: '{path_save}'")
                            break
                        else:
                            logging.info("That path already exists. Please choose another.")
                    elif choice == 'c':
                        raise RuntimeError("Operation cancelled by user.")
                    else:
                        logging.info("Invalid choice. Please enter 'O', 'N', or 'C'.")
        else:
            os.makedirs(path_save)
            logging.info(f"Created directory: '{path_save}'")

    # Move models to device
    model_encoder.to(device)
    model_downstream.to(device)

    # Use same batch size for validation if not provided
    if batch_size_val is None:
        batch_size_val = len(dset_val.yy[list(dset_val.yy.keys())[0]])

    # Use training set for validation if in overfit mode
    if seed_mode == "overfit":
        dset_val = dset_train
        batch_size_val = batch_size
        logging.info("🔁 Overfit mode: using training set as validation set.")

    # Set up optimizers
    if freeze_downstream_model:
        optimizer = torch.optim.AdamW(model_encoder.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=30, factor=0.3, min_lr=1e-8)
    else:
        optimizer = {
            "encoder": torch.optim.AdamW(model_encoder.parameters(), lr=lr, weight_decay=weight_decay),
            "downstream": torch.optim.AdamW(model_downstream.parameters(), lr=lr, weight_decay=weight_decay)
        }
        scheduler = {
            "encoder": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer["encoder"], mode='min', patience=30, factor=0.3, min_lr=1e-8),
            "downstream": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer["downstream"], mode='min', patience=30, factor=0.3, min_lr=1e-8)
        }

    # === Training Loop ===
    logging.info("🚀 Starting training loop...")
    min_val_loss = None
    for epoch in range(1, NN_epochs + 1):
        logging.info(f"\n📚 Epoch {epoch}/{NN_epochs}")

        seed0 = NN_batches_per_epoch * epoch if seed_mode == "deterministic" else 0

        # === Train ===
        train_single_epoch(
            dset_train=dset_train, optimizer=optimizer,
            model_encoder=model_encoder, model_downstream=model_downstream,
            loss_function_dict=loss_function_dict, batch_size=batch_size, NN_batches_per_epoch=NN_batches_per_epoch,
            clip_grad_norm=clip_grad_norm, freeze_downstream_model=freeze_downstream_model,
            seed_mode=seed_mode, seed=seed, seed0=seed0, device=device
        )
        
        # === Evaluate ===
        min_val_loss, train_loss, val_loss = eval_single_epoch(
            dset_train=dset_train, dset_val=dset_val, scheduler=scheduler, model_encoder=model_encoder, model_downstream=model_downstream,
            freeze_downstream_model=freeze_downstream_model, loss_function_dict=loss_function_dict, path_save=path_save,
            min_val_loss=min_val_loss, batch_size=batch_size_val, device=device, seed=seed,
            config_encoder=config_encoder, config_downstream=config_downstream
        )

    logging.info("✅ Training complete.")
    return min_val_loss

def train_single_epoch(
    dset_train,
    optimizer: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
    model_encoder: torch.nn.Module,
    model_downstream: torch.nn.Module,
    loss_function_dict: Dict,
    batch_size: int = 64,
    NN_batches_per_epoch: int = 10,
    clip_grad_norm: Optional[float] = None,
    freeze_downstream_model: bool = False,
    seed_mode: str = "random",
    seed: int = 0,
    seed0: int = 0,
    device: str = "cuda",
    print_progress: int = 10
) -> None:
    """
    Train one epoch on the training set.

    Parameters:
    - dset_train: Training DataLoader-compatible object.
    - optimizer: Optimizer or dictionary of optimizers.
    - model_encoder: Feature extractor model.
    - model_downstream: Classification head model.
    - loss_function_dict: Dictionary with loss config (must include type and class_weights).
    - batch_size: Samples per batch.
    - NN_batches_per_epoch: How many batches to train in this epoch.
    - clip_grad_norm: Optional gradient clipping value.
    - freeze_downstream_model: Whether to freeze the downstream model.
    - seed_mode: "random" | "deterministic"
    - seed: Random seed for reproducibility.
    - seed0: Offset for deterministic seeds.
    - device: Training device.
    - print_progress: Number of progress messages during training.
    """

    model_encoder.train()
    if not freeze_downstream_model:
        model_downstream.train()
    else:
        model_downstream.eval() # This will prevent batch noremalization layer from being updated

    print_every = max(1, NN_batches_per_epoch // print_progress)
    logging.info(f"📚 Training {NN_batches_per_epoch} batches...")
    for batch_idx in range(NN_batches_per_epoch):
        # Set random seed
        if seed_mode == "random":
            current_seed = datetime.now().microsecond % 13007
        elif seed_mode == "deterministic":
            current_seed = seed0 + batch_idx
        else:
            current_seed = seed
        logging.debug(f"📚 Batch {batch_idx}/{NN_batches_per_epoch} | Seed: {current_seed}")
        # === Load batch ===
        xx, yy_true = dset_train(
            batch_size,
            seed=current_seed,
            sampling_strategy=loss_function_dict["sampling_strategy"],
            to_torch=True,
            device=device
        )
        logging.debug(f"xx.shape: {xx.shape} | yy_true.shape: {yy_true.shape}")
        # === Forward ===
        features = model_encoder(xx)
        logits = model_downstream(features)
        logging.debug(f"features.shape: {features.shape} | logits.shape: {logits.shape}")
        logging.debug(f"features.dtype: {features.dtype} | logits.dtype: {logits.dtype}")
        logging.debug(f"yy_true: {yy_true}")
        # === Loss ===
        loss = loss_functions.cross_entropy(
            yy_true, logits,
            loss_function_dict["class_weights"].to(device),
            parameters=model_encoder.parameters(),
            l2_lambda=loss_function_dict["l2_lambda"],
        )
        logging.debug(f"loss: {loss}")
        # === Backprop ===
        loss.backward()

        # === Gradient step ===
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model_encoder.parameters(), clip_grad_norm)
            if not freeze_downstream_model:
                torch.nn.utils.clip_grad_norm_(model_downstream.parameters(), clip_grad_norm)

        if freeze_downstream_model:
            optimizer.step()
            optimizer.zero_grad()
        else:
            optimizer["encoder"].step()
            optimizer["encoder"].zero_grad()
            optimizer["downstream"].step()
            optimizer["downstream"].zero_grad()

        # === Print progress ===
        if (batch_idx + 1) % print_every == 0:
            logging.debug(f"🧪 Batch {batch_idx+1}/{NN_batches_per_epoch} | Loss: {loss.item():.6f}")

        model_encoder.eval()
        model_downstream.eval()

        return

def eval_single_epoch(
    dset_train,
    dset_val,
    scheduler,
    model_encoder: torch.nn.Module,
    model_downstream: torch.nn.Module,
    loss_function_dict: Dict,
    freeze_downstream_model: bool = False,
    path_save: Optional[str] = None,
    min_val_loss: Optional[float] = None,
    batch_size: int = 64,
    seed: int = 0,
    device: str = "cuda",
    config_encoder: Optional[Dict] = None,
    config_downstream: Optional[Dict] = None
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Evaluate one epoch and optionally save best-performing model weights.

    Parameters:
    - dset_train: Training DataLoader-compatible object.
    - dset_val: Validation DataLoader-compatible object.
    - scheduler: Learning rate scheduler or dict of schedulers.
    - model_encoder: Feature extractor model.
    - model_downstream: Classification head model.
    - loss_function_dict: Dictionary defining loss configuration.
    - freeze_downstream_model: Whether the downstream model is frozen.
    - path_save: Directory to save model checkpoints.
    - min_val_loss: Best validation loss seen so far.
    - batch_size: Evaluation batch size.
    - seed: Random seed for batch reproducibility.
    - device: Computation device.

    Returns:
    - min_val_loss (float): Updated minimum validation loss.
    - train_loss (dict): Training loss dictionary.
    - val_loss (dict): Validation loss dictionary.
    """

    # Evaluate training and validation loss
    train_loss = eval_dataset(
        dset=dset_train, batch_size=batch_size, model_encoder=model_encoder, model_downstream=model_downstream,
        loss_function_dict=loss_function_dict, seed=seed, device=device
    )

    val_loss = eval_dataset(
        dset=dset_val, batch_size=batch_size, model_encoder=model_encoder, model_downstream=model_downstream,
        loss_function_dict=loss_function_dict, seed=seed, device=device
    )

    # Learning rate logging
    if isinstance(scheduler, dict):
        current_lr = scheduler["encoder"].optimizer.param_groups[0]["lr"]
    else:
        current_lr = scheduler.optimizer.param_groups[0]["lr"]
    
    logging.info(
        f"📊 Epoch Eval | train_loss: {train_loss['loss']:.6f}, val_loss: {val_loss['loss']:.6f}, lr: {current_lr:.2e}"
    )

    if min_val_loss is None or val_loss["loss"] < min_val_loss:
        logging.info(f"✅ New best model!")
        min_val_loss = val_loss["loss"]
        if path_save:
            logging.info(f"Saving to: {path_save}")

            # Save encoder
            checkpoint_encoder = {'config': config_encoder, 'state_dict': model_encoder.state_dict()}
            path_encoder = os.path.join(path_save, "model_encoder.pt")
            os.makedirs(os.path.dirname(path_encoder), exist_ok=True)
            torch.save(checkpoint_encoder, path_encoder)
            logging.info(f"✅ Encoder saved to '{path_encoder}'")

            # Save downstream
            checkpoint_downstream = {'config': config_downstream, 'state_dict': model_downstream.state_dict()}
            path_downstream = os.path.join(path_save, "model_downstream.pt")
            os.makedirs(os.path.dirname(path_downstream), exist_ok=True)
            torch.save(checkpoint_downstream, path_downstream)
            logging.info(f"✅ Downstream model saved to '{path_downstream}'")

    if path_save:
        with open(os.path.join(path_save, 'register.txt'), 'a') as ff:
            ff.write('%.4e %.4e %.4e\n' % (train_loss['loss'], val_loss['loss'], current_lr))

    # Update learning rate scheduler
    if freeze_downstream_model:
        scheduler.step(val_loss["loss"])
    else:
        scheduler["encoder"].step(val_loss["loss"])
        scheduler["downstream"].step(val_loss["loss"])

    return min_val_loss, train_loss, val_loss

def eval_dataset(
    dset,
    batch_size: int,
    model_encoder: torch.nn.Module,
    model_downstream: torch.nn.Module,
    loss_function_dict: Dict,
    seed: int = 0,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluates a dataset using cross-entropy loss and returns the computed loss metric.

    Parameters:
    - dset: DataLoader-compatible callable object.
    - batch_size (int): Number of samples per evaluation batch.
    - model_encoder (nn.Module): Feature extractor model.
    - model_downstream (nn.Module): Classifier head model.
    - loss_function_dict (dict): Dictionary containing 'type', 'class_weights', and 'sampling_strategy'.
    - seed (int, optional): Seed for batch sampling (default: 0).
    - device (str, optional): Device for computation (default: "cuda").

    Returns:
    - metrics (dict): Dictionary containing the scalar loss.
    """
    assert loss_function_dict["type"] == "CrossEntropyLoss", "Only CrossEntropyLoss supported in simplified version"

    with torch.no_grad():
        # Fetch batch from dataset
        xx, yy_true = dset(
            batch_size, seed=seed, sampling_strategy=loss_function_dict.get("sampling_strategy", "true_random"), to_torch=True, device=device
        )

        # Forward pass
        features = model_encoder(xx)
        logits = model_downstream(features)

        # Compute loss
        loss = loss_functions.cross_entropy(yy_true, logits, loss_function_dict["class_weights"].to(device))

        metrics = {"loss": loss.item()}

    logging.debug(f"📉 Evaluation loss: {metrics['loss']:.6f}")
    return metrics