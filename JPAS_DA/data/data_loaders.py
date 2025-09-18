import numpy as np
import logging
import datetime
import torch


def sample_class_balanced_indices(class_counts, class_indexes, batch_size, seed=None):
    """
    Samples a batch of indices with approximately balanced representation across classes.

    This function ensures that each class contributes equally to the sampled batch (if possible),
    and if the batch size is not divisible evenly, it adds the remaining samples from randomly selected classes.

    Parameters:
    - class_counts (list or np.ndarray): List of number of samples in each class.
    - class_indexes (list of np.ndarray): Each entry is an array of the original indices for that class.
    - batch_size (int): Total number of samples to draw.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - batch_idxs (np.ndarray): 1D array of selected indices, shuffled. Total length == batch_size.
    """
    if seed is not None:
        np.random.seed(seed)

    samples_per_class = batch_size // len(class_counts)
    logging.debug(f"Sampling ~{samples_per_class} per class for batch of {batch_size} samples.")

    batch_idxs = []
    # Core balanced sampling
    for ii in range(len(class_counts)):
        tmp_NN_class = class_counts[ii]
        tmp_idxs = np.random.choice(tmp_NN_class, samples_per_class, replace=False)
        tmp_batch_idxs = class_indexes[ii][tmp_idxs]
        batch_idxs.extend(tmp_batch_idxs)

    # Add leftovers to reach desired batch size
    leftover = batch_size - len(class_counts) * samples_per_class
    logging.debug(f"Adding {leftover} leftover samples from random classes.")

    if leftover > 0:
        leftover_class_choices = np.random.choice(len(class_counts), leftover, replace=False)
        for class_label in leftover_class_choices:
            tmp_NN_class = class_counts[class_label]
            tmp_idx = np.random.choice(tmp_NN_class, 1, replace=False)
            tmp_batch_idx = class_indexes[class_label][tmp_idx]
            batch_idxs.extend(tmp_batch_idx)

    batch_idxs = np.array(batch_idxs)
    np.random.shuffle(batch_idxs)  # Optional: shuffle final batch
    return batch_idxs

def stack_features_from_dict_flattened(data_dict, indices):
    """
    Given a dict of arrays and a list of indices, this function slices and concatenates
    all arrays along the feature axis to produce a flat (batch_size, total_features) array.

    Parameters
    ----------
    data_dict : dict
        Dictionary of arrays with shape (N, ...) for each key.
    indices : list or np.ndarray
        Indices to extract from each array.

    Returns
    -------
    stacked : np.ndarray
        A 2D array of shape (len(indices), total_features), where all selected features
        from all keys have been concatenated.
    """
    logging.debug(f"📦 Flattening and concatenating features from keys: {list(data_dict.keys())}")

    features = []
    for key in data_dict:
        array = np.asarray(data_dict[key])[indices]

        if array.ndim == 1:
            array = array[:, np.newaxis]  # Convert shape (N,) → (N, 1)

        elif array.ndim > 2:
            array = array.reshape(array.shape[0], -1)  # Flatten all but batch dim

        features.append(array)

    stacked = np.concatenate(features, axis=1)  # Concatenate along feature dim

    logging.debug(f"✅ Final stacked shape: {stacked.shape}")
    return stacked


class DataLoader:
    def __init__(self, xx, yy):
        
        expected_length = next(iter(xx.values())).shape[0]
        logging.info(f"├── 💿 Initializing DataLoader object with {expected_length} samples...")
        
        self.xx = xx
        self.yy = yy
        self.NN_xx = expected_length

        # Class processing
        key_label  = None
        if 'SPECTYPE_int' in yy:
            key_label = 'SPECTYPE_int'
            self.class_labels, self.class_counts = np.unique(yy[key_label], return_counts=True)
            self.class_indexes = [np.where(yy[key_label] == label)[0] for label in self.class_labels]

            total = np.sum(self.class_counts)
            class_info = ", ".join([f"{label}: {count} ({100*count/total:.2f}%)" for label, count in zip(self.class_labels, self.class_counts)])
            logging.info(f"├── ✔ Finished Initialization. Class distribution: [{class_info}]")
        else:
            logging.info("├── ✔ Finished Initialization. No classification targets detected.")


    def __call__(
        self,
        batch_size,
        seed="random",
        sampling_strategy="true_random",
        custom_idxs=None,
        to_torch=False,
        device="cpu",
        return_batch_idxs=False
    ):
        # Generate seed if "random" mode is selected
        if seed == "random":
            seed = datetime.datetime.now().microsecond % 13037
        np.random.seed(seed)

        # Sample batch indices
        if custom_idxs is None:
            if sampling_strategy == "true_random":
                batch_idxs = np.random.choice(self.NN_xx, batch_size, replace=False)
            elif sampling_strategy == "class_random":
                batch_idxs = sample_class_balanced_indices(
                    class_counts=self.class_counts, class_indexes=self.class_indexes, batch_size=batch_size, seed=seed
                )
            else:
                logging.error("❌ ERROR: ")
                raise ValueError("Sampling strategy must be either 'true_random' or 'class_random'.")
        else:
            if len(custom_idxs) > self.NN_xx:
                logging.error("❌ ERROR: len(custom_idxs) must be smaller than len(idxs_dset).")
                raise ValueError("len(custom_idxs) must be smaller than len(idxs_dset).")
            batch_idxs = np.array(custom_idxs)
        batch_size = len(batch_idxs)

        # Sample xx and yy batches using batch_idxs
        xx_batch = stack_features_from_dict_flattened(self.xx, batch_idxs)
        yy_batch = stack_features_from_dict_flattened(self.yy, batch_idxs)[:,0]

        # HARDCODE SAMPLES WITH EASY SPLIT TO CHECK LEARNING CURVE
        # for ii, class_label in enumerate(self.class_labels):
        #     mask = yy_batch == class_label
        #     num_selected = np.sum(mask)
        #     if num_selected > 0:
        #         noise = np.random.normal(loc=0.0, scale=0.2, size=xx_batch[mask].shape)
        #         xx_batch[mask] = ii + noise

        # Convert to PyTorch tensors if requested
        if to_torch:
            xx_batch = torch.tensor(xx_batch, dtype=torch.float32, device=device)
            yy_batch = torch.tensor(yy_batch, dtype=torch.long, device=device)

        if return_batch_idxs:
            return xx_batch, yy_batch, batch_idxs
        else:
            return xx_batch, yy_batch