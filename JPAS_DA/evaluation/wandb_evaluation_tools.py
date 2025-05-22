import numpy as np
import logging
import torch
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

def load_and_plot_sorted_sweeps(
    root_path_load: str,
    max_runs_to_plot: int = 5
):
    """
    Loads sweep results from a directory and a register file,
    finds the intersection between directory names and registered sweeps,
    sorts them by validation loss, and plots the training curves of the best runs.

    Parameters
    ----------
    root_path_load : str
        Path where sweep directories and the register file are located.
    max_runs_to_plot : int, optional
        Maximum number of best runs to plot, by default 5.

    Returns
    -------
    sorted_list_sweep_names : List[str]
        List of sweep names sorted by loss.
    sorted_losses : np.ndarray
        Array of losses sorted in ascending order.
    """

    logging.info(f"🔍 Scanning sweep folders in: {root_path_load}")

    # Step 1: Get all sweep directories
    try:
        sweep_dirs = next(os.walk(root_path_load))[1]
        logging.info(f"📁 Found {len(sweep_dirs)} sweep directories.")
    except Exception as e:
        logging.error(f"❌ Failed to list directories in {root_path_load}: {e}")
        raise

    # Step 2: Read register.txt from each and extract min val loss
    sweep_losses = []
    valid_sweeps = []

    for sweep_name in sweep_dirs:
        sweep_path = os.path.join(root_path_load, sweep_name, "register.txt")
        if not os.path.isfile(sweep_path):
            logging.warning(f"⚠️ Skipping {sweep_name}: missing register.txt")
            continue

        try:
            losses = np.loadtxt(sweep_path)
            if losses.ndim != 2 or losses.shape[1] < 2:
                logging.warning(f"⚠️ Skipping {sweep_name}: register.txt malformed")
                continue
            min_val_loss = np.min(losses[:, 1])
            sweep_losses.append(min_val_loss)
            valid_sweeps.append(sweep_name)
        except Exception as e:
            logging.warning(f"⚠️ Skipping {sweep_name}: error reading file - {e}")

    if not valid_sweeps:
        logging.error("❌ No valid sweeps with readable register.txt found.")
        return [], np.array([])

    sweep_losses = np.array(sweep_losses)

    # Step 3: Sort by minimum validation loss
    sorted_indices = np.argsort(sweep_losses)
    sorted_list_sweep_names = [valid_sweeps[i] for i in sorted_indices]
    sorted_losses = sweep_losses[sorted_indices]

    # Plot training/validation loss and learning rate for best runs
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True,
        gridspec_kw={'hspace': 0.15, 'height_ratios': [3, 1]}
    )

    cmap = plt.get_cmap("tab10")
    custom_lines = []
    custom_labels = []
    for i, sweep_name in enumerate(sorted_list_sweep_names[:max_runs_to_plot]):
        sweep_path = os.path.join(root_path_load, sweep_name, "register.txt")
        if not os.path.isfile(sweep_path):
            logging.warning(f"⚠️ Missing register file for sweep {sweep_name}")
            continue

        data = np.loadtxt(sweep_path)
        color = cmap(i % 10)

        axes[0].plot(data[:, 0], lw=2, color=color)
        axes[0].plot(data[:, 1], lw=2, ls='--', color=color)
        axes[1].plot(data[:, 2], lw=2, color=color)

        custom_lines.append(mpl.lines.Line2D([0], [0], color=color, ls='-', lw=2))
        custom_labels.append(f"{sweep_name} (Loss: {sorted_losses[i]:.3f})")

        # Add a point indicating the position of the minimum validation loss
        min_idx = np.argmin(data[:, 1])  # epoch
        min_val = data[min_idx, 1]
        axes[0].scatter(min_idx, min_val, color=color, edgecolor='black', s=60, zorder=5)

        # Annotate the point with epoch and loss
        axes[0].annotate(
            f"Min @ {min_idx}\n{min_val:.4f}",
            xy=(min_idx, min_val),
            xytext=(min_idx - 1, min_val * 1.05),
            fontsize=11,
            color=color,
            arrowprops=dict(arrowstyle="->", lw=1.2, color=color),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1)
            )

    legend = axes[0].legend(custom_lines, custom_labels, loc='upper right', fancybox=True, shadow=True, fontsize=14)
    axes[0].add_artist(legend)

    axes[0].set_ylabel("Loss", fontsize=16)
    axes[0].tick_params(axis='both', labelsize=12)

    axes[1].set_xlabel("Epochs", fontsize=16)
    axes[1].set_ylabel("LR", fontsize=16)
    axes[1].tick_params(axis='both', labelsize=12)

    custom_lines = [
        mpl.lines.Line2D([0], [0], color='grey', ls='-', lw=4),
        mpl.lines.Line2D([0], [0], color='grey', ls='--', lw=4),
    ]
    custom_labels = ["Train Loss", "Validation Loss"]
    legend = axes[0].legend(custom_lines, custom_labels, loc='upper left', fancybox=True, shadow=True, fontsize=14)
    axes[0].add_artist(legend)

    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.show()

    logging.info("✅ Plotted best runs.")
    return sorted_list_sweep_names, sorted_losses