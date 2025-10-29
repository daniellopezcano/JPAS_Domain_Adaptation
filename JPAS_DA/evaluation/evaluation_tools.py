import numpy as np
import logging
import torch

from copy import deepcopy
import os
import sys
import re
from pathlib import Path
import pandas as pd

from math import ceil

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib import ticker as mticker
from matplotlib import colors as mcolors

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc as sk_auc

from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

from JPAS_DA import global_setup
from JPAS_DA.data import loading_tools, cleaning_tools, crossmatch_tools, process_dset_splits
from JPAS_DA.models import model_building_tools
from JPAS_DA.training import save_load_tools
from JPAS_DA.wrapper_wandb import wrapper_tools
from JPAS_DA.utils.plotting_utils import get_N_colors

from typing import Dict, Optional, Any

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

def tsne_per_key(
    features: Dict[str, np.ndarray],
    *,
    standardize: bool = False,
    subsample: Optional[Dict[str, int]] = None,   # e.g. {"A": 2000, "B": 2000}
    random_state: int = 42,
    tsne_kwargs: Optional[Dict[str, Any]] = None,
    return_all_key: Optional[str] = None,         # e.g. "ALL_tSNE" to also store the stacked embedding
) -> Dict[str, np.ndarray]:
    """
    Compute a single shared 2D t-SNE embedding for all arrays in `features`,
    then split back and return a dict with the same keys plus "<key>_tSNE".

    Parameters
    ----------
    features : dict[str, ndarray]
        Each value must be a 2D array (N_i, D).
    standardize : bool
        If True, z-score the stacked features column-wise before t-SNE.
    subsample : dict[str, int], optional
        If provided, randomly keep only `n` rows for the given key before stacking.
    random_state : int
        RNG seed for reproducibility.
    tsne_kwargs : dict
        Extra kwargs forwarded to sklearn.manifold.TSNE.
    return_all_key : str or None
        If not None, also store the full stacked embedding under this key.

    Returns
    -------
    out : dict[str, ndarray]
        Original items are preserved. For each input key `k`, an additional
        key `f"{k}_tSNE"` with shape (N_k_used, 2) is added. If `return_all_key`
        is set, an extra (N_total_used, 2) array is included under that name.
    """
    if not features:
        raise ValueError("`features` is empty.")

    # Keep insertion order of dict
    keys = list(features.keys())
    rng = np.random.RandomState(random_state)

    # Validate shapes + optional subsample
    per_key_arrays = []
    counts = []
    kept_indices = {}  # map key -> indices kept (within that key)

    for k in keys:
        X = np.asarray(features[k])
        if X.ndim != 2:
            raise ValueError(f"Value for key '{k}' must be 2D, got shape {X.shape}.")

        if subsample and k in subsample and subsample[k] < X.shape[0]:
            idx = rng.choice(X.shape[0], size=int(subsample[k]), replace=False)
            X = X[idx]
            kept_indices[k] = idx
        else:
            kept_indices[k] = np.arange(X.shape[0])

        per_key_arrays.append(X)
        counts.append(X.shape[0])

    # Stack
    X_all = np.vstack(per_key_arrays)

    # Optional standardization
    if standardize:
        mu = X_all.mean(axis=0, keepdims=True)
        sigma = X_all.std(axis=0, keepdims=True)
        sigma[sigma == 0.0] = 1.0
        X_all = (X_all - mu) / sigma

    # Prepare t-SNE kwargs with safe defaults
    n_total = X_all.shape[0]
    base = dict(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        perplexity=min(100, max(5, (n_total - 1) // 3)),  # must be < n_samples
        verbose=1,
    )
    if tsne_kwargs:
        base.update(tsne_kwargs)
        # Clamp perplexity to a valid value
        base["perplexity"] = min(base.get("perplexity", base["perplexity"]),
                                 max(5, n_total - 1))

    logging.info(f"t-SNE on N={n_total} (perplexity={base['perplexity']})...")
    X_all_emb = TSNE(**base).fit_transform(X_all)

    # Split back
    out = dict(features)  # keep original entries
    start = 0
    for k, c in zip(keys, counts):
        end = start + c
        out[f"{k}_tSNE"] = X_all_emb[start:end]
        start = end

    if return_all_key:
        out[return_all_key] = X_all_emb

    # Optionally, you could also return kept_indices if you need to map back
    return out

def radar_plot(
    dict_radar: dict,
    class_names,
    *,
    title: str = None,
    title_pad: float = 20,
    figsize=(8, 8),
    theta_offset=np.pi/2,
    theta_direction=-1,
    r_ticks=(0.1, 0.3, 0.5, 0.7, 0.9),
    r_lim=(0.0, 1.0),
    tick_labelsize=17,
    radial_labelsize=12,
    linewidth_default=2.0,
    show_legend=True,
    legend_kwargs={
        "loc": "upper left", "bbox_to_anchor": (0.73, 1.0), "fontsize": 15, "ncol": 1,
        "title": None, "frameon": True, "fancybox": True, "shadow": True, "borderaxespad": 0.0,
    },
    close_line=True,
    fill_default=True,
    fill_alpha_default=0.18,
):
    """
    Flexible radar-plot for per-class F1 scores across multiple cases.

    Legend labels are taken from the dict keys in `dict_radar` (case_name).
    Any 'label' inside payload['plot_kwargs'] is ignored/overridden.

    Per-series fill overrides in payload['plot_kwargs']:
      - 'fill': bool (default: fill_default)
      - 'fill_alpha': float (default: fill_alpha_default)
      - 'fill_color': matplotlib color (default: series line color)
    """
    class_names = list(class_names)
    C = len(class_names)
    labels_range = np.arange(C)

    # Angles per class
    angles = np.linspace(0, 2*np.pi, C, endpoint=False).tolist()
    angles_closed = angles + angles[:1] if close_line else angles

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.set_theta_offset(theta_offset)
    ax.set_theta_direction(theta_direction)

    # Class tick positions/labels
    tick_angles = angles
    tick_degs = np.degrees(tick_angles)
    ax.set_thetagrids(tick_degs, class_names, fontsize=tick_labelsize)

    ax.set_ylim(*r_lim)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([str(v) for v in r_ticks], fontsize=radial_labelsize)
    ax.grid(True, alpha=0.3)

    handles = []

    for case_name, payload in dict_radar.items():
        y_true = np.asarray(payload["y_true"])
        y_pred = np.asarray(payload["y_pred"])

        # Predicted class indices
        if y_pred.ndim == 2:
            if y_pred.shape[1] != C:
                raise ValueError(f"For '{case_name}', y_pred has {y_pred.shape[1]} columns but len(class_names)={C}.")
            y_hat = np.argmax(y_pred, axis=1)
        elif y_pred.ndim == 1:
            y_hat = y_pred
        else:
            raise ValueError(f"For '{case_name}', y_pred must be 1D or 2D; got shape {y_pred.shape}.")

        # Per-class F1 in fixed order 0..C-1
        f1 = f1_score(y_true, y_hat, average=None, labels=labels_range, zero_division=0)
        f1_closed = (f1.tolist() + [f1[0]]) if close_line else f1.tolist()

        # Style (force legend label to dict key)
        pk = dict(payload.get("plot_kwargs", {}))
        if "linstyle" in pk and "linestyle" not in pk:
            pk["linestyle"] = pk.pop("linstyle")
        pk.setdefault("linewidth", linewidth_default)
        pk["label"] = str(case_name)

        color = pk.get("color", None)
        marker = pk.get("marker", None)
        markersize = float(pk.get("markersize", 8.0))
        alpha = pk.get("alpha", 1.0)

        # Fill overrides (per-series)
        fill_on = bool(pk.pop("fill", fill_default))
        fill_alpha = float(pk.pop("fill_alpha", fill_alpha_default))
        fill_color = pk.pop("fill_color", None)

        # Plot line (legend handle)
        line = ax.plot(angles_closed, f1_closed, **pk)[0]
        handles.append(line)

        # Fill under curve (ensure closed polygon for fill even if close_line=False)
        if fill_on:
            face = fill_color if fill_color is not None else (color if color is not None else line.get_color())
            if close_line:
                angles_fill = angles_closed
                f1_fill = f1_closed
            else:
                angles_fill = angles + angles[:1]
                f1_fill = f1.tolist() + [f1[0]]
            ax.fill(
                angles_fill, f1_fill,
                color=face, alpha=fill_alpha,
                zorder=line.get_zorder() - 1,
                label="_nolegend_",
            )

        # Optional vertex markers (kept out of legend)
        if marker is not None:
            s = markersize ** 2
            mfc = pk.get("markerfacecolor", color)
            mec = pk.get("markeredgecolor", color)
            mew = pk.get("markeredgewidth", 1.0)
            ax.scatter(
                tick_angles, f1,
                s=s, marker=marker,
                facecolors=mfc if mfc is not None else "none",
                edgecolors="k",
                linewidths=mew,
                alpha=alpha,
                zorder=5,
                label="_nolegend_",
            )

    # Rotate class-name tick labels tangentially
    offset_deg = np.degrees(theta_offset)
    for lbl, base_deg in zip(ax.get_xticklabels(), tick_degs):
        ang_disp = (base_deg * theta_direction + offset_deg) % 360.0
        if 90.0 < ang_disp < 270.0:
            rotation, ha = ang_disp + 180.0, "right"
        else:
            rotation, ha = ang_disp, "left"
        lbl.set_rotation(rotation)
        lbl.set_rotation_mode("anchor")
        lbl.set_horizontalalignment(ha)
        lbl.set_verticalalignment("center")

    # Title / legend
    if title:
        ax.set_title(title, pad=title_pad, fontsize=16)
    if show_legend:
        legend_kwargs = dict(loc="upper right", frameon=True) | (legend_kwargs or {})
        ax.legend(handles=handles, **legend_kwargs)

    plt.tight_layout()
    return fig, ax

def compare_model_parameters(model1, model2, rtol=1e-5, atol=1e-8):
    """Compare parameters of two PyTorch models with optional tolerance.
    
    Returns True if all parameters match, False otherwise.
    """
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    if sd1.keys() != sd2.keys():
        print("❌ Model parameter keys do not match.")
        return False

    all_match = True

    for k in sd1.keys():
        p1 = sd1[k]
        p2 = sd2[k]
        if not torch.allclose(p1, p2, rtol=rtol, atol=atol):
            print(f"❌ Mismatch in parameter: {k}")
            all_match = False

    if all_match:
        print("✅ All parameters match.")
    else:
        print("⚠️ Some parameters differ.")

    return all_match

def plot_confusion_matrix(
    yy_true,
    yy_pred_P,
    class_names,
    normalize=True,
    figsize=(10, 7),
    cmap='RdBu',
    title=None,
    threshold_color = 0.5
):
    """
    Plots a confusion matrix with counts, percentages, and class-wise metrics.

    Parameters:
    ----------
    - yy_true: np.ndarray of true class labels
    - yy_pred_P: np.ndarray of predicted class probabilities (softmax output)
    - class_names: list of class names in correct order
    - normalize: bool, normalize rows or not
    - figsize: figure size
    - cmap: colormap (string or Colormap object)
    """
    yy_pred = np.argmax(yy_pred_P, axis=1)
    
    # Get unique classes in yy_true
    unique_classes = np.unique(yy_true)
    num_classes = len(class_names)

    logging.debug(f"Unique classes in yy_true: {unique_classes}")
    logging.debug(f"Number of classes in class_names: {num_classes}")

    # Ensure the number of classes in yy_true does not exceed class_names
    missing_classes = np.arange(num_classes)[~np.isin(np.arange(num_classes), np.array(unique_classes))]
    if missing_classes.size > 0:
        logging.warning(f"Missing classes in the confusion matrix: {missing_classes}")

    # If any class is missing from the true labels, create a zero matrix for those classes
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(yy_true, yy_pred):
        if t in unique_classes:  # Only update the confusion matrix for existing classes in yy_true
            cm[int(t), int(p)] += 1

    # Normalize the confusion matrix by row sums
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, where=row_sums != 0)

    logging.debug(f"Raw confusion matrix (cm):\n{cm}")

    # Normalize the cm_percent to span from 0 to 1 for colormap
    norm = Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=cmap, norm=norm)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=11)

    # Labels
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, fontsize=13)
    ax.set_yticklabels(class_names, fontsize=13)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=16)
    else:
        ax.set_title('Confusion Matrix', fontsize=16)

    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    # Compute precision, recall, F1 manually from confusion matrix
    precision = []
    recall = []
    f1 = []
    for i in range(num_classes):
        tp = cm[i, i]  # True positives
        fp = cm[:, i].sum() - tp  # False positives
        fn = cm[i, :].sum() - tp  # False negatives
        tn = cm.sum() - (tp + fp + fn)  # True negatives
        
        # Precision, Recall, and F1 calculations with handling for missing classes
        if tp + fp > 0:
            precision_i = tp / (tp + fp)
        else:
            precision_i = 0.0
        
        if tp + fn > 0:
            recall_i = tp / (tp + fn)
        else:
            recall_i = 0.0
        
        if precision_i + recall_i > 0:
            f1_i = 2 * precision_i * recall_i / (precision_i + recall_i)
        else:
            f1_i = 0.0
        
        precision.append(precision_i)
        recall.append(recall_i)
        f1.append(f1_i)
    logging.debug(f"Precision: {precision}")
    logging.debug(f"Recall: {recall}")
    logging.debug(f"F1 Score: {f1}")
    
    # Plot confusion matrix
    for i in range(num_classes):
        for j in range(num_classes):
            count = cm[i, j]
            percent = cm_percent[i, j] * 100 if row_sums[i] != 0 else 0
            text_color = "white" if cm_percent[i, j] > threshold_color else "black"
            if i == j:
                # Display precision, recall, F1 on the diagonal
                text = str(count) + "\nTPR:" + str(round(recall[i] * 100, 1)) + "%" + \
                       "\nPPV:" + str(round(precision[i] * 100, 1)) + "%" + \
                       "\nF1:" + str(round(f1[i], 2))
                ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')
            else:
                text = f"{count}\n{percent:.1f}%"
                ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=11)

    plt.tight_layout()
    plt.show()

    return cm

def compute_ece(y_true, y_pred_P, n_bins=10):
    y_conf = np.max(y_pred_P, axis=1)
    y_pred = np.argmax(y_pred_P, axis=1)
    correct = (y_pred == y_true)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_conf > bin_edges[i]) & (y_conf <= bin_edges[i + 1])
        if np.any(mask):
            acc = np.mean(correct[mask])
            conf = np.mean(y_conf[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(y_true)
    return ece

def plot_multiclass_rocs(
    *,
    dict_cases,                      # {case: {"y_true": (N,), "y_pred": (N,C), "plot_kwargs": {...}}}
    class_names=None,               # list[str] in y_pred column order; if None -> inferred from labels
    cmap="plasma",                  # <- NEW: any Matplotlib colormap name or Colormap object
    class_legend_loc="lower right",
    case_legend_loc="lower left",
    figsize=(8, 8),
    title=None,
    legend_fontsize=18,
    # --- AUC text box options ---
    draw_auc_text=True,
    auc_fontsize=18,
    auc_box_facecolor="white",
    auc_box_alpha=0.9,
    auc_text_dx=0.0,
    auc_text_dy=0.0,
    auc_text_spread=0.0,
    # --- axes & cosmetics ---
    x_lims=(0.0, 1.0),
    y_lims=(0.0, 1.0),
    x_label="False Positive Rate",
    y_label="True Positive Rate",
    diagonal_kwargs=None,
    linewidth_default=2.0,
    marker_every=None,
):
    """
    Single-panel multi-class ROC overlay.

    Colors are assigned PER CLASS from the provided colormap (`cmap`), while line
    style/markers come from each case's `plot_kwargs`. Two legends are shown:
      - Classes (color): one entry per class using the colormap.
      - Cases (style): black handles showing linestyle/marker/linewidth for each case.
    """
    if not dict_cases:
        raise ValueError("dict_cases must contain at least one case.")

    case_names = list(dict_cases.keys())

    # Infer classes
    if class_names is None:
        all_y = np.concatenate([np.asarray(v["y_true"]) for v in dict_cases.values()])
        classes = np.unique(all_y)
        class_names_used = [str(c) for c in classes]
    else:
        class_names_used = list(class_names)
        classes = np.arange(len(class_names_used))
    C = len(classes)

    # Validate shapes
    for nm, payload in dict_cases.items():
        y_pred = np.asarray(payload["y_pred"])
        if y_pred.ndim != 2 or y_pred.shape[1] != C:
            raise ValueError(f"[{nm}] y_pred must be (N,{C}). Got {y_pred.shape}.")

    # --- Build per-CLASS colors from the colormap ---
    # Use the first C colors from Matplotlib's default color cycle (respects current style)
    default_cycle = list(mcolors.TABLEAU_COLORS.values())

    # Build RGBA list; wrap around if C > len(default cycle)
    cls_colors_list = [mcolors.to_rgba(default_cycle[i % len(default_cycle)]) for i in range(C)]

    # Figure
    fig, ax = plt.subplots(figsize=figsize)

    # Diagonal baseline
    diag_kws = {"color": "0.85", "ls": "--", "lw": 1.0}
    if isinstance(diagonal_kwargs, dict):
        diag_kws.update(diagonal_kwargs)
    ax.plot([0, 1], [0, 1], **diag_kws, zorder=1)

    # Helper for marker thinning
    def _markevery(n):
        if marker_every is None:
            return None
        if isinstance(marker_every, int) and marker_every > 1:
            return marker_every
        if isinstance(marker_every, float) and 0 < marker_every < 1:
            return max(1, int(n * marker_every))
        return None

    # Compute & plot ROC curves
    curves = []
    for i_cls, cls in enumerate(classes):
        for case in case_names:
            payload = dict_cases[case]
            y_true = np.asarray(payload["y_true"]).astype(int)
            y_pred = np.asarray(payload["y_pred"]).astype(float)
            y_score = y_pred[:, i_cls]
            y_bin = (y_true == cls).astype(int)

            # style from case
            pkw = dict(payload.get("plot_kwargs", {}))
            ls = pkw.get("linestyle", "-")
            lw = pkw.get("linewidth", linewidth_default)
            marker = pkw.get("marker", None)
            ms = pkw.get("markersize", 6)
            label_case = pkw.get("label", case)

            color = cls_colors_list[i_cls]

            # ROC (skip if not computable)
            try:
                fpr, tpr, _ = roc_curve(y_bin, y_score)
            except Exception:
                continue
            AUC = sk_auc(fpr, tpr)

            me = _markevery(len(fpr))
            ax.plot(
                fpr, tpr,
                color=color,
                linestyle=ls,
                linewidth=lw,
                marker=marker,
                markersize=ms,
                markevery=me,
                label=f"{label_case} - {class_names_used[i_cls]}",
                zorder=3,
            )

            curves.append({
                "auc": AUC,
                "fpr": fpr, "tpr": tpr,
                "class_idx": i_cls,
                "class_name": class_names_used[i_cls],
                "case": label_case,
                "color": color,
                "linestyle": ls,
                "linewidth": lw,
            })

    # Axes cosmetics
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    # Legends
    # Classes legend (colors from cmap)
    class_handles = [
        Line2D([0], [0], color=cls_colors_list[i], lw=3, ls='-', label=class_names_used[i])
        for i in range(C)
    ]
    leg_classes = ax.legend(handles=class_handles, loc=class_legend_loc,
                            fontsize=legend_fontsize, title="Classes")
    ax.add_artist(leg_classes)

    # Cases legend — black handles showing each case's style
    case_handles = []
    for case in case_names:
        pkw = dict(dict_cases[case].get("plot_kwargs", {}))
        ls = pkw.get("linestyle", "-")
        lw = pkw.get("linewidth", linewidth_default)
        marker = pkw.get("marker", None)
        ms = pkw.get("markersize", 6)
        label_case = pkw.get("label", case)
        case_handles.append(
            Line2D([0], [0], color="black", lw=lw, ls=ls, marker=marker, markersize=ms, label=label_case)
        )
    ax.legend(handles=case_handles, loc=case_legend_loc, fontsize=legend_fontsize, title="Cases")

    # AUC text boxes at the point furthest from (1,0), with per-curve styling
    if draw_auc_text and curves:
        curves.sort(key=lambda d: (d["class_idx"], d["case"]))
        for k, rec in enumerate(curves):
            fpr, tpr = rec["fpr"], rec["tpr"]
            n = len(fpr)
            if n == 0:
                continue
            d2 = (fpr - 1.0) ** 2 + (tpr - 0.0) ** 2
            anchor_idx = int(np.argmax(d2))
            step = max(1, int(auc_text_spread * n))
            idx = int(np.clip(anchor_idx + k * step, 0, n - 1))
            tx = float(fpr[idx]) + float(auc_text_dx)
            ty = float(tpr[idx]) + float(auc_text_dy)
            bbox = dict(
                facecolor=auc_box_facecolor,
                alpha=auc_box_alpha,
                edgecolor=rec["color"],
                boxstyle="round,pad=0.2",
                linestyle=rec["linestyle"],
                linewidth=max(1.2, rec["linewidth"] * 0.8),
            )
            ax.text(tx, ty, f"AUC={rec['auc']:.3f}", fontsize=auc_fontsize, color=rec["color"],
                    ha="left", va="bottom", bbox=bbox)

    return fig, ax

def plot_confusion_matrices_grid(
    *,
    dict_cases,                         # {case_label: {"y_true": (N,), "y_pred": (N,C)}}
    class_names,                        # list[str], length C
    # --- Figure / layout ---
    figsize=(18, 16),
    dpi=150,
    nrows=2,
    ncols=None,                         # if None, computed from number of cases
    constrained_layout=True,
    grid_wspace=0.0,
    grid_hspace=0.0,
    # --- Normalization & colormap ---
    normalize="row",                    # "row" | "col" | "none"
    cmap="RdYlGn",
    vmin=0,                          # None -> auto (0..1 for normalized; 0..max count for 'none')
    vmax=1,
    threshold_color=0.5,                # threshold for text color switch (normalized values)
    # --- Text & fonts ---
    fs_title=24,
    fs_label=24,
    fs_ticks=24,
    fs_cell=20,
    fs_cell_diag=18,
    fs_cbar_label=24,
    fs_cbar_ticks=24,
    tick_rotation=20,
    # --- Annotations & options ---
    annotate_offdiag=True,              # draw counts + normalized % on off-diagonal cells
    annotate_diagonal=True,             # draw TPR/PPV/F1 on the diagonal
    percent_decimals=1,                 # decimals for % values
    show_outer_labels=True,             # only show axes labels on outer edges
    # --- Panel titles ---
    panel_title_inside=False,           # if True, draw title inside panel (top-left)
    panel_title_bbox=True,
    panel_title_bbox_fc="white",
    panel_title_bbox_alpha=0.85,
    panel_title_color="black",
    # --- Colorbar ---
    add_colorbar=True,
    cbar_orientation="vertical",        # "vertical" or "horizontal"
    cbar_pad_fraction=0.05,             # width (if vertical) or height (if horizontal) fraction for cbar
    cbar_ticks=(0.0, 0.25, 0.5, 0.75, 1.0),
    cbar_label="True-label (row) normalized ratio",
    # --- Output ---
    save_path=None                      # e.g. "confusion_matrices.pdf"
):
    """
    Plot a grid of confusion matrices for multiple cases, with a shared colorbar.

    dict_cases: {
        "Case label A": {"y_true": (N,), "y_pred": (N,C)},  # y_pred are probabilities or scores
        "Case label B": {...},
        ...
    }
    class_names: list of C class names in the probability column order of y_pred.

    Normalization:
      - "row": each row (true class) sums to 1 (default).
      - "col": each column (pred class) sums to 1.
      - "none": raw counts (shared vmax auto-computed if not given).
    """
    if not dict_cases:
        raise ValueError("dict_cases must contain at least one case.")
    case_labels = list(dict_cases.keys())
    n_cases = len(case_labels)
    C = len(class_names)

    # Validate shapes
    for name, payload in dict_cases.items():
        y_true = np.asarray(payload["y_true"])
        y_pred = np.asarray(payload["y_pred"])
        if y_pred.ndim != 2 or y_pred.shape[1] != C:
            raise ValueError(f"[{name}] y_pred must be (N,{C}). Got {y_pred.shape}.")

    # Layout
    if ncols is None:
        ncols = int(np.ceil(n_cases / nrows))
    # Colorbar occupies an extra column (if vertical) or row (if horizontal)
    if add_colorbar:
        if cbar_orientation == "vertical":
            width_ratios = [1]*ncols + [cbar_pad_fraction]
            height_ratios = [1]*nrows
            total_cols = ncols + 1
            total_rows = nrows
            cbar_spec = ("right", slice(None))  # all rows, last column
        else:
            width_ratios = [1]*ncols
            height_ratios = [1]*nrows + [cbar_pad_fraction]
            total_cols = ncols
            total_rows = nrows + 1
            cbar_spec = ("bottom", slice(None))  # last row, all columns
    else:
        width_ratios = [1]*ncols
        height_ratios = [1]*nrows
        total_cols = ncols
        total_rows = nrows
        cbar_spec = None

    # Figure & gridspec
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=constrained_layout)
    gs = fig.add_gridspec(
        total_rows, total_cols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=grid_wspace, hspace=grid_hspace
    )

    # Create axes for panels
    axes = []
    for idx in range(n_cases):
        r = idx // ncols
        c = idx % ncols
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)
    axes = np.array(axes, dtype=object)

    # Color normalization
    if normalize in ("row", "col"):
        norm = mpl.colors.Normalize(vmin=0.0 if vmin is None else vmin,
                                    vmax=1.0 if vmax is None else vmax)
    else:
        # compute shared max count if needed
        max_count = 0
        for nm in case_labels:
            y_true = np.asarray(dict_cases[nm]["y_true"]).astype(int)
            y_pred = np.argmax(dict_cases[nm]["y_pred"], axis=1).astype(int)
            cm_counts = np.zeros((C, C), dtype=int)
            valid = (y_true >= 0) & (y_true < C)
            for t, p in zip(y_true[valid], y_pred[valid]):
                if 0 <= t < C and 0 <= p < C:
                    cm_counts[t, p] += 1
            max_count = max(max_count, int(cm_counts.max()) if cm_counts.size else 0)
        norm = mpl.colors.Normalize(vmin=0.0 if vmin is None else vmin,
                                    vmax=float(max_count) if vmax is None else vmax)

    cmap_obj = plt.get_cmap(cmap)

    # Shared limits / ticks
    xlim = (-0.5, C - 0.5)
    ylim = (C - 0.5, -0.5)             # origin='upper' feel
    ticks = np.arange(C)

    # Shared colorbar mappable
    mappable_for_cbar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable_for_cbar.set_array([])

    # Draw each panel
    for idx, nm in enumerate(case_labels):
        ax = axes[idx]
        payload = dict_cases[nm]
        y_true = np.asarray(payload["y_true"]).astype(int)
        y_pred = np.argmax(payload["y_pred"], axis=1).astype(int)

        # Confusion matrix (counts)
        cm_counts = np.zeros((C, C), dtype=int)
        valid = (y_true >= 0) & (y_true < C)
        for t, p in zip(y_true[valid], y_pred[valid]):
            if 0 <= t < C and 0 <= p < C:
                cm_counts[t, p] += 1

        # Normalized for display
        if normalize == "row":
            denom = cm_counts.sum(axis=1, keepdims=True)
            cm_display = np.divide(cm_counts, denom, where=(denom != 0))
        elif normalize == "col":
            denom = cm_counts.sum(axis=0, keepdims=True)
            cm_display = np.divide(cm_counts, denom, where=(denom != 0))
        else:
            cm_display = cm_counts.astype(float)

        # Heatmap
        im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap_obj, norm=norm, origin='upper')

        # Axes styling
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, fontsize=fs_ticks)
        ax.set_yticklabels(class_names, fontsize=fs_ticks)
        ax.set_aspect('equal', adjustable='box')

        # Panel title
        if panel_title_inside:
            bbox = dict(boxstyle="round,pad=0.3", facecolor=panel_title_bbox_fc,
                        alpha=panel_title_bbox_alpha, edgecolor="none") if panel_title_bbox else None
            ax.text(0.01, 0.98, nm, transform=ax.transAxes, ha="left", va="top",
                    fontsize=fs_title, color=panel_title_color, bbox=bbox)
        else:
            ax.set_title(nm, fontsize=fs_title, pad=20)

        # Per-class diag metrics: TPR/PPV/F1
        if annotate_diagonal or annotate_offdiag:
            precision = np.zeros(C, dtype=float)
            recall    = np.zeros(C, dtype=float)
            f1        = np.zeros(C, dtype=float)
            for i in range(C):
                tp = cm_counts[i, i]
                fp = cm_counts[:, i].sum() - tp
                fn = cm_counts[i, :].sum() - tp
                precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1[i]        = (2 * precision[i] * recall[i] / (precision[i] + recall[i])
                                if (precision[i] + recall[i]) > 0 else 0.0)

        # Cell annotations
        pct_fmt = f"{{:.{percent_decimals}f}}%"
        for i in range(C):
            for j in range(C):
                val = cm_display[i, j]
                # choose text color based on normalized value (map to [0,1] for threshold)
                # If normalize == 'none', map counts to [0,1] using norm:
                vis_level = float((val - norm.vmin) / (norm.vmax - norm.vmin + 1e-12))
                text_color = "white" if vis_level > threshold_color else "black"

                if i == j and annotate_diagonal:
                    tp = cm_counts[i, i]
                    text = (f"{tp}\n"
                            f"TPR:{(recall[i]*100):.{percent_decimals}f}%"
                            f"\nPPV:{(precision[i]*100):.{percent_decimals}f}%"
                            f"\nF1:{f1[i]:.2f}")
                    ax.text(j, i, text, ha="center", va="center",
                            color=text_color, fontsize=fs_cell_diag, fontweight='bold', linespacing=1.2)
                elif i != j and annotate_offdiag:
                    count = cm_counts[i, j]
                    if normalize == "row":
                        denom = cm_counts[i, :].sum()
                        percent = (count / denom * 100) if denom > 0 else 0.0
                    elif normalize == "col":
                        denom = cm_counts[:, j].sum()
                        percent = (count / denom * 100) if denom > 0 else 0.0
                    else:
                        # Still show row-based percent for interpretability
                        denom = cm_counts[i, :].sum()
                        percent = (count / denom * 100) if denom > 0 else 0.0
                    text = f"{count}\n{pct_fmt.format(percent)}"
                    ax.text(j, i, text, ha="center", va="center",
                            color=text_color, fontsize=fs_cell, fontweight='bold', linespacing=1.2)

        # Outer labels only
        if show_outer_labels:
            r, c = divmod(idx, ncols)
            if c == 0:
                ax.set_ylabel("True Label", fontsize=fs_label, labelpad=6)
            else:
                ax.set_ylabel("")
                for lab in ax.get_yticklabels():
                    lab.set_visible(False)
            if r == (nrows - 1):
                ax.set_xlabel("Predicted Label", fontsize=fs_label, labelpad=6)
                for lab in ax.get_xticklabels():
                    lab.set_rotation(tick_rotation)
                    lab.set_ha("right")
                    lab.set_rotation_mode("anchor")
            else:
                ax.set_xlabel("")
                for lab in ax.get_xticklabels():
                    lab.set_visible(False)

    # Colorbar axis
    if add_colorbar:
        if cbar_spec[0] == "right":
            cax = fig.add_subplot(gs[:, -1])
        else:
            cax = fig.add_subplot(gs[-1, :])

        cbar = fig.colorbar(mappable_for_cbar, cax=cax, orientation=cbar_orientation)
        if cbar_orientation == "vertical":
            cbar.set_label(cbar_label, fontsize=fs_cbar_label)
        else:
            cbar.set_label(cbar_label, fontsize=fs_cbar_label)
        cbar.ax.tick_params(labelsize=fs_cbar_ticks)
        # Pick sensible ticks if 'none' normalization
        if normalize in ("row", "col"):
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)
        else:
            # counts scale: let Matplotlib choose, unless provided
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes

def multiclass_brier_score(y_true, y_prob, n_classes=None):
    if n_classes is None:
        n_classes = y_prob.shape[1]
    y_true_one_hot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1))

def compare_models_performance(
    *,
    dict_cases,                         # {case_name: {"y_true": (N,), "y_pred": (N,C), "plot_kwargs": {...}}}
    class_names=None,                   # list[str] in y_pred column order; if None, inferred from labels
    title=None,
    figsize=(5, 24),
    palette=None,                       # optional list of colors for cases (fallback to tab10)
    save_path=None,                     # save figure (suffix decides format)
    include_metrics=("Accuracy", "Macro F1", "Macro TPR", "Macro Precision", "Macro AUROC", "ECE", "Brier Score"),
    # Layout (can be multi-row; x-axis shared across rows)
    nrows=7,
    subplot_hspace=-1.166,
    subplot_wspace=0.25,
    # Per-subpanel y-ranges
    y_ranges=None,                      # dict: {metric_name: (ymin, ymax)}; others auto-scaled
    y_margin_frac=0.07,
    # Bars & annotations
    bar_alpha=1.0,
    bar_edgecolor="black",
    bar_width=0.7,                      # width for one bar per case in each subplot
    annotate_values=True,
    value_label_fontsize=12,
    # Axes cosmetics
    ylabel_text="Score",                # only shown on the FIRST subplot
    left_margin=0.12,
    # Tick formatting
    ytick_step=None,                    # used if ytick_count is None
    ytick_format="{x:.2f}",
    two_line_xticklabels=True,          # break each case label into two lines
    # NEW: y-ticks by count (overrides ytick_step when provided)
    ytick_count=3,                   # e.g., 5 -> exactly 5 ticks via LinearLocator
    # Best-line preferences
    metric_best_high_low={"ECE": "low",  "Brier Score": "low"}, # others default to "high"
    best_line_kwargs={"ls": "--", "lw": 1.8, "alpha": 0.9}, # "color" intentionally omitted; it’s set to the winner’s bar color
    # Title-as-text-in-axes options
    metric_title_fontsize=22,
    metric_title_bbox={"facecolor": "white", "alpha": 0.9, "boxstyle": "round,pad=0.2", "edgecolor": "k"},
    # ===========================
    # NEW: print a metrics table
    # ===========================
    show_table=True,
):
    """
    Multi-subpanel comparison of global metrics. One subplot per metric, each with its own y-axis.
    - Titles are drawn as top-left in-axes text (no Axes.set_title).
    - Optional ytick_count to control number of y ticks (else step/format used).
    - Best-performing case marked with a horizontal line in that case's bar color.
    """
    if not dict_cases:
        raise ValueError("dict_cases must contain at least one case.")

    case_names = list(dict_cases.keys())
    n_cases = len(case_names)

    # Determine classes and consistency
    if class_names is None:
        all_y = np.concatenate([np.asarray(v["y_true"]) for v in dict_cases.values()])
        classes = np.unique(all_y)
    else:
        classes = np.arange(len(class_names))
    C = len(classes)
    for name, payload in dict_cases.items():
        y_pred = np.asarray(payload["y_pred"])
        if y_pred.ndim != 2 or y_pred.shape[1] != C:
            raise ValueError(f"[{name}] y_pred must be (N,{C}). Got {y_pred.shape}.")

    # Fallbacks for Brier/ECE
    def _one_hot(y, C):
        oh = np.zeros((y.size, C), dtype=float)
        oh[np.arange(y.size), y.astype(int)] = 1.0
        return oh

    # Helper: preference ("high" or "low") for a metric
    def _pref(metric_name: str) -> str:
        if isinstance(metric_best_high_low, dict) and metric_name in metric_best_high_low:
            val = metric_best_high_low[metric_name].lower()
            return "low" if val.startswith("low") else "high"
        # sensible defaults
        low_is_better = {"ECE", "Brier Score"}
        return "low" if metric_name in low_is_better else "high"

    # compute metrics
    per_case_vals = {metric: [] for metric in include_metrics}
    for name, payload in dict_cases.items():
        y_true = np.asarray(payload["y_true"])
        y_pred = np.asarray(payload["y_pred"])
        y_hat = np.argmax(y_pred, axis=1)

        is_multiclass = (C > 2) or (len(np.unique(y_true)) > 2)

        acc = accuracy_score(y_true, y_hat)
        f1_macro = float(np.mean(f1_score(y_true, y_hat, average=None, zero_division=0)))
        tpr_macro = float(np.mean(recall_score(y_true, y_hat, average=None, zero_division=0)))
        prec_macro = precision_score(y_true, y_hat, average='macro', zero_division=0)

        if len(np.unique(y_true)) > 1:
            if C == 2 and not is_multiclass:
                try:
                    auroc = roc_auc_score(y_true, y_pred[:, 1])
                except Exception:
                    auroc = np.nan
            else:
                try:
                    auroc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo')
                except Exception:
                    auroc = np.nan
        else:
            auroc = np.nan

        try:
            ece = compute_ece(y_true, y_pred)
        except Exception:
            ece = np.nan
        try:
            brier = multiclass_brier_score(y_true, y_pred)
        except Exception:
            brier = np.nan

        vals_map = {
            "Accuracy": acc, "Macro F1": f1_macro, "Macro TPR": tpr_macro, "Macro Precision": prec_macro,
            "Macro AUROC": auroc, "ECE": ece, "Brier Score": brier,
        }
        for metric in include_metrics:
            per_case_vals[metric].append(vals_map[metric])

    # ===========================
    # NEW: pretty-printed table
    # ===========================
    if show_table:
        # Case display names (prefer plot label if present)
        case_labels = [dict_cases[nm].get("plot_kwargs", {}).get("label", nm) for nm in case_names]
        # Build string table
        header_cells = ["Metric"] + case_labels
        # compute column widths
        col_widths = [max(len("Metric"), max(len(m) for m in include_metrics))]
        for j in range(len(case_labels)):
            max_val_width = max(
                len(f"{v:.4f}") if (isinstance(v, (int, float)) and np.isfinite(v)) else len("n/a")
                for v in (per_case_vals[m][j] for m in include_metrics)
            )
            col_widths.append(max(len(case_labels[j]), max_val_width))
        # print header
        header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header_cells))
        sep = "-+-".join("-" * w for w in col_widths)
        print("\n=== Global Metrics Table ===")
        print(header_row)
        print(sep)
        # rows
        for m in include_metrics:
            row = [m.ljust(col_widths[0])]
            for j in range(len(case_labels)):
                v = per_case_vals[m][j]
                s = f"{v:.4f}" if (isinstance(v, (int, float)) and np.isfinite(v)) else "n/a"
                row.append(s.rjust(col_widths[j + 1]))
            print(" | ".join(row))
        print()  # trailing newline

    # figure & subpanels (share x across rows)
    M = len(include_metrics)
    ncols = ceil(M / nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    if isinstance(axes, np.ndarray):
        axes = np.array(axes).reshape(nrows, ncols)
    else:
        axes = np.array([[axes]])

    # colors per case (fallback to tab10)
    if palette is None:
        palette = [plt.cm.tab10(i % 10) for i in range(n_cases)]
    case_colors = []
    for k, nm in enumerate(case_names):
        kw = dict(dict_cases[nm].get("plot_kwargs", {}))
        case_colors.append(kw.get("color", palette[k]))

    # helper: split label into two lines at the middle space (if any)
    def _two_line(name: str) -> str:
        if not two_line_xticklabels:
            return name
        s = str(name)
        spaces = [i for i, ch in enumerate(s) if ch == " "]
        if not spaces:
            return s
        center = len(s) / 2.0
        split_idx = min(spaces, key=lambda i: abs(i - center))
        return s[:split_idx] + "\n" + s[split_idx+1:]

    xtick_labels = [_two_line(dict_cases[nm].get("plot_kwargs", {}).get("label", nm)) for nm in case_names]
    x = np.arange(n_cases)

    for idx_metric, metric in enumerate(include_metrics):
        r = idx_metric // ncols
        c = idx_metric % ncols
        ax = axes[r, c]

        vals = per_case_vals[metric]

        # y-limits (explicit or auto)
        if isinstance(y_ranges, dict) and metric in y_ranges:
            ymin, ymax = y_ranges[metric]
        else:
            finite = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
            if finite:
                ymin = float(min(0.0, np.min(finite)))
                ymax = float(max(1.0, np.max(finite)))
                if ymax - ymin < 1e-6:
                    ymax = ymin + 1.0
                margin = y_margin_frac * (ymax - ymin)
                ymin -= margin
                ymax += margin
            else:
                ymin, ymax = 0.0, 1.0

        # one bar per case
        bars = ax.bar(
            x, [v if (isinstance(v, (int, float)) and np.isfinite(v)) else 0.0 for v in vals],
            width=bar_width,
            color=case_colors,
            edgecolor=bar_edgecolor,
            alpha=bar_alpha,
        )

        # annotate values
        if annotate_values:
            for rect, val in zip(bars, vals):
                txt = "n/a" if not (isinstance(val, (int, float)) and np.isfinite(val)) else f"{val:.3f}"
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    rect.get_height() + 0.01 * (ymax - ymin),
                    txt,
                    ha="center", va="bottom", fontsize=value_label_fontsize
                )

        # y-limits
        ax.set_ylim(ymin, ymax)

        # y-label only on first subplot
        ax.set_ylabel(ylabel_text, fontsize=22)

        # x ticks: case names in two lines (only bottom row shows labels)
        dx = -0.2
        ax.set_xticks(x + dx)
        if r == nrows - 1:
            ax.set_xticklabels(xtick_labels, rotation=40, fontsize=16)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)

        # y-ticks: either fixed count or step-based
        if ytick_count is not None:
            ax.yaxis.set_major_locator(mticker.LinearLocator(ytick_count))
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(ytick_format))
        else:
            ax.yaxis.set_major_locator(mticker.MultipleLocator(ytick_step))
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(ytick_format))

        # no grid
        ax.grid(False)

        # --- TITLE AS IN-AXES TEXT (top-left) ---
        bbox = metric_title_bbox if metric_title_bbox is not None else None
        ax.text(
            0.07, 0.93, metric,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=metric_title_fontsize, bbox=bbox
        )

        # --- best horizontal line (colored by winner) ---
        vals_arr = np.array([
            v if (isinstance(v, (int, float)) and np.isfinite(v)) else np.nan
            for v in vals
        ], dtype=float)

        if not np.all(np.isnan(vals_arr)):
            best_idx = int(np.nanargmin(vals_arr)) if _pref(metric) == "low" else int(np.nanargmax(vals_arr))
            best_val = float(vals_arr[best_idx])

            line_kwargs = dict(color=case_colors[best_idx], ls="--", lw=1.8, alpha=0.9)
            if best_line_kwargs:
                tmp = line_kwargs.copy()
                tmp.update({k: v for k, v in best_line_kwargs.items() if k != "color"})
                line_kwargs = tmp
            ax.axhline(best_val, **line_kwargs)

    # hide any unused axes
    M = len(include_metrics)
    for idx_extra in range(M, nrows * ncols):
        r = idx_extra // ncols
        c = idx_extra % ncols
        axes[r, c].axis("off")

    # figure title
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    # margins & spacing
    fig.subplots_adjust(left=left_margin, hspace=subplot_hspace, wspace=subplot_wspace)
    fig.tight_layout()

    # save
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes


def compare_models_performance_per_class(
    *,
    dict_cases,                         # {case_name: {"y_true": (N,), "y_pred": (N,C), "plot_kwargs": {...}}}
    class_names=None,                   # list[str] in y_pred column order; if None, inferred from labels
    xtick_labels_fontsize=24,
    title=None,
    figsize=(9, 25),
    palette=None,                       # optional list of colors for cases (fallback to tab10)
    save_path=None,                     # save figure (suffix decides format)
    include_metrics=("Accuracy", "F1", "TPR", "Precision", "AUROC", "ECE", "Brier"),
    # Layout
    nrows=7,
    subplot_hspace=0.1,
    subplot_wspace=0.25,
    # Per-subpanel y-ranges (by metric)
    y_ranges=None,                      
    y_margin_frac=0.07,
    # Bars & annotations
    bar_alpha=1.0,
    bar_edgecolor="black",
    group_width=0.9,                    
    annotate_values=True,
    value_label_fontsize=9,
    # Axes cosmetics
    ylabel_text="Score",                
    left_margin=0.10,
    # Tick formatting
    ytick_step=None,
    ytick_format="{x:.2f}",
    ytick_count=3,                   
    two_line_class_xticklabels=False,   
    # Best-line options (per class tick
    metric_best_high_low={"Accuracy": "high", "F1": "high", "TPR": "high", "Precision": "high", "AUROC": "high", "ECE": "low", "Brier": "low"},          
    best_line_kwargs={"ls": "--", "lw": 2.0, "alpha": 0.9},              
    # Metric title inside axes
    metric_title_fontsize=22,           
    metric_title_bbox={"facecolor": "white", "alpha": 0.9, "boxstyle": "round,pad=0.2", "edgecolor": "k"},             
):
    """
    Plot a multi-subpanel comparison of per-class metrics across an arbitrary number of cases.
    One subplot per metric; within each subplot, x-axis = classes, and bars = cases.

    Improvements:
    - Shared x-axis across rows (internal rows hide x labels).
    - Titles as in-axes text (top-left), customizable.
    - Optional fixed number of y-ticks (ytick_count) or step-based ticks (ytick_step).
    - Per-class best indicator lines colored by the winning case’s bar color.

    Notes:
    - "Accuracy" per class is taken as within-class accuracy, i.e., TPR/recall for that class.
    - AUROC per class is computed one-vs-rest using the probability of that class.
    - ECE per class is computed for the binary problem (class vs rest) using the predicted probability of that class.
    - Brier per class is the mean squared error of the predicted probability for that class vs. the one-hot target for that class.
    """

    if not dict_cases:
        raise ValueError("dict_cases must contain at least one case.")

    # ---- Defaults for best-line logic ----
    if metric_best_high_low is None:
        metric_best_high_low = {}
    # sensible defaults: low is better for ECE/Brier; high for others
    def _is_high_better(metric_name: str) -> bool:
        m = metric_name.lower()
        if m in ("ece", "brier", "brier score"):
            return metric_best_high_low.get(metric_name, "low") == "high"
        return metric_best_high_low.get(metric_name, "high") == "high"

    if best_line_kwargs is None:
        best_line_kwargs = {"ls": "--", "lw": 1.8, "alpha": 0.9}

    case_names = list(dict_cases.keys())
    n_cases = len(case_names)

    # Determine classes and consistency
    if class_names is None:
        all_y = np.concatenate([np.asarray(v["y_true"]) for v in dict_cases.values()])
        classes = np.unique(all_y)
        class_names_used = [str(c) for c in classes]
    else:
        class_names_used = list(class_names)
        classes = np.arange(len(class_names_used))
    C = len(classes)

    for name, payload in dict_cases.items():
        y_pred = np.asarray(payload["y_pred"])
        if y_pred.ndim != 2 or y_pred.shape[1] != C:
            raise ValueError(f"[{name}] y_pred must be (N,{C}). Got {y_pred.shape}.")

    # ---- Helpers for per-class ECE & Brier (binary, class-vs-rest) ----
    def _brier_per_class(y_true, p):
        return float(np.mean((p - y_true.astype(float)) ** 2))

    def _ece_binary(y_true, p, n_bins=15, strategy="uniform"):
        y_true = np.asarray(y_true).astype(int)
        p = np.asarray(p).astype(float)
        if p.size == 0:
            return np.nan
        if strategy == "uniform":
            bins = np.linspace(0.0, 1.0, n_bins + 1)
        else:
            bins = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
            bins[0], bins[-1] = 0.0, 1.0
        bin_ids = np.digitize(p, bins) - 1
        ece = 0.0
        N = len(p)
        for b in range(n_bins):
            mask = (bin_ids == b)
            if not np.any(mask):
                continue
            conf_b = float(np.mean(p[mask]))
            acc_b = float(np.mean(y_true[mask]))
            ece += (np.sum(mask) / N) * abs(acc_b - conf_b)
        return float(ece)

    # ---- Compute per-class metrics for each case ----
    per_metric_vals = {m: np.full((C, n_cases), np.nan, dtype=float) for m in include_metrics}

    for j, (case, payload) in enumerate(dict_cases.items()):
        y_true = np.asarray(payload["y_true"])
        y_pred = np.asarray(payload["y_pred"])
        y_hat = np.argmax(y_pred, axis=1)

        f1_vec  = f1_score(y_true, y_hat, average=None, zero_division=0)
        rec_vec = recall_score(y_true, y_hat, average=None, zero_division=0)
        prec_vec= precision_score(y_true, y_hat, average=None, zero_division=0)

        for i, cls in enumerate(classes):
            # Accuracy per class = recall for that class
            if "Accuracy" in per_metric_vals:
                per_metric_vals["Accuracy"][i, j] = rec_vec[i]
            if "F1" in per_metric_vals:
                per_metric_vals["F1"][i, j] = f1_vec[i]
            if "TPR" in per_metric_vals:
                per_metric_vals["TPR"][i, j] = rec_vec[i]
            if "Precision" in per_metric_vals:
                per_metric_vals["Precision"][i, j] = prec_vec[i]

            y_bin = (y_true == cls).astype(int)
            p_cls = y_pred[:, i]

            if "AUROC" in per_metric_vals:
                if len(np.unique(y_bin)) > 1:
                    try:
                        per_metric_vals["AUROC"][i, j] = roc_auc_score(y_bin, p_cls)
                    except Exception:
                        per_metric_vals["AUROC"][i, j] = np.nan
                else:
                    per_metric_vals["AUROC"][i, j] = np.nan
            if "ECE" in per_metric_vals:
                try:
                    per_metric_vals["ECE"][i, j] = _ece_binary(y_bin, p_cls)
                except Exception:
                    per_metric_vals["ECE"][i, j] = np.nan
            if "Brier" in per_metric_vals:
                try:
                    per_metric_vals["Brier"][i, j] = _brier_per_class(y_bin, p_cls)
                except Exception:
                    per_metric_vals["Brier"][i, j] = np.nan

    # ---- Figure & subpanels (shared x across rows) ----
    M = len(include_metrics)
    ncols = ceil(M / nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    if isinstance(axes, np.ndarray):
        axes = np.array(axes).reshape(nrows, ncols)
    else:
        axes = np.array([[axes]])

    # colors per case (fallback to tab10)
    if palette is None:
        palette = [plt.cm.tab10(i % 10) for i in range(n_cases)]
    case_colors = []
    for k, nm in enumerate(case_names):
        kw = dict(dict_cases[nm].get("plot_kwargs", {}))
        case_colors.append(kw.get("color", palette[k]))
    case_labels = [dict_cases[nm].get("plot_kwargs", {}).get("label", nm) for nm in case_names]

    # x-axis: classes; grouped bars by case
    x = np.arange(C)
    width = min(group_width / max(n_cases, 1), 0.9 / max(n_cases, 1))
    offsets = (np.arange(n_cases) - (n_cases - 1) / 2.0) * width

    # helper: split class labels in two lines
    def _two_line(name: str) -> str:
        s = str(name)
        if not two_line_class_xticklabels:
            return s
        splits = [i for i, ch in enumerate(s) if ch in (" ", "_", "-")]
        if not splits:
            return s
        center = len(s) / 2.0
        split_idx = min(splits, key=lambda i: abs(i - center))
        return s[:split_idx] + "\n" + s[split_idx+1:]

    xtick_labels = [_two_line(nm) for nm in class_names_used]

    # default metric title bbox
    if metric_title_bbox is None:
        metric_title_bbox = dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2", edgecolor="none")

    # Draw each metric subplot
    for m_idx, metric in enumerate(include_metrics):
        r = m_idx // ncols
        c = m_idx % ncols
        ax = axes[r, c]
        vals = per_metric_vals[metric]  # shape (C, n_cases)

        # y-limits (explicit or auto)
        if isinstance(y_ranges, dict) and metric in y_ranges:
            ymin, ymax = y_ranges[metric]
        else:
            finite = vals[np.isfinite(vals)]
            if finite.size:
                ymin = float(min(0.0, np.min(finite)))
                ymax = float(max(1.0, np.max(finite)))
                if ymax - ymin < 1e-6:
                    ymax = ymin + 1.0
                margin = y_margin_frac * (ymax - ymin)
                ymin -= margin
                ymax += margin
            else:
                ymin, ymax = 0.0, 1.0

        # grouped bars
        for j in range(n_cases):
            y_j = [vals[i, j] if np.isfinite(vals[i, j]) else 0.0 for i in range(C)]
            ax.bar(
                x + offsets[j], y_j,
                width=width,
                color=case_colors[j],
                edgecolor=bar_edgecolor,
                alpha=bar_alpha,
            )
            if annotate_values:
                for xi, yv in zip(x + offsets[j], y_j):
                    ax.text(
                        xi, yv + 0.01 * (ymax - ymin),
                        f"{yv:.3f}",
                        ha="center", va="bottom", fontsize=value_label_fontsize
                    )

        # per-class best lines (short segments centered at each class tick)
        high_better = _is_high_better(metric)
        for i_cls in range(C):
            col = vals[i_cls, :]
            if not np.any(np.isfinite(col)):
                continue
            if high_better:
                j_best = int(np.nanargmax(col))
            else:
                j_best = int(np.nanargmin(col))
            y_best = float(col[j_best])
            # draw a short horizontal segment spanning the group at this class tick
            x_left  = x[i_cls] - group_width / 2.0
            x_right = x[i_cls] + group_width / 2.0
            line_kwargs = dict(best_line_kwargs)
            line_kwargs["color"] = case_colors[j_best]  # color of the winning case
            ax.hlines(y_best, x_left, x_right, **line_kwargs)

        # cosmetics per subplot
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(ylabel_text)

        # x ticks: class names (set only once; sharex propagates)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=0, fontsize=xtick_labels_fontsize)
        ax.set_xlim(x[0] - group_width/2.0 - 0.2, x[-1] + group_width/2.0 + 0.2)

        # y ticks: either fixed count or step-based
        if ytick_count is not None and ytick_count > 1:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(ytick_count))
        else:
            ax.yaxis.set_major_locator(mticker.MultipleLocator(ytick_step))
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(ytick_format))

        # titles as in-axes text (top-left)
        ax.text(
            0.05, 0.90, str(metric),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=metric_title_fontsize,
            bbox=metric_title_bbox
        )

        ax.grid(False)

    # hide any unused axes
    for idx_extra in range(M, nrows * ncols):
        r = idx_extra // ncols
        c = idx_extra % ncols
        axes[r, c].axis("off")

    # hide shared x tick labels for all but bottom row
    for r in range(nrows - 1):
        for c in range(ncols):
            axes[r, c].tick_params(axis="x", which="both", labelbottom=False)

    # figure title (no legend)
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    # reserve left margin so first y-label isn't clipped
    fig.subplots_adjust(left=left_margin, hspace=subplot_hspace, wspace=subplot_wspace)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes

def plot_latents_scatter(
    X_emb, y_labels,
    class_counts=None,
    class_names=None,
    title="t-SNE Plot",
    n_bins=128,
    sigma=2.0,
    scatter_size=1,
    scatter_alpha=1.0,
    xlim=None,
    ylim=None,
    xlabel="t-SNE 1",
    ylabel="t-SNE 2"
):

    unique_classes = np.unique(y_labels)
    cmap = plt.cm.get_cmap("tab10")
    class_color_dict = {cls: cmap(i) for i, cls in enumerate(unique_classes)}
    class_rgb = np.array([class_color_dict[cls][:3] for cls in unique_classes])

    if class_counts is None:
        class_counts = np.array([(y_labels == cls).sum() for cls in unique_classes])

    inv_freq_weights = 1.0 / np.maximum(class_counts, 1)
    inv_freq_weights /= inv_freq_weights.sum()

    # Determine plot window
    x_data_min, x_data_max = np.min(X_emb[:, 0]), np.max(X_emb[:, 0])
    y_data_min, y_data_max = np.min(X_emb[:, 1]), np.max(X_emb[:, 1])
    x_min, x_max = (x_data_min, x_data_max) if xlim is None else (xlim[0], xlim[1])
    y_min, y_max = (y_data_min, y_data_max) if ylim is None else (ylim[0], ylim[1])

    # Sanity: swap if limits are reversed
    if x_min > x_max: x_min, x_max = x_max, x_min
    if y_min > y_max: y_min, y_max = y_max, y_min

    # Precompute in-window mask (used for scatter)
    in_x = (X_emb[:, 0] >= x_min) & (X_emb[:, 0] <= x_max)
    in_y = (X_emb[:, 1] >= y_min) & (X_emb[:, 1] <= y_max)
    in_win = in_x & in_y

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)

    # Fix limits up front + disable autoscale so artists won't expand the view
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_autoscale_on(False)  # <- key line

    # Density per class (bounded to the requested window)
    H_class = np.zeros((n_bins, n_bins, len(unique_classes)))
    for i, cls in enumerate(unique_classes):
        idx = (y_labels == cls)
        # For histogram: still count everything that falls in the specified window
        stat, _, _, _ = binned_statistic_2d(
            X_emb[idx, 0], X_emb[idx, 1], None,
            statistic='count', bins=n_bins,
            range=[[x_min, x_max], [y_min, y_max]]
        )
        stat = gaussian_filter(stat.T, sigma=sigma)
        H_class[:, :, i] = stat * inv_freq_weights[i]

        # For scatter: only draw points inside the window (faster & respects limits)
        scatter_mask = idx & in_win
        if scatter_mask.any():
            ax.scatter(
                X_emb[scatter_mask, 0], X_emb[scatter_mask, 1],
                color=class_color_dict[cls],
                s=scatter_size, alpha=scatter_alpha,
                clip_on=True
            )

    # Composite RGB density overlay
    H_total = np.sum(H_class, axis=2, keepdims=True)
    proportions = np.divide(H_class, H_total, out=np.zeros_like(H_class), where=(H_total != 0))
    image_rgb = np.tensordot(proportions, class_rgb, axes=(2, 0))

    density = H_total.squeeze()
    eps = 1e-3
    density_log = np.log1p(density / eps)
    panel_max = np.max(density_log)
    density_mod = density_log / panel_max if panel_max > 0 else density_log
    density_mod[density < eps] = 0
    image_rgb *= density_mod[..., None]

    ax.imshow(
        image_rgb,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower', aspect='auto', interpolation='nearest',
        zorder=0
    )

    legend_elements = [
        mpatches.Patch(color=class_color_dict[cls],
                       label=(class_names[i] if class_names else f"Class {cls}"))
        for i, cls in enumerate(unique_classes)
    ]
    ax.legend(handles=legend_elements, title="Class", fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.show()

def plot_latents_scatter_val_test(
    X_val, y_val,
    X_test, y_test,
    *,
    class_names=None,
    title="Latent space (val vs test)",
    marker_val="o",
    marker_test="^",
    size_val=8,
    size_test=8,
    alpha_val=0.9,
    alpha_test=0.9,
    xlim=None,
    ylim=None,
    subsample=None,            # float in (0,1] for fraction, or int >=1 for max count; applies to each split
    seed=42,
    edgecolor="none",
    linewidths=0.0,
    legend_split_1="Val",
    legend_split_2="Test",
):
    """
    Overlay a 2D latent embedding for validation and test sets as a scatter plot.
    Colors encode classes; markers encode split (val vs test).

    Parameters
    ----------
    X_val, X_test : (N, 2) arrays of embeddings
    y_val, y_test : (N,) label arrays
    class_names   : list or dict (label->name), optional
    subsample     : float in (0,1] or int >=1, optional (applied independently to val and test)
    """

    X_val = np.asarray(X_val);  X_test = np.asarray(X_test)
    y_val = np.asarray(y_val);  y_test = np.asarray(y_test)
    assert X_val.shape[1] == 2 and X_test.shape[1] == 2, "X_val/X_test must be (N,2)."

    rng = np.random.default_rng(seed)

    def _subsample_global(X, y, subsample):
        if subsample is None:
            return X, y
        n = len(y)
        if isinstance(subsample, float):
            if not (0 < subsample <= 1):
                # clamp silently to valid range
                subs = min(max(subsample, 1e-9), 1.0)
            else:
                subs = subsample
            k = max(1, int(round(subs * n)))
        elif isinstance(subsample, (int, np.integer)):
            k = min(int(subsample), n)
        else:
            return X, y
        if k >= n:
            return X, y
        idx = rng.choice(n, size=k, replace=False)
        return X[idx], y[idx]

    # Global (per-split) subsampling to keep relative class frequencies ~unchanged
    X_val, y_val   = _subsample_global(X_val, y_val, subsample)
    X_test, y_test = _subsample_global(X_test, y_test, subsample)

    # Classes and colors
    unique_classes = np.unique(np.concatenate([y_val, y_test]))
    classes_sorted = list(unique_classes)
    cmap = plt.cm.get_cmap("tab10")
    color_map = {cls: cmap(i % 10) for i, cls in enumerate(classes_sorted)}

    # Ax limits
    x_all = np.concatenate([X_val[:, 0], X_test[:, 0]])
    y_all = np.concatenate([X_val[:, 1], X_test[:, 1]])
    x_min, x_max = (x_all.min(), x_all.max()) if xlim is None else xlim
    y_min, y_max = (y_all.min(), y_all.max()) if ylim is None else ylim

    fig, ax = plt.subplots(figsize=(6., 6.))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("dim 1", fontsize=12)
    ax.set_ylabel("dim 2", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(labelsize=10)

    # Plot per class for consistent colors
    for cls in classes_sorted:
        # validation
        iv = (y_val == cls)
        if iv.any():
            ax.scatter(
                X_val[iv, 0], X_val[iv, 1],
                s=size_val, marker=marker_val,
                c=[color_map[cls]], alpha=alpha_val,
                edgecolor=edgecolor, linewidths=linewidths
            )
        # test
        it = (y_test == cls)
        if it.any():
            ax.scatter(
                X_test[it, 0], X_test[it, 1],
                s=size_test, marker=marker_test,
                c=[color_map[cls]], alpha=alpha_test,
                edgecolor=edgecolor, linewidths=linewidths
            )

    # Legend A: class colors
    if isinstance(class_names, dict):
        class_labels = [class_names.get(cls, str(cls)) for cls in classes_sorted]
    elif isinstance(class_names, (list, tuple)) and len(class_names) == len(classes_sorted):
        class_labels = list(class_names)
    else:
        class_labels = [str(cls) for cls in classes_sorted]

    class_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color_map[cls], markeredgecolor="none",
               markersize=8, label=lab)
        for cls, lab in zip(classes_sorted, class_labels)
    ]
    leg_classes = ax.legend(
        handles=class_handles, title="Class", loc="upper left",
        fontsize=9, title_fontsize=10, frameon=True
    )
    ax.add_artist(leg_classes)

    # Legend B: split markers
    ds_handles = [
        Line2D([0], [0], marker=marker_val, color="k", linestyle="None", markersize=8, label=legend_split_1),
        Line2D([0], [0], marker=marker_test, color="k", linestyle="None", markersize=8, label=legend_split_2),
    ]
    ax.legend(handles=ds_handles, title="Split", loc="lower right",
              fontsize=9, title_fontsize=10, frameon=True)

    plt.tight_layout()
    plt.show()

def plot_latent_density_2d(
    X,
    *,
    title="Latent density",
    # ---- density method ----
    density_method="hist",     # "hist" or "kde"
    bins=128,                  # used if density_method="hist" or for KDE grid resolution
    sigma=1.5,                 # smoothing (bin units) for "hist" only
    kde_bw="scott",            # KDE bandwidth: "scott", "silverman", or float
    # ---- normalization & color ----
    norm_mode="max",           # "max" -> normalize by max; "sum" -> normalized to integrate to 1 over grid
    color_scale="log",         # "log" or "linear"
    linear_vmax=99,            # percentile for vmax in linear mode (helps visibility)
    mask_zero_support=True,    # keep true zeros as 0 so they map to the lowest colormap color
    cmap="viridis",
    # ---- contours ----
    contour_fracs=(0.8, 0.5, 0.2),   # as fractions of the normalized density (i.e., 0..1)
    contour_colors="k",
    contour_linewidths=0.8,
    contour_label_fontsize=8,        # fontsize for contour labels
    contour_label_color="k",         # color for contour labels
    # ---- points overlay ----
    show_points=False,
    points_alpha=0.15,
    points_size=3,
    random_subsample=None,
    # ---- view ----
    xlim=None,
    ylim=None,
    seed=42,
):
    """
    Plot a 2D density of latent points with optional log coloring, isodensity contours,
    and a light scatter overlay.

    density_method:
      - "hist": 2D histogram + Gaussian smoothing (sigma in bin units)
      - "kde" : Gaussian KDE (bandwidth via `kde_bw`), evaluated on a regular grid
    """
    X = np.asarray(X, dtype=float)
    assert X.ndim == 2 and X.shape[1] == 2, "X must be (N,2)."
    rng = np.random.default_rng(seed)

    # ---- bounds ----
    if xlim is None:
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
    else:
        x_min, x_max = xlim
    if ylim is None:
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
    else:
        y_min, y_max = ylim

    # choose grid resolution
    nx = ny = int(bins)

    # >>> use EDGES (len = bins+1) for histogram
    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    # centers (len = bins) for contour grid so shapes match Z
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    XX, YY = np.meshgrid(x_centers, y_centers, indexing="xy")

    # ---- density on the grid ----
    if density_method == "hist":
        # H shape: (nx, ny) before transpose → transpose to (ny, nx)
        H, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[x_edges, y_edges])
        H = H.T  # now (ny, nx), same as XX, YY
        if sigma and sigma > 0:
            H = gaussian_filter(H, sigma=sigma, mode="nearest")
        density = H.astype(float)

    elif density_method == "kde":
        # evaluate KDE on centers so it matches (ny, nx)
        kde = gaussian_kde(X.T, bw_method=kde_bw)
        Zk = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(ny, nx)
        density = Zk
    else:
        raise ValueError("density_method must be 'hist' or 'kde'.")

    # ---- normalization ----
    if norm_mode == "sum":
        total = density.sum()
        if total > 0:
            dens_norm = density / total
        else:
            dens_norm = density * 0.0
    elif norm_mode == "max":
        m = density.max()
        dens_norm = density / m if m > 0 else density * 0.0
    else:
        raise ValueError("norm_mode must be 'max' or 'sum'.")

    # ---- color scaling ----
    if color_scale == "log":
        # log1p on nonzeros; keep exact zeros as 0 so they map to the lowest color
        Z = dens_norm.copy()
        nonzero = Z > 0
        if nonzero.any():
            Z[nonzero] = np.log1p(Z[nonzero] / Z[nonzero].max())
        # nonzero max rescales into (0,1]; zeros remain 0
        vmin, vmax = 0.0, 1.0
    elif color_scale == "linear":
        # stretch to [0, 1] using percentile for robustness
        vmin = 0.0
        vmax = np.percentile(dens_norm, linear_vmax)
        if vmax <= 0:
            vmax = 1.0
        Z = np.clip(dens_norm / vmax, 0, 1)
    else:
        raise ValueError("color_scale must be 'log' or 'linear'.")

    # ---- optional masking of zero-support regions ----
    if mask_zero_support:
        # Leave zeros as 0 so they take the lowest color of the colormap
        pass  # already handled by keeping zeros as 0 in Z
    # else: nothing special — Z already in 0..1

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("dim 1", fontsize=12)
    ax.set_ylabel("dim 2", fontsize=12)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(labelsize=10)

    # image
    im = ax.imshow(
        Z, origin="lower", aspect="auto",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest"
    )

    # contours on the normalized density (so levels == fracs)
    if contour_fracs and len(contour_fracs) > 0:
        levels = np.clip(np.array(contour_fracs, dtype=float), 0.0, 1.0)
        CS = ax.contour(
            XX, YY, dens_norm,
            levels=levels,
            colors=contour_colors,
            linewidths=contour_linewidths
        )
        # label contours: values are the normalized levels (0..1)
        if CS.allsegs and any(len(s) for s in CS.allsegs):
            fmt = lambda v: f"{v:.2f}"
            ax.clabel(
                CS, inline=True, fmt=fmt,
                fontsize=contour_label_fontsize,
                colors=contour_label_color
            )

    # scatter overlay
    if show_points:
        pts = X
        if isinstance(random_subsample, (int, np.integer)) and random_subsample > 0:
            k = min(int(random_subsample), len(pts))
            idx = rng.choice(len(pts), size=k, replace=False)
            pts = pts[idx]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=points_size, alpha=points_alpha,
            c="k", marker=".", linewidths=0
        )

    # colorbar always 0..1 after our normalization
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("normalized density (0–1)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.show()

def safe_compare(a, b, path="root"):
    """Recursively compare two structures with detailed debug logs and NumPy-safe checks."""
    
    if isinstance(a, dict) and isinstance(b, dict):
        logging.debug(f"🔍 Comparing dicts at {path}")
        keys_a = set(a.keys())
        keys_b = set(b.keys())

        for key in keys_a.union(keys_b):
            if key not in a:
                logging.debug(f"❌ Key '{key}' missing in first config at {path}")
                return False
            if key not in b:
                logging.debug(f"❌ Key '{key}' missing in second config at {path}")
                return False
            if not safe_compare(a[key], b[key], f"{path}['{key}']"):
                return False
        return True

    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        logging.debug(f"🔍 Comparing {'list' if isinstance(a, list) else 'tuple'} at {path}")
        if len(a) != len(b):
            logging.debug(f"❌ Length mismatch at {path}: {len(a)} ≠ {len(b)}")
            return False
        for i, (item_a, item_b) in enumerate(zip(a, b)):
            if not safe_compare(item_a, item_b, f"{path}[{i}]"):
                return False
        return True

    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        logging.debug(f"🔍 Comparing NumPy arrays at {path} with shape {a.shape} and {b.shape}")
        if not np.array_equal(a, b):
            logging.debug(f"❌ Array mismatch at {path}")
            return False
        return True

    else:
        logging.debug(f"🔍 Comparing values at {path}: {a} vs {b}")
        if a != b:
            logging.debug(f"❌ Value mismatch at {path}: {a} ≠ {b}")
            return False
        return True
