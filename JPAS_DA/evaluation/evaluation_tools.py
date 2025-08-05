import numpy as np
import logging
import torch

from copy import deepcopy
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter

from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

from JPAS_DA import global_setup
from JPAS_DA.data import loading_tools, cleaning_tools, crossmatch_tools, process_dset_splits
from JPAS_DA.models import model_building_tools
from JPAS_DA.training import save_load_tools
from JPAS_DA.wrapper_wandb import wrapper_tools
from JPAS_DA.utils.plotting_utils import get_N_colors

def assert_array_lists_equal(list1, list2, rtol=1e-5, atol=1e-8) -> bool:
    """
    Compare two lists of NumPy arrays and log results for each comparison.

    Parameters:
    ----------
    list1, list2 : list of np.ndarray
        Lists to compare.
    rtol, atol : float
        Tolerances for np.allclose comparison.

    Returns:
    --------
    all_match : bool
        True if all arrays match, False otherwise.
    """
    if len(list1) != len(list2):
        logging.error(f"❌ List lengths differ: {len(list1)} != {len(list2)}")
        return False

    all_match = True
    for i, (arr1, arr2) in enumerate(zip(list1, list2)):
        if np.allclose(arr1, arr2, rtol=rtol, atol=atol):
            logging.debug(f"✅ Arrays at index {i} match.")
        else:
            logging.warning(f"❌ Arrays at index {i} differ.")
            all_match = False

    if all_match:
        logging.info("🎉 All arrays match.")
    else:
        logging.info("⚠️ Some arrays differ.")

    return all_match

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

def compare_TPR_confusion_matrices(
    yy_true_val,
    yy_pred_P_val,
    yy_true_test,
    yy_pred_P_test,
    class_names,
    figsize=(10, 8),
    cmap='bwr',
    title='Difference in Normalized Confusion Matrices (Test - Validation)',
    name_1 = "Val",
    name_2 = "Test"
):
    """
    Compares full normalized confusion matrices (TPR-style) between test and validation datasets.

    Parameters:
    ----------
    - yy_true_val: np.ndarray of true labels for validation set
    - yy_pred_P_val: np.ndarray of predicted probabilities for validation set
    - yy_true_test: np.ndarray of true labels for test set
    - yy_pred_P_test: np.ndarray of predicted probabilities for test set
    - class_names: list of class names
    - figsize: size of the plot
    - cmap: colormap for matrix difference
    - title: title of the plot
    """
    yy_pred_val = np.argmax(yy_pred_P_val, axis=1)
    yy_pred_test = np.argmax(yy_pred_P_test, axis=1)
    num_classes = len(class_names)

    def get_normalized_confusion_matrix(y_true, y_pred, num_classes):
        cm = np.zeros((num_classes, num_classes), dtype=np.float32)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            cm_normalized = np.divide(cm, row_sums, where=row_sums != 0)
        return cm_normalized, cm.astype(int)

    cm_val_norm, cm_val_raw = get_normalized_confusion_matrix(yy_true_val, yy_pred_val, num_classes)
    cm_test_norm, cm_test_raw = get_normalized_confusion_matrix(yy_true_test, yy_pred_test, num_classes)
    cm_diff = cm_test_norm - cm_val_norm

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_diff, interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("TPR("+name_2+") - TPR("+name_1+")", fontsize=12)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title(title, fontsize=15)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    for i in range(num_classes):
        for j in range(num_classes):
            val_pct = cm_val_norm[i, j] * 100
            test_pct = cm_test_norm[i, j] * 100
            diff_pct = cm_diff[i, j] * 100
            text_color = "white" if abs(cm_diff[i, j]) > 0.5 else "black"
            ax.text(
                j, i,
                f"{name_1}:{val_pct:.1f}%\n{name_2}:{test_pct:.1f}%\nΔ:{diff_pct:+.1f}%",
                ha="center", va="center", color=text_color, fontsize=9,
                fontweight='bold' if i == j else 'normal'
            )

    plt.tight_layout()
    plt.show()

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

def make_colored_diff(val, is_higher_better=True):
    if np.isnan(val):
        return "       NaN    "
    color = "\033[92m" if (val > 0 and is_higher_better) or (val < 0 and not is_higher_better) else "\033[91m"
    return f"{color}{val:12.4f}\033[0m"

def multiclass_brier_score(y_true, y_prob, n_classes=None):
    if n_classes is None:
        n_classes = y_prob.shape[1]
    y_true_one_hot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1))

def compare_sets_performance(
    yy_true_1, yy_pred_P_1,
    yy_true_2, yy_pred_P_2,
    class_names=None,
    y_min_Delta_F1=-0.2, y_max_Delta_F1=0.2,
    name_1="Set 1", name_2="Set 2",
    plot_ROC_curves = True
):
    yy_pred_1 = np.argmax(yy_pred_P_1, axis=1)
    yy_pred_2 = np.argmax(yy_pred_P_2, axis=1)

    # Compute per-class recall (for metrics) and per-class F1 (for plot)
    tpr_1 = recall_score(yy_true_1, yy_pred_1, average=None, zero_division=0)
    tpr_2 = recall_score(yy_true_2, yy_pred_2, average=None, zero_division=0)
    f1_1 = f1_score(yy_true_1, yy_pred_1, average=None, zero_division=0)
    f1_2 = f1_score(yy_true_2, yy_pred_2, average=None, zero_division=0)

    is_multiclass = yy_pred_P_1.ndim == 2 and yy_pred_P_1.shape[1] > 2

    metrics = {
        "Accuracy": (accuracy_score(yy_true_1, yy_pred_1), accuracy_score(yy_true_2, yy_pred_2), True),
        "Macro F1": (np.mean(f1_1), np.mean(f1_2), True),
        "Macro TPR": (np.mean(tpr_1), np.mean(tpr_2), True),
        "Macro Precision": (precision_score(yy_true_1, yy_pred_1, average='macro', zero_division=0),
                            precision_score(yy_true_2, yy_pred_2, average='macro', zero_division=0), True),
        "Macro AUROC": (
            roc_auc_score(
                yy_true_1,
                yy_pred_P_1 if is_multiclass else yy_pred_P_1[:, 1],
                average='macro' if is_multiclass else None,
                multi_class='ovo' if is_multiclass else 'raise'
            ) if len(np.unique(yy_true_1)) > 1 else np.nan,

            roc_auc_score(
                yy_true_2,
                yy_pred_P_2 if is_multiclass else yy_pred_P_2[:, 1],
                average='macro' if is_multiclass else None,
                multi_class='ovo' if is_multiclass else 'raise'
            ) if len(np.unique(yy_true_2)) > 1 else np.nan,

            True
        ),
        "Expected Calibration Error": (compute_ece(yy_true_1, yy_pred_P_1), compute_ece(yy_true_2, yy_pred_P_2), False),
        "Brier Score": (
            multiclass_brier_score(yy_true_1, yy_pred_P_1),
            multiclass_brier_score(yy_true_2, yy_pred_P_2),
            False
        )
    }

    print(f"\n=== {name_1} vs {name_2} Metrics ===")
    header = f"{'Metric':<30}{name_1:>12}{name_2:>12}{f'Δ ({name_2} - {name_1})':>18}"
    print(header)
    print("-" * len(header))
    for metric, (v1, v2, higher_is_better) in metrics.items():
        delta = v2 - v1
        delta_str = f"{delta:18.4f}"
        if (higher_is_better and delta > 0) or (not higher_is_better and delta < 0):
            delta_str = f"\033[92m{delta_str}\033[0m"  # green
        elif delta != 0:
            delta_str = f"\033[91m{delta_str}\033[0m"  # red
        print(f"{metric:<30}{v1:12.4f}{v2:12.4f}{delta_str}")

    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(f1_1))]
    colors = get_N_colors(len(class_names), plt.cm.tab10)

    # Plot generalization gap in per-class F1
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, f1_2 - f1_1, color=colors)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.ylabel(f"Δ F1-score")
    plt.title(f"{name_2} - {name_1}")
    plt.xticks(rotation=15, ha='right')
    plt.ylim(y_min_Delta_F1, y_max_Delta_F1)
    plt.tight_layout()
    plt.show()

    # ROC Curves
    if plot_ROC_curves:
        plot_combined_multiclass_roc_and_diff(
            yy_true_1, yy_pred_P_1, yy_true_2, yy_pred_P_2,
            class_names=class_names, name_1=name_1, name_2=name_2
        )
    
    return metrics, f1_1, f1_2

def safe_interp(fpr, tpr, x_new):
    fpr_unique, idx = np.unique(fpr, return_index=True)
    tpr_unique = tpr[idx]
    interpolator = interp1d(fpr_unique, tpr_unique, kind='linear', bounds_error=False, fill_value='extrapolate')
    y_new = interpolator(x_new)
    return np.nan_to_num(y_new, nan=0.0, posinf=0.0, neginf=0.0)

def plot_combined_multiclass_roc_and_diff(y_true_1, y_pred_P_1, y_true_2, y_pred_P_2,
                                          class_names=None, name_1="Model 1", name_2="Model 2"):
    """
    Plot multiclass or binary ROC curves (top) and TPR difference (bottom).
    """

    classes = np.unique(np.concatenate([y_true_1, y_true_2]))
    num_classes = len(classes)
    colors = get_N_colors(num_classes, plt.cm.tab10)
    fpr_common = np.linspace(0, 1, 200)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_diff = fig.add_subplot(gs[1], sharex=ax_main)

    if num_classes == 2:
        # Binary classification: use positive class (assume class 1)
        if y_pred_P_1.ndim == 2:
            y_pred_1 = y_pred_P_1[:, 1]
            y_pred_2 = y_pred_P_2[:, 1]
        else:
            y_pred_1 = y_pred_P_1
            y_pred_2 = y_pred_P_2

        fpr1, tpr1, _ = roc_curve(y_true_1, y_pred_1)
        fpr2, tpr2, _ = roc_curve(y_true_2, y_pred_2)
        auc1 = auc(fpr1, tpr1)
        auc2 = auc(fpr2, tpr2)

        label = class_names[1] if class_names else "Positive class"
        ax_main.plot(fpr1, tpr1, '-', color=colors[1], label=f"{label} ({name_1}) [AUC={auc1:.2f}]")
        ax_main.plot(fpr2, tpr2, '--', color=colors[1], label=f"{label} ({name_2}) [AUC={auc2:.2f}]")

        delta_tpr = safe_interp(fpr2, tpr2, fpr_common) - safe_interp(fpr1, tpr1, fpr_common)
        ax_diff.plot(fpr_common, delta_tpr, color=colors[1], lw=2, label=label)

    else:
        # Multiclass
        y_true_1_bin = label_binarize(y_true_1, classes=classes)
        y_true_2_bin = label_binarize(y_true_2, classes=classes)

        for i, cls in enumerate(classes):
            fpr1, tpr1, _ = roc_curve(y_true_1_bin[:, i], y_pred_P_1[:, i])
            fpr2, tpr2, _ = roc_curve(y_true_2_bin[:, i], y_pred_P_2[:, i])
            auc1 = auc(fpr1, tpr1)
            auc2 = auc(fpr2, tpr2)

            label = f"Class {cls}" if class_names is None else class_names[i]
            ax_main.plot(fpr1, tpr1, '-', color=colors[i], label=f"{label} ({name_1}) [AUC={auc1:.2f}]")
            ax_main.plot(fpr2, tpr2, '--', color=colors[i], label=f"{label} ({name_2}) [AUC={auc2:.2f}]")

            delta_tpr = safe_interp(fpr2, tpr2, fpr_common) - safe_interp(fpr1, tpr1, fpr_common)
            ax_diff.plot(fpr_common, delta_tpr, color=colors[i], lw=2, label=label)

    # Final formatting
    ax_main.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_main.set_xlim([0.0, 1.0])
    ax_main.set_ylim([0.0, 1.05])
    ax_main.set_ylabel("True Positive Rate", fontsize=13)
    ax_main.set_title("ROC Curves", fontsize=15)
    ax_main.legend(fontsize=9, loc="lower right")
    ax_main.grid(True, linestyle='--', alpha=0.4)
    ax_main.tick_params(axis='x', labelbottom=False)

    ax_diff.axhline(0, color="black", linestyle="--", lw=1)
    ax_diff.set_xlabel("False Positive Rate (FPR)", fontsize=13)
    ax_diff.set_ylabel(f"ΔTPR ({name_2} - {name_1})", fontsize=11)
    ax_diff.grid(True, linestyle='--', alpha=0.4)
    ax_diff.set_yscale("symlog", linthresh=1e-2)

    plt.tight_layout()
    plt.show()

def plot_tsne_comparison_single_pair(
    X_set1, y_set1,
    X_set2, y_set2,
    class_counts,
    class_names=None,
    title_set1="Set 1",
    title_set2="Set 2",
    n_bins=128,
    sigma=2.0,
    scatter_size=1,
    scatter_alpha=1.0
):
    """
    Compare two t-SNE projections with color-coded density maps of class composition.

    Parameters:
        X_set1, y_set1 : ndarray
            t-SNE embedding and labels for the first set.
        X_set2, y_set2 : ndarray
            t-SNE embedding and labels for the second set.
        class_counts : ndarray
            Number of objects per class (used for inverse frequency weighting).
        class_names : list, optional
            List of class names (for legend display).
        title_set1, title_set2 : str
            Titles to display above each subplot.
        n_bins : int
            Number of bins in the 2D histogram grid.
        sigma : float
            Gaussian smoothing sigma.
        scatter_size : float
            Size of individual points in scatter plot.
        scatter_alpha : float
            Transparency of scatter points.
    """
    y_all = np.concatenate([y_set1, y_set2])
    unique_classes = np.unique(y_all)
    cmap = plt.cm.get_cmap("tab10")
    class_color_dict = {cls: cmap(i) for i, cls in enumerate(unique_classes)}
    class_rgb = np.array([class_color_dict[cls][:3] for cls in unique_classes])

    inv_freq_weights = 1 / class_counts
    inv_freq_weights /= np.sum(inv_freq_weights)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    datasets = [
        (title_set1, X_set1, y_set1, axs[0]),
        (title_set2, X_set2, y_set2, axs[1])
    ]

    all_points = np.vstack([X_set1, X_set2])
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])

    for title, X_emb, y_labels, ax in datasets:
        ax.set_title(f"t-SNE: {title}", fontsize=14)
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(labelsize=10)

        H_class = np.zeros((n_bins, n_bins, len(unique_classes)))
        for i, cls in enumerate(unique_classes):
            idx = y_labels == cls
            if np.sum(idx) == 0:
                continue
            stat, _, _, _ = binned_statistic_2d(
                X_emb[idx, 0], X_emb[idx, 1], None,
                statistic='count', bins=n_bins,
                range=[[x_min, x_max], [y_min, y_max]]
            )
            stat = gaussian_filter(stat.T, sigma=sigma)
            H_class[:, :, i] = stat * inv_freq_weights[i]
            ax.scatter(X_emb[idx, 0], X_emb[idx, 1],
                       color=class_color_dict[cls], s=scatter_size, alpha=scatter_alpha)

        H_total = np.sum(H_class, axis=2, keepdims=True)
        proportions = np.divide(H_class, H_total, out=np.zeros_like(H_class), where=H_total != 0)
        image_rgb = np.tensordot(proportions, class_rgb, axes=(2, 0))

        density = H_total.squeeze()
        eps = 1e-3
        density_log = np.log1p(density / eps)
        panel_max = np.max(density_log)
        density_mod = density_log / panel_max if panel_max > 0 else density_log
        density_mod[density < eps] = 0
        image_rgb *= density_mod[..., None]

        ax.imshow(image_rgb, extent=[x_min, x_max, y_min, y_max],
                  origin='lower', aspect='auto', interpolation='nearest')

    legend_elements = [
        mpatches.Patch(color=class_color_dict[cls], label=class_names[i] if class_names else f"Class {cls}")
        for i, cls in enumerate(unique_classes)
    ]
    axs[0].legend(handles=legend_elements, title="Class", fontsize=10, title_fontsize=11)

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


def evaluate_results_from_load_paths(
    paths_load,
    return_keys=['val_DESI_only', 'test_JPAS_matched'],
    define_dataset_loaders_keys=['DESI_only', 'JPAS_matched'],
    keys_yy=["SPECTYPE_int", "TARGETID", "DESI_FLUX_R"]
):

    # ─────────────────────────────────────────────────────────────
    # Load and validate data config across all paths
    # ─────────────────────────────────────────────────────────────
    logging.info("🔍 Validating model configs...")
    configs = []
    for path in paths_load:
        _, config = wrapper_tools.load_and_massage_config_file(
            os.path.join(path, "config.yaml"), path
        )
        configs.append(config)

    config_ref = configs[0]
    for i, cfg in enumerate(configs[1:], 1):
        logging.debug(f"🔍 Comparing config 0 and config {i}")
        if not safe_compare(cfg['data'], config_ref['data']):
            raise ValueError(f"🚫 Data config mismatch between model 0 and model {i}")

    config_data = config_ref["data"]
    keys_xx = config_data["features_labels_options"]["keys_xx"]

    # Extract paths and options
    path_save = config_ref['training']['path_save']
    means, stds = save_load_tools.load_means_stds(path_save)

    data_paths = config_data["data_paths"]
    root_path = data_paths["root_path"]
    load_JPAS_data = data_paths["load_JPAS_data"]
    load_DESI_data = data_paths["load_DESI_data"]
    random_seed_load = data_paths["random_seed_load"]

    clean_opts = config_data["dict_clean_data_options"]
    split_opts = config_data["dict_split_data_options"]

    # ─────────────────────────────────────────────────────────────
    # Load and preprocess shared data
    # ─────────────────────────────────────────────────────────────
    logging.info("\n\n1️⃣: Loading datasets from disk...")
    DATA = loading_tools.load_dsets(root_path, load_JPAS_data, load_DESI_data, random_seed_load)

    logging.info("\n\n2️⃣: Cleaning and masking data...")
    DATA = cleaning_tools.clean_and_mask_data(DATA=DATA, **clean_opts)

    logging.info("\n\n3️⃣: Crossmatching JPAS and DESI TARGETIDs...")
    Dict_LoA = {"both": {}, "only": {}}
    _, _, _, Dict_LoA["only"]["DESI"], Dict_LoA["only"]["JPAS"], Dict_LoA["both"]["DESI"], Dict_LoA["both"]["JPAS"] = \
        crossmatch_tools.crossmatch_IDs_two_datasets(DATA["DESI"]["TARGETID"], DATA["JPAS"]["TARGETID"])

    logging.info("\n\n4️⃣: Splitting data into train/val/test...")
    Dict_LoA_split = {"both": {}, "only": {}}

    # Always split 'both' JPAS and DESI (assumed always needed)
    Dict_LoA_split["both"]["JPAS"] = process_dset_splits.split_LoA(
        Dict_LoA["both"]["JPAS"],
        split_opts["train_ratio_both"], split_opts["val_ratio_both"], split_opts["test_ratio_both"],
        seed=split_opts["random_seed_split_both"]
    )
    Dict_LoA_split["both"]["DESI"] = process_dset_splits.split_LoA(
        Dict_LoA["both"]["DESI"],
        split_opts["train_ratio_both"], split_opts["val_ratio_both"], split_opts["test_ratio_both"],
        seed=split_opts["random_seed_split_both"]
    )
    # Split 'only' DESI if available
    if "DESI" in Dict_LoA["only"]:
        Dict_LoA_split["only"]["DESI"] = process_dset_splits.split_LoA(
            Dict_LoA["only"]["DESI"],
            split_opts["train_ratio_only_DESI"], split_opts["val_ratio_only_DESI"], split_opts["test_ratio_only_DESI"],
            seed=split_opts["random_seed_split_only_DESI"]
        )
    # Optionally split 'only' JPAS if available
    if "JPAS" in Dict_LoA["only"]:
        Dict_LoA_split["only"]["JPAS"] = process_dset_splits.split_LoA(
            Dict_LoA["only"]["JPAS"],
            split_opts.get("train_ratio_only_JPAS", 0.7),
            split_opts.get("val_ratio_only_JPAS", 0.15),
            split_opts.get("test_ratio_only_JPAS", 0.15),
            seed=split_opts.get("random_seed_split_only_JPAS", 42)
        )
    
    logging.info("\n\n5️⃣: Load and normalize data...")
    xx_dict, yy_dict = {}, {}

    for split in ["val", "test"]:
        xx_dict[split] = {}
        yy_dict[split] = {}

        for loader in define_dataset_loaders_keys:
            assert isinstance(loader, str), f"❌ Loader key is not a string: {loader}"
            source = "DESI" if "DESI" in loader else "JPAS"
            split_type = "both" if "matched" in loader or "combined" in loader else "only"

            if split_type not in Dict_LoA_split or source not in Dict_LoA_split[split_type]:
                logging.warning(f"⚠️ Skipping loader '{loader}' because '{source}' not found in split type '{split_type}'")
                continue

            subset = Dict_LoA_split[split_type][source].get(split, [])
            if not subset:
                logging.warning(f"⚠️ No entries found for split '{split}' in loader '{loader}'")
                continue

            LoA, xx, yy = process_dset_splits.extract_data_using_LoA(subset, DATA[source], keys_xx, keys_yy)

            stacked_features = []
            for i, k in enumerate(xx):
                arr = np.asarray(xx[k])
                if arr.size == 0:
                    continue
                normed = (arr - means[i]) / stds[i]
                stacked_features.append(np.atleast_2d(normed).reshape(arr.shape[0], -1))

            if stacked_features:
                xx_stacked = np.concatenate(stacked_features, axis=1)
            else:
                logging.warning(f"⚠️ Feature stack is empty for split={split}, loader={loader}. Creating dummy array.")
                xx_stacked = np.empty((0, sum([np.prod(np.shape(xx[k])[1:]) for k in keys_xx])))

            xx_dict[split][str(loader)] = torch.tensor(xx_stacked, dtype=torch.float32)
            yy_dict[split][str(loader)] = yy

    # ─────────────────────────────────────────────────────────────
    # Evaluate each model and collect results
    # ─────────────────────────────────────────────────────────────
    logging.info("\n\n Evaluate each model and collect results...")
    out = {}
    for model_idx, path in enumerate(paths_load):
        _, model_encoder = save_load_tools.load_model_from_checkpoint(
            os.path.join(path, "model_encoder.pt"), model_building_tools.create_mlp)
        _, model_downstream = save_load_tools.load_model_from_checkpoint(
            os.path.join(path, "model_downstream.pt"), model_building_tools.create_mlp)

        out[model_idx] = {}
        for key in return_keys:
            split, loader = key.split("_", maxsplit=1)
            xx_input = xx_dict[split][loader]

            with torch.no_grad():
                features = model_encoder(xx_input)
                logits = model_downstream(features)
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

            out[model_idx][key] = {
                "true": yy_dict[split][loader]["SPECTYPE_int"],
                "prob": probs,
                "label": np.argmax(probs, axis=1),
                "features": features.cpu().numpy(),
                "xx": xx_input.cpu().numpy(),
                "TARGETID": yy_dict[split][loader]["TARGETID"],
                "DESI_FLUX_R": yy_dict[split][loader]["DESI_FLUX_R"]
            }

    return out


def add_magnitude_bins_to_results(
    out_dict,
    magnitude_key="DESI_FLUX_R",
    mag_bin_edges=(17, 19, 21, 22, 22.5),
    output_key="MAG_BIN_ID"
):
    """
    Adds magnitude bin indices to each dataset entry in the out_dict.
    The binning is done using the R-band magnitude computed from DESI_FLUX_R.

    Parameters
    ----------
    out_dict : dict
        Dictionary returned by evaluate_results_from_load_paths.
    magnitude_key : str
        Key inside each subdict to use for flux (to convert to magnitude).
    mag_bin_edges : tuple or list
        Magnitude bin edges (right-exclusive).
    output_key : str
        New key to store the bin index (-1 if not assigned).
    """
    import numpy as np

    bin_edges = np.array(mag_bin_edges)

    for model_idx in out_dict:
        for key in out_dict[model_idx]:
            flux = out_dict[model_idx][key][magnitude_key]
            # Convert flux to magnitude
            magnitude = np.full_like(flux, np.nan, dtype=np.float32)
            valid = flux > 0
            magnitude[valid] = 22.5 - 2.5 * np.log10(flux[valid])

            # Compute bin indices
            bin_indices = np.full_like(magnitude, -1, dtype=int)
            for i in range(len(bin_edges) - 1):
                in_bin = (magnitude >= bin_edges[i]) & (magnitude < bin_edges[i + 1])
                bin_indices[in_bin] = i

            # Store bin index in the result dictionary
            out_dict[model_idx][key][output_key] = bin_indices

    return out_dict