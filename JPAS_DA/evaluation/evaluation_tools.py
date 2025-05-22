import numpy as np
import logging
import torch

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter

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
    title=None
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
    num_classes = len(class_names)

    # Raw confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(yy_true, yy_pred):
        cm[int(t), int(p)] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=cmap)

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

    # Compute precision, recall, F1
    precision = precision_score(yy_true, yy_pred, average=None, zero_division=0)
    recall = recall_score(yy_true, yy_pred, average=None, zero_division=0)
    f1 = f1_score(yy_true, yy_pred, average=None, zero_division=0)

    # Threshold for annotation color
    threshold = cm_percent.max() / 2.0

    for i in range(num_classes):
        for j in range(num_classes):
            count = cm[i, j]
            percent = cm_percent[i, j] * 100 if row_sums[i] != 0 else 0
            text_color = "white" if cm_percent[i, j] > threshold else "black"

            if i == j:
                ax.text(
                    j, i,
                    f"{count}\nTPR:{recall[i]*100:.1f}%\nPPV:{precision[i]*100:.1f}%\nF1:{f1[i]:.2f}",
                    ha="center", va="center", color=text_color, fontsize=10, fontweight='bold'
                )
            else:
                ax.text(j, i, f"{count}\n({percent:.1f}%)", ha="center", va="center", color=text_color, fontsize=11)

    plt.tight_layout()
    plt.show()

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

def plot_multiclass_roc(y_true_1, y_pred_P_1, y_true_2, y_pred_P_2, class_names=None, name_1="Model 1", name_2="Model 2"):

    classes = np.unique(np.concatenate([y_true_1, y_true_2]))
    y_true_1_bin = label_binarize(y_true_1, classes=classes)
    y_true_2_bin = label_binarize(y_true_2, classes=classes)
    colors = get_N_colors(len(classes), plt.cm.tab10)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(classes):
        fpr1, tpr1, _ = roc_curve(y_true_1_bin[:, i], y_pred_P_1[:, i])
        fpr2, tpr2, _ = roc_curve(y_true_2_bin[:, i], y_pred_P_2[:, i])
        auc1 = auc(fpr1, tpr1)
        auc2 = auc(fpr2, tpr2)

        label = f"Class {cls}" if class_names is None else class_names[i]
        ax.plot(fpr1, tpr1, linestyle='-', color=colors[i], label=f"{label} ({name_1}) [AUC={auc1:.2f}]")
        ax.plot(fpr2, tpr2, linestyle='--', color=colors[i], label=f"{label} ({name_2}) [AUC={auc2:.2f}]")

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multiclass ROC Curves")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def compare_sets_performance(
    yy_true_1, yy_pred_P_1,
    yy_true_2, yy_pred_P_2,
    class_names=None,
    name_1="Set 1", name_2="Set 2"
):
    yy_pred_1 = np.argmax(yy_pred_P_1, axis=1)
    yy_pred_2 = np.argmax(yy_pred_P_2, axis=1)

    # Compute per-class recall (for metrics) and per-class F1 (for plot)
    tpr_1 = recall_score(yy_true_1, yy_pred_1, average=None, zero_division=0)
    tpr_2 = recall_score(yy_true_2, yy_pred_2, average=None, zero_division=0)
    f1_1 = f1_score(yy_true_1, yy_pred_1, average=None, zero_division=0)
    f1_2 = f1_score(yy_true_2, yy_pred_2, average=None, zero_division=0)

    metrics = {
        "Accuracy": (accuracy_score(yy_true_1, yy_pred_1), accuracy_score(yy_true_2, yy_pred_2), True),
        "Macro F1": (np.mean(f1_1), np.mean(f1_2), True),
        "Macro TPR": (np.mean(tpr_1), np.mean(tpr_2), True),
        "Macro Precision": (precision_score(yy_true_1, yy_pred_1, average='macro', zero_division=0),
                            precision_score(yy_true_2, yy_pred_2, average='macro', zero_division=0), True),
        "Macro AUROC": (roc_auc_score(yy_true_1, yy_pred_P_1, multi_class='ovo', average='macro') if len(np.unique(yy_true_1)) > 1 else np.nan,
                        roc_auc_score(yy_true_2, yy_pred_P_2, multi_class='ovo', average='macro') if len(np.unique(yy_true_2)) > 1 else np.nan, True),
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
    plt.ylabel(f"Δ F1-score ({name_2} - {name_1})")
    plt.title(f"F1 Gap: {name_2} - {name_1}")
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.show()

    # ROC Curves
    plot_multiclass_roc(
        yy_true_1, yy_pred_P_1,
        yy_true_2, yy_pred_P_2,
        class_names=class_names,
        name_1=name_1,
        name_2=name_2
    )

    return metrics

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

