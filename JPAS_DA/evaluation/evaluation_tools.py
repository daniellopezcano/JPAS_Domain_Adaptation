import numpy as np
import logging
import torch

from copy import deepcopy
import os
import sys
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

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

def plot_overall_deltaF1_two_comparisons(
    y_true_src,              # DESI_mocks_Raul test labels
    y_pred_src_noDA,         # probs no-DA on source test
    y_true_tgt,              # JPAS_x_DESI_Raul test labels
    y_pred_tgt_noDA,         # probs no-DA on target test
    y_pred_tgt_DA,           # probs DA on target test
    class_names,
    *,
    title="ΔF1 per class (overall)",
    colors=("royalblue", "darkorange"),           # (Target no-DA − Source no-DA, Target DA − Target no-DA)
    labels=("Target no-DA − Source no-DA", "Target DA − Target no-DA"),
    figsize=(11, 6),
    ylim=(-0.5, 0.5),
    alpha=0.9,
    edgecolor="black",
    bar_width=0.36,
    legend_kwargs=None,                            # e.g. {"loc":"upper right", "frameon":True}
    show=True,
    save_dir=None, save_format="png", save_dpi=200, filename="deltaF1_overall_combined"
):
    """
    One grouped-bar figure with two bars per class:
      - ΔF1_1 = F1(Target no-DA) − F1(Source no-DA)
      - ΔF1_2 = F1(Target DA)    − F1(Target no-DA)
    """
    n_classes = len(class_names)

    def _f1_per_class(y_true, y_pred_probs):
        if len(y_true) == 0:
            return np.zeros(n_classes, dtype=float)
        y_pred = np.argmax(y_pred_probs, axis=1)
        return f1_score(y_true, y_pred, labels=np.arange(n_classes), average=None, zero_division=0)

    # --- Compute per-class F1s
    f1_src_noDA = _f1_per_class(y_true_src,  y_pred_src_noDA)
    f1_tgt_noDA = _f1_per_class(y_true_tgt,  y_pred_tgt_noDA)
    f1_tgt_DA   = _f1_per_class(y_true_tgt,  y_pred_tgt_DA)

    # --- Deltas with your convention
    delta_target_noDA_minus_source_noDA = f1_tgt_noDA - f1_src_noDA
    delta_target_DA_minus_target_noDA   = f1_tgt_DA   - f1_tgt_noDA

    # --- Plot
    x = np.arange(n_classes)
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(x - bar_width/2, delta_target_noDA_minus_source_noDA, width=bar_width,
           color=colors[0], alpha=alpha, edgecolor=edgecolor, label=labels[0])
    ax.bar(x + bar_width/2, delta_target_DA_minus_target_noDA, width=bar_width,
           color=colors[1], alpha=alpha, edgecolor=edgecolor, label=labels[1])

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("ΔF1")
    ax.set_xlabel("Class")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    if legend_kwargs is None:
        legend_kwargs = {"loc": "upper right", "frameon": True}
    ax.legend(**legend_kwargs)

    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    # Save if requested
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_dir) / f"{filename}.{save_format}"
        fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

def radar_plot(
    dict_radar: dict,
    class_names,
    *,
    title: str = None,
    title_pad: float = 20,
    figsize=(10, 10),
    theta_offset=np.pi/2,
    theta_direction=-1,
    r_ticks=(0.2, 0.4, 0.6, 0.8, 1.0),
    r_lim=(0.0, 1.0),
    tick_labelsize=14,
    radial_labelsize=12,
    linewidth_default=2.0,
    show_legend=True,
    legend_kwargs=None,
    close_line=True,
    # NEW: fill controls (global defaults; can be overridden per case in plot_kwargs)
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
                edgecolors=mec if mec is not None else face if fill_on else (color or line.get_color()),
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


def evaluate_all_plots_by_mag_bins(
    masks_magnitudes: dict,
    yy: dict,
    probs: dict,
    class_names,
    *,
    dict_radar_styles: dict = None,   # template dict to copy plot_kwargs (values under ['plot_kwargs'])
    radar_title_base: str = "F1 Radar Plot",
    radar_kwargs: dict = None,        # forwarded to radar_plot
    colors=None,                      # e.g., ['blue','green','orange','red']  (per-bin color)
    colormaps=None,                   # e.g., [plt.cm.Blues, plt.cm.Greens, plt.cm.YlOrBr, plt.cm.Reds] (per-bin cmap)
    show: bool = True,
    # ------- saving controls -------
    save_dir: str = None,             # folder to save figures; if None, don't save
    save_format: str = "png",
    save_dpi: int = 200,
    close_after_save: bool = False,   # close figs after saving (useful when show=False in loops)
    required_entries = [
        ("Mocks", "Test"),
        ("JPAS x DESI", "Test"),
        ("JPAS x DESI", "Train"),  # only to align available bins; not plotted
    ]
):
    """
    Per magnitude bin:
      - Confusion matrices for:
          * no-DA Source-Test (Mocks)
          * no-DA Target-Test (JPAS x DESI)
          * DA   Target-Test (JPAS x DESI)
      - Radar plot for the 3 cases above (all lines use the bin's color; styles differentiate cases)
      - Two set-performance comparisons (both use the bin's color):
          * Source no-DA (Mocks Test) vs Target no-DA (JPAS Test)
          * Target no-DA (JPAS Test) vs DA (JPAS Test)

    NEW (combined):
      - One combined radar plot overlaying all magnitude bins (color-coded by bin)
      - Two combined ΔF1 histograms (one per comparison), grouped by class with one bar per bin.

    DA-Train plots and TPR comparison are intentionally omitted.
    """
    # ---------------- helpers ----------------
    def _bin_tag(lo, hi):
        fmt = lambda v: str(v).replace('.', 'p')
        return f"mag_{fmt(lo)}-{fmt(hi)}"

    def _sanitize(name):
        return re.sub(r"[^A-Za-z0-9._\-]+", "_", name)

    def _save_current(fig, filename):
        if not save_dir:
            return
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / filename
        fig.savefig(fpath, dpi=save_dpi, bbox_inches="tight")

    def _f1_per_class(y_true, y_pred_probs, n_classes):
        if len(y_true) == 0:
            return np.zeros(n_classes, dtype=float)
        y_pred = np.argmax(y_pred_probs, axis=1)
        return f1_score(y_true, y_pred, labels=np.arange(n_classes), average=None, zero_division=0)

    def _plot_combined_deltaF1_hist(
            delta_by_bin: dict, *, title: str, colors_for_bins: list,
            class_names: list, ylim=(-0.5, 0.5), figsize=(11, 6),
            required_entries = [
                ("Mocks", "Test"),
                ("JPAS x DESI", "Test"),
                ("JPAS x DESI", "Train"),  # only to align available bins; not plotted
            ]
        ):
        """
        delta_by_bin: dict {bin_label: np.ndarray(n_classes,)}
        One grouped bar chart: per class, bars for each bin (colored by bin color).
        """
        bins_list = list(delta_by_bin.keys())
        B = len(bins_list)
        C = len(class_names)
        x = np.arange(C)
        width = 0.8 / max(B, 1)  # pack bars within class group

        fig, ax = plt.subplots(figsize=figsize)
        for b_idx, bin_lbl in enumerate(bins_list):
            deltas = delta_by_bin[bin_lbl]
            # offset within the class group
            offset = (b_idx - (B - 1) / 2) * width
            ax.bar(x + offset, deltas, width=width, color=colors_for_bins[b_idx % len(colors_for_bins)],
                   edgecolor="black", alpha=0.85, label=bin_lbl)

        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=20, ha="right")
        ax.set_ylabel("ΔF1")
        ax.set_xlabel("Class")
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.legend(title="Magnitude bins", ncol=B, fontsize=9, frameon=True)
        fig.tight_layout()
        return fig, ax

    # -----------------------------------------
    if colors is None:
        colors = ['blue', 'green', 'orange', 'red']
    if colormaps is None:
        colormaps = [plt.cm.Blues, plt.cm.Greens, plt.cm.YlOrBr, plt.cm.Reds]

    # Intersect bins across required entries
    bins_sets = []
    for entry in required_entries:
        entry_bins = [k for k in masks_magnitudes.get(entry, {}).keys() if isinstance(k, tuple) and len(k) == 2]
        if not entry_bins:
            print(f"[WARN] No bins found for entry {entry}; skipping all.")
            return
        bins_sets.append(set(entry_bins))
    common_bins = sorted(set.intersection(*bins_sets), key=lambda x: (x[0], x[1]))
    if not common_bins:
        print("[WARN] No common magnitude bins across required entries.")
        return

    # Pull plot styles (if provided) for radar lines
    styles = {}
    if dict_radar_styles is not None:
        for key, spec in dict_radar_styles.items():
            if "plot_kwargs" in spec:
                styles[key] = dict(spec["plot_kwargs"])  # copy

    # Defaults if not provided
    styles.setdefault("Source-Test (Mocks) no-DA", {
        "linestyle": "--", "linewidth": 2.0, "color": "orange",
        "marker": "+", "markersize": 8.0, "label": "Source-Test (Mocks) no-DA"
    })
    styles.setdefault("Target-Test (JPAS x DESI) no-DA", {
        "linestyle": "-.", "linewidth": 2.0, "color": "crimson",
        "marker": "x", "markersize": 8.0, "label": "Target-Test (JPAS x DESI) no-DA"
    })
    styles.setdefault("Target-Test (JPAS x DESI) DA", {
        "linestyle": "-", "linewidth": 2.0, "color": "limegreen",
        "marker": "o", "markersize": 8.0, "label": "Target-Test (JPAS x DESI) DA"
    })

    # Defaults for radar kwargs (can be overridden via radar_kwargs)
    _radar_kwargs = dict(
        figsize=(8, 8),
        theta_offset=np.pi / 2,
        r_ticks=(0.1, 0.3, 0.5, 0.7, 0.9),
        r_lim=(0.0, 1.0),
        tick_labelsize=16,
        radial_labelsize=12,
        show_legend=True,
        legend_kwargs={
            "loc": "upper left", "bbox_to_anchor": (0.73, 1.0), "fontsize": 9, "ncol": 1,
            "title": "Evaluation Cases", "frameon": True, "fancybox": True,
            "shadow": True, "borderaxespad": 0.0,
        },
        title_pad=20,
    )
    if radar_kwargs:
        _radar_kwargs.update(radar_kwargs)

    # === COMBINED ACCUMULATORS ===
    # 1) Combined radar: build a giant dict_radar with entries (case×bin) each colored by the bin color.
    dict_radar_combined = {}

    # 2) Combined ΔF1 per comparison, per bin
    delta_tgt_vs_src_noDA_by_bin = {}  # bin_label -> (n_classes,)
    delta_tgt_DA_vs_noDA_by_bin = {}   # bin_label -> (n_classes,)

    n_classes = len(class_names)

    for bin_idx, (lo, hi) in enumerate(common_bins):
        bin_label = f"({lo}, {hi}]"
        tag = _bin_tag(lo, hi)
        print(f"\n=== Magnitude bin {bin_label} ===")

        # Per-bin color/cmap
        color_this_bin = colors[bin_idx % len(colors)]
        cmap_this_bin  = colormaps[bin_idx % len(colormaps)]

        # --- Masks ---
        m_mocks_test = masks_magnitudes[("Mocks", "Test")][(lo, hi)]
        m_jpas_test  = masks_magnitudes[("JPAS x DESI", "Test")][(lo, hi)]

        # --- Slices ---
        y_true_src = yy["DESI_mocks_Raul"]["test"]["SPECTYPE_int"][m_mocks_test]
        y_pred_src_noDA = probs["no-DA"]["DESI_mocks_Raul"]["test"][m_mocks_test]

        y_true_tgt_test = yy["JPAS_x_DESI_Raul"]["test"]["SPECTYPE_int"][m_jpas_test]
        y_pred_tgt_test_noDA = probs["no-DA"]["JPAS_x_DESI_Raul"]["test"][m_jpas_test]
        y_pred_tgt_test_DA = probs["DA"]["JPAS_x_DESI_Raul"]["test"][m_jpas_test]

        def _nonempty(*arrs):
            return all(len(a) > 0 for a in arrs)

        # --- Confusion matrices (3) ---
        if _nonempty(y_true_src, y_pred_src_noDA):
            res = plot_confusion_matrix(
                y_true_src, y_pred_src_noDA,
                class_names=class_names, cmap=cmap_this_bin,
                title=f"no-DA Source-Test (Mocks)  {bin_label}"
            )
            fig_cm = res[0] if (isinstance(res, tuple) and hasattr(res[0], "savefig")) else (res if hasattr(res, "savefig") else plt.gcf())
            _save_current(fig_cm, _sanitize(f"cm_noDA_Source-Test_Mocks_{tag}.{save_format}"))
            if close_after_save and save_dir: plt.close(fig_cm)

        if _nonempty(y_true_tgt_test, y_pred_tgt_test_noDA):
            res = plot_confusion_matrix(
                y_true_tgt_test, y_pred_tgt_test_noDA,
                class_names=class_names, cmap=cmap_this_bin,
                title=f"no-DA Target-Test (JPAS x DESI)  {bin_label}"
            )
            fig_cm = res[0] if (isinstance(res, tuple) and hasattr(res[0], "savefig")) else (res if hasattr(res, "savefig") else plt.gcf())
            _save_current(fig_cm, _sanitize(f"cm_noDA_Target-Test_JPASxDESI_{tag}.{save_format}"))
            if close_after_save and save_dir: plt.close(fig_cm)

        if _nonempty(y_true_tgt_test, y_pred_tgt_test_DA):
            res = plot_confusion_matrix(
                y_true_tgt_test, y_pred_tgt_test_DA,
                class_names=class_names, cmap=cmap_this_bin,
                title=f"DA Target-Test (JPAS x DESI)  {bin_label}"
            )
            fig_cm = res[0] if (isinstance(res, tuple) and hasattr(res[0], "savefig")) else (res if hasattr(res, "savefig") else plt.gcf())
            _save_current(fig_cm, _sanitize(f"cm_DA_Target-Test_JPASxDESI_{tag}.{save_format}"))
            if close_after_save and save_dir: plt.close(fig_cm)

        # --- Per-bin radar (as before) ---
        dict_radar_bin = {
            "Source-Test (Mocks) no-DA": {
                "y_true": y_true_src,
                "y_pred": y_pred_src_noDA,
                "plot_kwargs": {**styles["Source-Test (Mocks) no-DA"], "color": color_this_bin},
            },
            "Target-Test (JPAS x DESI) no-DA": {
                "y_true": y_true_tgt_test,
                "y_pred": y_pred_tgt_test_noDA,
                "plot_kwargs": {**styles["Target-Test (JPAS x DESI) no-DA"], "color": color_this_bin},
            },
            "Target-Test (JPAS x DESI) DA": {
                "y_true": y_true_tgt_test,
                "y_pred": y_pred_tgt_test_DA,
                "plot_kwargs": {**styles["Target-Test (JPAS x DESI) DA"], "color": color_this_bin},
            },
        }
        if any(len(v["y_true"]) > 0 for v in dict_radar_bin.values()):
            fig_radar, ax_radar = radar_plot(
                dict_radar=dict_radar_bin,
                class_names=class_names,
                title=f"{radar_title_base}  {bin_label}",
                **_radar_kwargs,
            )
            _save_current(fig_radar, _sanitize(f"radar_{tag}.{save_format}"))
            if close_after_save and save_dir: plt.close(fig_radar)

        # --- Accumulate for COMBINED radar: split into (case×bin) items ---
        # Keep linestyles by case, override color by bin, and append bin tag to label.
        for case_key in ("Source-Test (Mocks) no-DA",
                         "Target-Test (JPAS x DESI) no-DA",
                         "Target-Test (JPAS x DESI) DA"):
            if case_key == "Source-Test (Mocks) no-DA" and _nonempty(y_true_src, y_pred_src_noDA):
                dict_radar_combined[f"{case_key} [{bin_label}]"] = {
                    "y_true": y_true_src,
                    "y_pred": y_pred_src_noDA,
                    "plot_kwargs": {**styles[case_key], "color": color_this_bin, "label": f"{case_key} [{bin_label}]"},
                }
            elif case_key == "Target-Test (JPAS x DESI) no-DA" and _nonempty(y_true_tgt_test, y_pred_tgt_test_noDA):
                dict_radar_combined[f"{case_key} [{bin_label}]"] = {
                    "y_true": y_true_tgt_test,
                    "y_pred": y_pred_tgt_test_noDA,
                    "plot_kwargs": {**styles[case_key], "color": color_this_bin, "label": f"{case_key} [{bin_label}]"},
                }
            elif case_key == "Target-Test (JPAS x DESI) DA" and _nonempty(y_true_tgt_test, y_pred_tgt_test_DA):
                dict_radar_combined[f"{case_key} [{bin_label}]"] = {
                    "y_true": y_true_tgt_test,
                    "y_pred": y_pred_tgt_test_DA,
                    "plot_kwargs": {**styles[case_key], "color": color_this_bin, "label": f"{case_key} [{bin_label}]"},
                }

        # --- Compute ΔF1 per bin for the two comparisons ---
        if _nonempty(y_true_src, y_pred_src_noDA) and _nonempty(y_true_tgt_test, y_pred_tgt_test_noDA):
            f1_src_noDA = _f1_per_class(y_true_src, y_pred_src_noDA, n_classes)
            f1_tgt_noDA = _f1_per_class(y_true_tgt_test, y_pred_tgt_test_noDA, n_classes)
            delta_tgt_vs_src_noDA_by_bin[bin_label] = f1_tgt_noDA - f1_src_noDA

        if _nonempty(y_true_tgt_test, y_pred_tgt_test_noDA) and _nonempty(y_true_tgt_test, y_pred_tgt_test_DA):
            f1_tgt_noDA = _f1_per_class(y_true_tgt_test, y_pred_tgt_test_noDA, n_classes)
            f1_tgt_DA   = _f1_per_class(y_true_tgt_test, y_pred_tgt_test_DA, n_classes)
            delta_tgt_DA_vs_noDA_by_bin[bin_label] = f1_tgt_DA - f1_tgt_noDA

        if show and not close_after_save:
            plt.show()

    # ================== COMBINED PLOTS ==================

    # --- Combined radar over all bins ---
    if dict_radar_combined:
        fig_cr, ax_cr = radar_plot(
            dict_radar=dict_radar_combined,
            class_names=class_names,
            title=f"{radar_title_base} — Combined bins",
            **_radar_kwargs,
        )
        _save_current(fig_cr, _sanitize(f"radar_combined.{save_format}"))
        if close_after_save and save_dir: plt.close(fig_cr)

    # --- Combined ΔF1 histograms ---
    # Colors per bin in order of bins encountered in the dict
    def _colors_for_bins(delta_dict):
        bins_list = list(delta_dict.keys())
        return [colors[common_bins.index(tuple(map(float, b.strip("()").split(", ")))) % len(colors)]
                if isinstance(b, str) and ", " in b
                else colors[i % len(colors)]
                for i, b in enumerate(bins_list)]

    if delta_tgt_vs_src_noDA_by_bin:
        bins_colors = [colors[i % len(colors)] for i in range(len(delta_tgt_vs_src_noDA_by_bin))]
        fig_d1, ax_d1 = _plot_combined_deltaF1_hist(
            delta_tgt_vs_src_noDA_by_bin,
            title="JPAS Obs. VS Mocks (no-DA)",
            colors_for_bins=bins_colors,
            class_names=class_names,
            ylim=(-0.78, 0.15),
            required_entries=required_entries
        )
        _save_current(fig_d1, _sanitize(f"deltaF1_Source_vs_Target_noDA_combined.{save_format}"))
        if close_after_save and save_dir: plt.close(fig_d1)

    if delta_tgt_DA_vs_noDA_by_bin:
        bins_colors = [colors[i % len(colors)] for i in range(len(delta_tgt_DA_vs_noDA_by_bin))]
        fig_d2, ax_d2 = _plot_combined_deltaF1_hist(
            delta_tgt_DA_vs_noDA_by_bin,
            title="DA VS no-DA (JPAS Obs.)",
            colors_for_bins=bins_colors,
            class_names=class_names,
            ylim=(-0.15, 0.35),
            required_entries=required_entries
        )
        _save_current(fig_d2, _sanitize(f"deltaF1_Target_noDA_vs_DA_combined.{save_format}"))
        if close_after_save and save_dir: plt.close(fig_d2)

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
    y_min_Delta_F1=-0.24, y_max_Delta_F1=0.24,
    name_1="Set 1", name_2="Set 2",
    plot_ROC_curves=True,
    color='royalblue',
    title_fontsize=22,
    f1_save_path=None  # <- NEW: optional path to save the ΔF1 figure as PDF
):
    """
    Compare two sets' performance and plot ΔF1 per class.

    Parameters
    ----------
    ...
    f1_save_path : str or None
        If provided, saves the ΔF1 figure as a PDF. Can be a filename ending in .pdf
        or a directory path (in which case a default filename is used).
    """
    import os

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
        "Macro Precision": (
            precision_score(yy_true_1, yy_pred_1, average='macro', zero_division=0),
            precision_score(yy_true_2, yy_pred_2, average='macro', zero_division=0),
            True
        ),
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

    # ---- ΔF1 bar plot ----
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(class_names, f1_2 - f1_1, color=color)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel("Δ F1-score")
    ax.set_title(f"{name_2} - {name_1}", fontsize=title_fontsize)
    plt.xticks(rotation=15, ha='right')
    ax.set_ylim(y_min_Delta_F1, y_max_Delta_F1)
    fig.tight_layout()

    # --- Optional save to PDF ---
    if f1_save_path is not None:
        # If no .pdf suffix, or looks like a directory, build a filename
        target_path = f1_save_path
        base, ext = os.path.splitext(target_path)
        if ext.lower() != ".pdf":
            # treat as directory or base without .pdf
            # if it's a directory (existing or intended), ensure it exists
            if (ext == "") and (not os.path.basename(base)):  # path ends with slash-like
                os.makedirs(base, exist_ok=True)
                filename = f"Delta_F1_{name_2.replace(' ', '_')}_minus_{name_1.replace(' ', '_')}.pdf"
                target_path = os.path.join(base, filename)
            else:
                # add .pdf to whatever they passed
                os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
                target_path = base + ".pdf"
        else:
            os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

        fig.savefig(target_path, format="pdf", bbox_inches="tight")

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

    for split in ["train", "val", "test"]:
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

            xx_dict[split][str(loader)] = torch.tensor(xx, dtype=torch.float32)
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