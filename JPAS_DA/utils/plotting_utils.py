import os
import sys

import numpy as np
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.stats as sp
import scipy.interpolate as interp


def matplotlib_default_config():

    font = {
        'family' : 'serif',
        'weight' : 'normal',
        'size'   : 10
    }
    
    rcnew = {
        "mathtext.fontset" : "cm", 
        "text.usetex": False,

        'figure.frameon': True,
        'axes.linewidth': 2.,

        "axes.titlesize" : 32, 
        "axes.labelsize" : 28,
        "legend.fontsize" : 28,
        'legend.fancybox': True,
        'lines.linewidth': 2.5,

        'xtick.alignment': 'center',
        'xtick.bottom': True,
        'xtick.color': 'black',
        'xtick.direction': 'in',
        'xtick.labelbottom': True,
        'xtick.labelsize': 24, #17.5,
        'xtick.labeltop': False,
        'xtick.major.bottom': True,
        'xtick.major.pad': 6.0,
        'xtick.major.size': 14.0,
        'xtick.major.top': True,
        'xtick.major.width': 1.5,
        'xtick.minor.bottom': True,
        'xtick.minor.pad': 3.4,
        'xtick.minor.size': 7.0,
        'xtick.minor.top': True,
        'xtick.minor.visible': True,
        'xtick.minor.width': 1.0,
        'xtick.top': True,

        'ytick.alignment': 'center_baseline',
        'ytick.color': 'black',
        'ytick.direction': 'in',
        'ytick.labelleft': True,
        'ytick.labelright': False,
        'ytick.labelsize': 24, #17.5,
        'ytick.left': True,
        'ytick.major.left': True,
        'ytick.major.pad': 6.0,
        'ytick.major.right': True,
        'ytick.major.size': 14.0,
        'ytick.major.width': 1.5,
        'ytick.minor.left': True,
        'ytick.minor.pad': 3.4,
        'ytick.minor.right': True,
        'ytick.minor.size': 7.0,
        'ytick.minor.visible': True,
        'ytick.minor.width': 1.0,
        'ytick.right': True
    }
    
    return font, rcnew

def get_N_colors(N, colormap=plt.cm.viridis):
    """
    Generates N distinct colors from a given colormap.

    Parameters:
    - N (int): Number of colors to generate.
    - colormap (matplotlib.colors.Colormap, optional): A Matplotlib colormap to use. Default is 'viridis'.

    Returns:
    - colors (numpy array): An array of RGBA colors.

    Example Usage:
    ```
    colors = get_N_colors(5, plt.cm.plasma)
    print(colors)  # Displays 5 RGBA color values
    ```
    """
    def get_colors(inp, colormap, vmin=None, vmax=None):
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(inp))

    colors = get_colors(np.linspace(1, N, num=N, endpoint=True), colormap)
    return colors

def get_N_markers(N):
    """
    Generates N unique marker styles for plotting.

    Parameters:
    - N (int): Number of marker styles to return.

    Returns:
    - markers (numpy array): An array of N marker styles.

    Example Usage:
    ```
    markers = get_N_markers(5)
    print(markers)  # Displays 5 different marker styles
    ```
    """
    all_markers = np.array([
        'o', 'x', 's', 'v', '*', 'P', '^', '<', '>', '1', '2', '3', '4', '8', 'p', 'h', 'H', 
        '+', 'D', 'd', '|', '_', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    ])
    markers = all_markers[np.arange(N)]
    return markers

def get_N_linestyles(N):
    """
    Generates N unique line styles for plotting.

    Parameters:
    - N (int): Number of line styles to return.

    Returns:
    - linestyles (list): A list of N tuples defining line styles.

    Example Usage:
    ```
    linestyles = get_N_linestyles(5)
    print(linestyles)  # Displays 5 different line styles
    ```
    """
    all_linestyles = [
        (0, ()),               # Solid line
        (0, (5, 2)),           # Dashed
        (0, (1, 1)),           # Dotted
        (0, (3, 1, 1, 1)),     # Dash-dot
        (0, (4, 2, 10, 2)),    # Long dash
        (0, (3, 5, 1, 5)),     # Custom mixed pattern
        (0, (1, 10)),          # Sparse dots
        (0, (1, 1)),           # Densely dotted
        (5, (10, 3)),          # Dotted-dashed
        (0, (5, 10)),          # Large dashed
        (0, (5, 1)),           # Short dashed
        (0, (3, 10, 1, 10)),   # Complex pattern 1
        (0, (3, 5, 1, 5, 1, 5)),  # Complex pattern 2
        (0, (3, 10, 1, 10, 1, 10)),  # Complex pattern 3
        (0, (3, 1, 1, 1, 1, 1))  # Dash-dot-dot
    ]
    linestyles = all_linestyles[:N]
    return linestyles

def plot_2d_classification_with_kde(X_all, y_all, title="2D Data with Class-Conditional PDFs", class_color_dict=None):
    """
    Plot a 2D scatter distribution with class-colored points, class-conditional KDE contours,
    global KDE, and marginal distributions on top and right.

    Parameters
    ----------
    X_all : ndarray of shape (N, 2)
        Feature array (only 2D supported).
    y_all : ndarray of shape (N,)
        Integer class labels.
    title : str
        Title of the entire figure.
    """
    unique_labels = np.unique(y_all)
    if class_color_dict is None:
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))
        class_color_dict = {label: cmap(i) for i, label in enumerate(unique_labels)}
    colors = [class_color_dict[label] for label in unique_labels]

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=(6, 2), height_ratios=(2, 6),
                          left=0.1, right=0.95, bottom=0.1, top=0.9, hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # === Global 2D KDE (independent of class) ===
    nbins_contour = 60
    x = X_all[:, 0]
    y = X_all[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xi, yi = np.mgrid[xmin:xmax:nbins_contour*1j, ymin:ymax:nbins_contour*1j]
    values = np.vstack([x, y])
    kernel = sp.gaussian_kde(values)
    positions = np.vstack([xi.ravel(), yi.ravel()])
    zi = np.reshape(kernel(positions).T, xi.shape)
    zi /= zi.sum()

    t = np.linspace(0, zi.max(), 1000)
    integral = ((zi >= t[:, None, None]) * zi).sum(axis=(1, 2))
    f = interp.interp1d(integral, t)
    t_contours = f(np.array([1 - 0.68]))

    ax_main.pcolormesh(xi, yi, zi, shading='auto', cmap='Blues', alpha=0.99)
    # ax_main.contour(zi.T, t_contours, extent=[xmin, xmax, ymin, ymax], colors='darkblue', linewidths=1.5, linestyles='--')

    # === Scatter and KDE contours per class ===
    for ii, label in enumerate(unique_labels):
        mask = y_all == label
        x = X_all[mask, 0]
        y = X_all[mask, 1]
        ax_main.scatter(x, y, s=10, alpha=0.4, color=colors[ii], edgecolor='black', linewidth=0.05)

        values = np.vstack([x, y])
        kernel = sp.gaussian_kde(values)
        xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
        zi = kernel(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        zi /= zi.sum()

        t = np.linspace(0, zi.max(), 1000)
        integral = ((zi >= t[:, None, None]) * zi).sum(axis=(1, 2))
        f = interp.interp1d(integral, t, bounds_error=False, fill_value="extrapolate")
        t_contours = f([1 - 0.68])
        ax_main.contour(xi, yi, zi, levels=t_contours, colors=[colors[ii]], linewidths=2, linestyles='-')

    # === Top marginal (Feature 1) ===
    for ii, label in enumerate(unique_labels):
        mask = y_all == label
        x = X_all[mask, 0]
        kde = sp.gaussian_kde(x)
        x_plot = np.linspace(x.min(), x.max(), 200)
        ax_top.plot(x_plot, kde(x_plot), color=colors[ii], lw=2, label=f"Class {label}")
    ax_top.set_ylabel("Density", fontsize=14)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, linestyle='--', alpha=0.4)

    # === Right marginal (Feature 2) ===
    for ii, label in enumerate(unique_labels):
        mask = y_all == label
        y = X_all[mask, 1]
        kde = sp.gaussian_kde(y)
        y_plot = np.linspace(y.min(), y.max(), 200)
        ax_right.plot(kde(y_plot), y_plot, color=colors[ii], lw=2)
    ax_right.set_xlabel("Density", fontsize=14)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, linestyle='--', alpha=0.4)

    # === Formatting main plot ===
    ax_main.set_xlabel(r"$\mathrm{Feature~1}$", fontsize=18)
    ax_main.set_ylabel(r"$\mathrm{Feature~2}$", fontsize=18)
    ax_main.tick_params(axis='both', labelsize=14)
    ax_main.axhline(0, ls=':', lw=0.6, color='black', alpha=0.3)
    ax_main.axvline(0, ls=':', lw=0.6, color='black', alpha=0.3)
    ax_main.grid(True, linestyle='--', alpha=0.3)

    # Legend
    legend_handles = [
        mpl.lines.Line2D([0], [0], marker='o', color='w', label=f"Class {label}",
                         markerfacecolor=colors[ii], markersize=10, markeredgecolor='black', lw=0)
        for ii, label in enumerate(unique_labels)
    ]
    ax_main.legend(handles=legend_handles, loc='upper left', fontsize=13, fancybox=True, shadow=True,
                   title="Class", title_fontsize=14)

    fig.suptitle(title, fontsize=20)
    
    return fig, ax_main

def plot_training_curves(path_save):
    # Load training data
    losses = np.loadtxt(os.path.join(path_save, 'register.txt'))

    # Find epoch with minimum validation loss
    epoch_min = np.argmin(losses[:, 1])
    val_min = losses[epoch_min, 1]

    # Initialize figure and axis
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 10), sharex=True, gridspec_kw={'hspace': 0.1, 'height_ratios': [3, 1]}
    )

    # Plot training and validation loss
    axes[0].plot(losses[:, 0], color='blue', lw=3, label='Train Loss')
    axes[0].plot(losses[:, 1], color='red', lw=3, linestyle='dashed', label='Validation Loss')

    # Mark minimum validation loss
    axes[0].axvline(epoch_min, color='gray', linestyle='--', lw=1.5, alpha=0.7)
    axes[0].plot(epoch_min, val_min, 'ro', markersize=8)
    axes[0].annotate(f"Min @ {epoch_min}", xy=(epoch_min, val_min),
                     xytext=(epoch_min + 3, val_min + 0.01),
                     fontsize=12, arrowprops=dict(arrowstyle='->', lw=1.5))

    axes[0].set_ylabel("Loss", fontsize=22)
    axes[0].tick_params(axis='both', which='major', labelsize=14)

    # Plot learning rate below
    axes[1].plot(losses[:, 2], color='green', lw=3, label='Learning Rate')
    axes[1].set_xlabel("Epochs", fontsize=22)
    axes[1].set_ylabel("Learning Rate", fontsize=22)
    axes[1].tick_params(axis='both', which='major', labelsize=14)

    # Custom legends
    custom_lines = [
        mpl.lines.Line2D([0], [0], color='blue', ls='-', lw=2),
        mpl.lines.Line2D([0], [0], color='red', ls='--', lw=2),
        mpl.lines.Line2D([0], [0], color='green', ls='-', lw=2)
    ]
    custom_labels = ["Train Loss", "Validation Loss", "Learning Rate"]
    legend = axes[0].legend(custom_lines, custom_labels, loc='upper right', fancybox=True, shadow=True, fontsize=14)
    axes[0].add_artist(legend)

    plt.tight_layout()
    plt.show()

