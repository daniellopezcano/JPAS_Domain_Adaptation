import os
import sys

import numpy as np
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.lines import Line2D

import scipy.stats as sp
import scipy.interpolate as interp

from sklearn.metrics import f1_score

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
        cmap = plt.colormaps.get_cmap('tab10', len(unique_labels))
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

def plot_dual_targetid_logscale(labels_counts_A, labels_counts_B, top_n=15):
    labels_A, counts_A = labels_counts_A
    labels_B, counts_B = labels_counts_B

    # Sort by count descending
    idx_sorted_A = np.argsort(counts_A)[::-1][:top_n]
    idx_sorted_B = np.argsort(counts_B)[::-1][:top_n]

    top_labels_A = labels_A[idx_sorted_A].astype(str)
    top_counts_A = counts_A[idx_sorted_A]

    top_labels_B = labels_B[idx_sorted_B].astype(str)
    top_counts_B = counts_B[idx_sorted_B]

    x = np.arange(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(x, top_counts_A, marker='o', lw=2, label="JPAS")
    ax.plot(x, top_counts_B, marker='s', lw=2, label="JPAS_Ignasi")

    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)", fontsize=12)

    # Bottom x-axis with JPAS labels
    ax.set_xticks(x)
    ax.set_xticklabels(top_labels_A, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel("TARGETIDs from JPAS", fontsize=12)

    # Top x-axis with JPAS_Ignasi labels
    ax_top = ax.secondary_xaxis('top')
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(top_labels_B, rotation=45, ha='left', fontsize=10)
    ax_top.set_xlabel("TARGETIDs from JPAS_Ignasi", fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_title(f"Top {top_n} Most Frequent TARGETIDs", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_pie_chart(labels, counts, title="Pie Chart", class_names=None, custom_colors=None, explode=None):
    """
    Plots a generic pie chart with additional features like displaying count and percentage, custom colors, and explode.

    Parameters:
    - labels (list of str): Labels for each slice of the pie.
    - counts (list or np.ndarray): Counts or values for each slice.
    - title (str): Title of the plot. Default is 'Pie Chart'.
    - class_names (list of str): Names of the classes for labeling. Default is None, in which case `labels` will be used.
    - custom_colors (list or np.ndarray): Colors for the slices. Default is None, and the function will use a colormap.
    - explode (list): List of "explode" values for each slice. Default is None.
    """

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Function to display count and percentage in two lines
    def make_autopct(counts):
        def my_autopct(pct):
            total = sum(counts)
            absolute = int(round(pct * total / 100.0))
            return f"{absolute}\n({pct:.1f}%)"
        return my_autopct

    # If class names are not provided, use the labels themselves
    if class_names is None:
        class_names = labels

    # Use a colormap if custom_colors is not provided
    if custom_colors is None:
        colors = plt.cm.inferno(np.linspace(0., 0.8, len(counts)))
    else:
        colors = custom_colors

    # Default explode values if not provided
    if explode is None:
        explode = [0.05] * len(counts)

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=class_names,
        autopct=make_autopct(counts),
        startangle=140,
        colors=colors,
        explode=explode,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1},
        textprops={'fontsize': 12}
    )

    # Customize font color inside pie
    for autotext in autotexts:
        autotext.set_color("white")

    # General title for the whole plot
    plt.suptitle(f"Distribution of {title}", fontsize=20)

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_neighbors_and_spatial_distribution(kd_counts, R_arcmin, positions, min_density=0.01, max_density=0.5, zoom_in_regions=None):
    """
    Generates two plots: a histogram of the neighbors' counts for different radii and a spatial scatter plot with zoom-in views.

    Parameters:
    kd_counts (list of np.ndarray): Neighbor counts for different radii.
    R_arcmin (np.ndarray): Array of radius values in arcminutes.
    positions (np.ndarray): Array of positions, expected to have shape (n_points, 2) for RA and DEC.
    min_density (float): Minimum density value for the color scale. Default is 0.01.
    max_density (float): Maximum density value for the color scale. Default is 0.5.
    zoom_in_regions (dict): Dictionary containing zoom-in regions with keys:
        - 'min_x_array': Minimum x-values for zoom regions.
        - 'max_x_array': Maximum x-values for zoom regions.
        - 'min_y_array': Minimum y-values for zoom regions.
        - 'max_y_array': Maximum y-values for zoom regions.
        - 'zoom_point_sizes': Point sizes for each zoom region.
        - 'zoom_width': Width of the zoom-in views for each region.
        - 'zoom_height': Height of the zoom-in views for each region.
    """

    # Input validation
    assert isinstance(positions, np.ndarray) and positions.shape[1] == 2, "positions should have shape (n_points, 2)"
    assert len(kd_counts) == len(R_arcmin), "kd_counts and R_arcmin should have the same length"

    # Validate zoom_in_regions dictionary
    if zoom_in_regions:
        required_keys = ['min_x_array', 'max_x_array', 'min_y_array', 'max_y_array', 'zoom_point_sizes', 'zoom_width', 'zoom_height']
        for key in required_keys:
            assert key in zoom_in_regions, f"{key} is missing from zoom_in_regions"
        # Check if zoom_width and zoom_height match the number of zoom regions
        assert len(zoom_in_regions['zoom_width']) == len(zoom_in_regions['min_x_array']), "zoom_width and zoom_height arrays must match the number of zoom regions"

    # Constants for plotting
    cmap = mpl.colormaps['tab20c']
    line_colors = mpl.colormaps['cividis'](np.linspace(0, 1, len(R_arcmin)))

    # === First plot: Histogram with color-coded background ===
    logging.info("Generating histogram plot for neighbors' counts.")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Background color stripes using tab20c colormap
    # Create a smoother gradient by dividing the range of density into steps and mapping it to a colormap
    n_stripes = 100  # Number of stripes
    stripe_width = (max_density - min_density) / n_stripes  # Width of each stripe
    colors = [cmap(i / n_stripes) for i in range(n_stripes)]  # Gradient color steps from colormap
    for ii, color in enumerate(colors):
        ax.axvspan(
            min_density + ii * stripe_width,  # Start of the stripe
            min_density + (ii + 1) * stripe_width,  # End of the stripe
            color=color,  # Color of the stripe
            linewidth=0  # No border around stripes
        )
    
    # Plot each radius's histogram
    for ii, tmp_R_arcmin in enumerate(R_arcmin):
        hist_counts, edges = np.histogram(kd_counts[ii], bins=60, range=(min_density, max_density))
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist_counts, c=line_colors[ii], lw=2, label=fr"$R = {tmp_R_arcmin:.2f}$ arcmin")
    
    ax.set_yscale('log')
    ax.set_xlabel(r'# of Neighbors / Circle Area (KD-Neighbors)', fontsize=14)
    ax.set_ylabel('Normalized Frequency', fontsize=14)
    ax.set_title('Neighbors within $R$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=10, loc='upper right', title="Search Radius", title_fontsize=12)
    plt.tight_layout()

    # === Second plot: Spatial scatter with shared colormap and zooms ===
    if zoom_in_regions:
        logging.info("Generating spatial scatter plot with zoom-in views.")
    else:
        logging.info("Generating spatial scatter plot without zoom-in views.")
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # Main scatter plot
    all_vals = np.array(kd_counts[-1])  # Assuming we are using the last kd_counts for the main plot
    sc = ax.scatter(positions[:, 0], positions[:, 1],
                    c=all_vals,
                    s=1.0,
                    cmap=cmap, vmin=min_density, vmax=max_density,
                    alpha=0.9, edgecolor='none')

    # Shared colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("KD Neighbors Count", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    # Labels and formatting
    ax.set_xlabel('RA [deg]', fontsize=14)
    ax.set_ylabel('DEC [deg]', fontsize=14)
    ax.set_title("Spatial Distribution with Zoom-in Views", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_aspect('equal', adjustable='datalim')

    # Zoom-in views if specified
    if zoom_in_regions:
        min_y_array = np.array(zoom_in_regions['min_y_array'])
        max_y_array = np.array(zoom_in_regions['max_y_array'])
        min_x_array = np.array(zoom_in_regions['min_x_array'])
        max_x_array = np.array(zoom_in_regions['max_x_array'])
        zoom_point_sizes = zoom_in_regions['zoom_point_sizes']
        zoom_width = np.array(zoom_in_regions['zoom_width'])
        zoom_height = np.array(zoom_in_regions['zoom_height'])

        for i in range(len(min_x_array)):
            axins = inset_axes(ax,
                               width=zoom_width[i],
                               height=zoom_height[i],
                               bbox_to_anchor=(zoom_in_regions['zoom_positions'][i][0], zoom_in_regions['zoom_positions'][i][1]),
                               bbox_transform=ax.transAxes,
                               loc='lower left')

            xmask = (positions[:, 0] >= min_x_array[i]) & (positions[:, 0] <= max_x_array[i])
            ymask = (positions[:, 1] >= min_y_array[i]) & (positions[:, 1] <= max_y_array[i])
            mask = xmask & ymask

            axins.scatter(positions[:, 0][mask],
                          positions[:, 1][mask],
                          c=all_vals[mask],
                          s=zoom_point_sizes[i],
                          cmap=cmap, vmin=min_density, vmax=max_density,
                          alpha=0.9, edgecolor='none')

            axins.set_xlim(min_x_array[i], max_x_array[i])
            axins.set_ylim(min_y_array[i], max_y_array[i])
            axins.set_aspect('equal', adjustable='box')
            axins.set_xticks([])
            axins.set_yticks([])

            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1)

    plt.tight_layout()
    plt.show()

def plot_histogram_with_ranges(
        magnitudes, ranges=None, colors=None, bins=200, x_label='DESI Magnitude (R)', title='Histogram of DESI Magnitudes (R)',
        x_range=None, y_range=None
    ):
    """
    Plot a histogram of magnitudes with specified background color ranges and text annotations for percentage of objects in each range.

    Parameters:
    magnitudes (np.ndarray): Array of magnitudes.
    bins (int): Number of bins for the histogram.
    ranges (list of tuples): List of magnitude ranges (lower, upper) for each color.
    colors (list of str): List of colors corresponding to each magnitude range.

    Returns:
    dict: A dictionary of masks for each range.
    """
    # Initialize the dictionary of masks
    masks_dict = {}

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute the histogram
    hist_counts, edges = np.histogram(magnitudes, bins=bins, range=(np.nanmin(magnitudes), np.nanmax(magnitudes)))
    centers = (edges[:-1] + edges[1:]) / 2
    ax.plot(centers, hist_counts, color='k', linewidth=2)

    # Total number of magnitudes
    total_objects = len(magnitudes)

    # Apply the background color stripes using the specified magnitude ranges
    y_position = np.max(hist_counts) * 0.95  # Starting Y position for annotations, slightly below the max y value

    for (lower, upper), color in zip(ranges, colors):
        # Apply the background color bands using axvspan
        ax.axvspan(lower, upper, color=color, alpha=0.5, linewidth=0)
        
        # Store the mask for the current range
        masks_dict[f'{lower}_{upper}'] = (magnitudes >= lower) & (magnitudes < upper)

        # Calculate the percentage of objects within this range
        count_in_range = np.sum(masks_dict[f'{lower}_{upper}'])
        percentage = (count_in_range / total_objects) * 100

        # Add a text annotation with the percentage
        ax.text(
            (lower + upper) / 2,  # X position in the middle of the range
            y_position,  # Y position (slightly below the top of the histogram)
            f'{percentage:.2f}%',  # Text showing percentage
            color='black',  # Text color
            fontsize=12,
            ha='center',  # Horizontal alignment
            va='center',  # Vertical alignment
            bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.5', linewidth=2)  # Box with white background and colored border
        )
        
        # Decrease the y-position for the next annotation to avoid overlap
        y_position -= np.max(hist_counts) * 0.1  # Adjust 10% downward for each annotation

    # Set axis labels and title
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(title, fontsize=16)

    # limit the x-axis range if provided
    if x_range is not None:
        ax.set_xlim(x_range)

    # limit the y-axis range if provided
    if y_range is not None:
        ax.set_ylim(y_range)

    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Ensure tight layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Return the dictionary of masks
    return masks_dict

def plot_multi_histograms_two_legends(
    *,
    series,                           # list of dicts: {"values": array, "survey": str, "cls": str, "label": Optional[str], "bins": Optional[int]}
    ranges=None,                      # list[(low, high)] for shaded bands
    range_colors=None,                # list[str] same length as ranges
    bins=200,                         # default bins if a series doesn't specify its own
    # style maps
    survey_styles=None,               # dict: survey -> linestyle (e.g., {"JPAS_x_DESI_Raul":"-","DESI_mocks_Raul":"--"})
    survey_labels=None,               # dict: survey -> display label (e.g., {"JPAS_x_DESI_Raul":"JPAS Obs.","DESI_mocks_Raul":"Mocks"})
    class_colors=None,                # dict: cls -> color
    class_labels=None,                # dict: cls -> display label
    # axes/title
    x_label='DESI Magnitude (R)',
    y_label="Normalized Frequency (fraction of 'All' in range)",
    title=None,
    x_range=None,
    y_range=None,
    figsize=(7, 5),
    # legends
    show_survey_legend=True,
    survey_legend_loc='upper right',
    show_class_legend=True,
    class_legend_loc='upper left',
    legend_fontsize=12,
    # log scaling
    logy=False,
    # optional label on curve
    label_on_curve=False,
    label_fontsize=11,
    label_x_offset=0.0,
    label_y_offset_frac=0.08,
    label_y_offset_dec=0.06,
    # band annotations
    show_band_counts=True,            # now only raw N (no percentages)
    band_box_alpha=0.9,
    band_box_pad=0.2,
    band_fontsize=10,
    band_y_offset_frac=0.06,
    band_y_offset_dec=0.04,
    band_series_vstack=0.015,
    band_series_vstack_dec=0.02,
    # line width
    linewidth=2.0,
):
    """
    Plot multiple magnitude histograms normalized PER SURVEY by that survey's
    'All' population within the displayed magnitude range (x_range).
    Subclass curves are divided by N_all_in_range(survey). Band boxes show raw N.
    """
    survey_styles = {} if survey_styles is None else dict(survey_styles)
    survey_labels = {} if survey_labels is None else dict(survey_labels)
    class_colors  = {} if class_colors  is None else dict(class_colors)
    class_labels  = {} if class_labels  is None else dict(class_labels)

    if ranges is not None and range_colors is not None:
        if len(ranges) != len(range_colors):
            raise ValueError("ranges and range_colors must have the same length.")

    fig, ax = plt.subplots(figsize=figsize)

    # Determine x-lims
    if x_range is not None:
        xmin, xmax = x_range
    else:
        all_vals = np.concatenate([np.asarray(s["values"]) for s in series if len(s.get("values", [])) > 0])
        xmin, xmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    # Shaded magnitude bands
    if ranges is not None and range_colors is not None:
        for (low, high), col in zip(ranges, range_colors):
            ax.axvspan(low, high, color=col, alpha=0.3, linewidth=0)

    # ── Per-survey normalization baselines: N_all_in_range[survey]
    surveys = sorted({s.get("survey", "unknown") for s in series})
    N_all_in_range = {}
    for sv in surveys:
        all_series = [s for s in series if s.get("survey") == sv and s.get("cls") == "All"]
        if not all_series:
            raise ValueError(f"Normalization requires an 'All' series for survey '{sv}'.")
        # baseline = count of 'All' within x_range for that survey
        vals = np.asarray(all_series[0]["values"])
        vals = vals[np.isfinite(vals)]
        mask_range = (vals >= xmin) & (vals <= xmax)
        N_all_in_range[sv] = int(np.sum(mask_range))

    # Plot each series (normalized by its survey's baseline)
    plotted = []
    for s in series:
        vals = np.asarray(s["values"])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        this_bins = int(s.get("bins", bins))
        survey = s.get("survey", "unknown")
        cls    = s.get("cls", "unknown")
        color  = class_colors.get(cls, "black")
        ls     = survey_styles.get(survey, "-")
        disp_s = survey_labels.get(survey, survey)
        disp_c = class_labels.get(cls, cls)
        lab    = s.get("label", f"{disp_s} · {disp_c}")

        counts, edges = np.histogram(vals, bins=this_bins, range=(xmin, xmax))
        centers = 0.5 * (edges[:-1] + edges[1:])

        denom = max(N_all_in_range.get(survey, 0), 1)  # PER SURVEY baseline
        norm_counts = counts.astype(float) / float(denom)

        ax.plot(centers, norm_counts, color=color, linestyle=ls, linewidth=linewidth, label=lab)

        if label_on_curve and norm_counts.size:
            imax = int(np.argmax(norm_counts))
            x_peak, y_peak = centers[imax], float(norm_counts[imax])
            if logy:
                y_text = max(y_peak, 1e-16) * (10.0 ** label_y_offset_dec)
            else:
                y_text = y_peak * (1.0 + label_y_offset_frac)
            ax.text(
                x_peak + label_x_offset, y_text, lab,
                color=color, fontsize=label_fontsize, ha='center', va='bottom',
                bbox=dict(facecolor='white', edgecolor=color, linestyle=ls,
                          boxstyle=f'round,pad={band_box_pad}', linewidth=1.2, alpha=band_box_alpha)
            )

        plotted.append({
            "survey": survey, "cls": cls, "label": lab, "color": color, "ls": ls,
            "norm_counts": norm_counts, "counts": counts, "edges": edges, "centers": centers,
            "N_all_range": denom,
        })

    # Band text boxes: only raw N (no %), placed above each curve at the band center
    if show_band_counts and (ranges is not None) and len(plotted) > 0:
        ax.relim(); ax.autoscale_view()
        ylo, yhi = ax.get_ylim()
        ax_span = max(yhi - ylo, 1e-16)

        for s_idx, info in enumerate(plotted):
            counts  = info["counts"]
            ncounts = info["norm_counts"]
            edges   = info["edges"]
            centers = info["centers"]
            color, ls = info["color"], info["ls"]

            if counts.size == 0:
                continue

            for (low, high) in ranges:
                x_mid = 0.5 * (low + high)
                j = int(np.argmin(np.abs(centers - x_mid)))
                y_here = float(ncounts[j])

                # count objects whose bin overlaps the band
                bin_lefts = edges[:-1]; bin_rights = edges[1:]
                left  = max(low, edges[0]); right = min(high, edges[-1])
                overlap = (bin_rights > left) & (bin_lefts < right)
                N_band = int(np.sum(counts[overlap]))

                if logy:
                    y_text = max(y_here, 1e-16) * (10.0 ** (band_y_offset_dec + s_idx * band_series_vstack_dec))
                else:
                    y_text = y_here + (band_y_offset_frac + s_idx * band_series_vstack) * ax_span

                ax.text(
                    x_mid, y_text, f"N={N_band}",
                    ha='center', va='bottom', fontsize=band_fontsize, color='black',
                    bbox=dict(facecolor='white', edgecolor=color, linestyle=ls,
                              boxstyle=f'round,pad={band_box_pad}', linewidth=1.5, alpha=band_box_alpha)
                )

    # Axes
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    if title:
        ax.set_title(title, fontsize=20)

    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)

    if logy:
        ax.set_yscale('log')

    ax.tick_params(axis='both', which='major', labelsize=12)

    # Two legends via proxy artists
    if show_survey_legend and survey_styles:
        handles1, labels1 = [], []
        for survey, ls in survey_styles.items():
            disp = survey_labels.get(survey, survey)
            h = Line2D([0], [0], color='black', linestyle=ls, linewidth=linewidth, label=disp)
            handles1.append(h); labels1.append(disp)
        leg1 = ax.legend(handles=handles1, labels=labels1, title="Survey", ncols=1,
                         loc=survey_legend_loc, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        ax.add_artist(leg1)

    if show_class_legend and class_colors:
        handles2, labels2 = [], []
        for cls, col in class_colors.items():
            disp = class_labels.get(cls, cls)
            h = Line2D([0], [0], color=col, linestyle='-', linewidth=linewidth, label=disp)
            handles2.append(h); labels2.append(disp)
        leg2 = ax.legend(handles=handles2, labels=labels2, title="Class", ncols=5,
                         loc=class_legend_loc, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        ax.add_artist(leg2)

    fig.tight_layout()
    return fig, ax

def plot_multi_histograms_two_legends_new(
    *,
    series,                           # list of dicts: {"values": array, "survey": str, "cls": str, "label": Optional[str], "bins": Optional[int]}
    ranges=None,                      # keep for API compatibility; will omit in call
    range_colors=None,                # keep for API compatibility; will omit in call
    bins=200,
    # style maps
    survey_styles=None,
    survey_labels=None,
    class_colors=None,
    class_labels=None,
    # axes/title
    x_label='DESI Magnitude (R)',
    y_label="Normalized Frequency (fraction of 'All' in range)",
    title=None,
    x_range=None,
    y_range=None,
    figsize=(7, 5),
    # legends
    show_survey_legend=True,
    survey_legend_loc='upper right',
    show_class_legend=True,
    class_legend_loc='upper left',
    legend_fontsize=18,
    # log scaling
    logy=False,
    # single on-curve label per curve
    label_on_curve=True,              # now defaults True
    curve_label_mode="count",         # "count" | "name" | "name+count"
    total_label_fmt="N={N:,}",        # thousands separator
    label_fontsize=18,
    label_x_offset=0.0,
    label_y_offset_frac=0.08,
    label_y_offset_dec=0.06,
    curve_label_vstack=0.02,          # vertical stacking between curves
    # (deprecated) band annotations
    show_band_counts=False,           # disabled by default
    band_box_alpha=0.9,
    band_box_pad=0.2,
    band_fontsize=10,
    band_y_offset_frac=0.06,
    band_y_offset_dec=0.04,
    band_series_vstack=0.015,
    band_series_vstack_dec=0.02,
    # line width
    linewidth=2.0,
):
    """
    Plot multiple magnitude histograms normalized PER SURVEY by that survey's
    'All' population within the displayed magnitude range (x_range).
    Subclass curves are divided by N_all_in_range(survey).
    For each curve, optionally draw a single on-curve label with total N in range.
    """
    survey_styles = {} if survey_styles is None else dict(survey_styles)
    survey_labels = {} if survey_labels is None else dict(survey_labels)
    class_colors  = {} if class_colors  is None else dict(class_colors)
    class_labels  = {} if class_labels  is None else dict(class_labels)

    fig, ax = plt.subplots(figsize=figsize)

    # Determine x-lims
    if x_range is not None:
        xmin, xmax = x_range
    else:
        all_vals = np.concatenate([np.asarray(s["values"]) for s in series if len(s.get("values", [])) > 0])
        xmin, xmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    # (No shaded magnitude bands: pass ranges=None in the call)

    # ── Per-survey normalization baselines: N_all_in_range[survey]
    surveys = sorted({s.get("survey", "unknown") for s in series})
    N_all_in_range = {}
    for sv in surveys:
        all_series = [s for s in series if s.get("survey") == sv and s.get("cls") == "All"]
        if not all_series:
            raise ValueError(f"Normalization requires an 'All' series for survey '{sv}'.")
        vals = np.asarray(all_series[0]["values"])
        vals = vals[np.isfinite(vals)]
        mask_range = (vals >= xmin) & (vals <= xmax)
        N_all_in_range[sv] = int(np.sum(mask_range))

    # Plot each series (normalized by its survey's baseline)
    plotted = []
    for s in series:
        vals = np.asarray(s["values"])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        this_bins = int(s.get("bins", bins))
        survey = s.get("survey", "unknown")
        cls    = s.get("cls", "unknown")
        color  = class_colors.get(cls, "black")
        ls     = survey_styles.get(survey, "-")
        disp_s = survey_labels.get(survey, survey)
        disp_c = class_labels.get(cls, cls)
        lab    = s.get("label", f"{disp_s} · {disp_c}")

        counts, edges = np.histogram(vals, bins=this_bins, range=(xmin, xmax))
        centers = 0.5 * (edges[:-1] + edges[1:])

        denom = max(N_all_in_range.get(survey, 0), 1)
        norm_counts = counts.astype(float) / float(denom)

        ax.plot(centers, norm_counts, color=color, linestyle=ls, linewidth=linewidth, label=lab)

        N_total = int(counts.sum())  # total within displayed range
        plotted.append({
            "survey": survey, "cls": cls, "label": lab, "color": color, "ls": ls,
            "norm_counts": norm_counts, "counts": counts, "edges": edges, "centers": centers,
            "N_all_range": denom, "N_total": N_total,
        })

    # Single on-curve label per curve (e.g., N={N})
    if label_on_curve and len(plotted) > 0:
        ax.relim(); ax.autoscale_view()
        ylo, yhi = ax.get_ylim()
        ax_span = max(yhi - ylo, 1e-16)

        for s_idx, info in enumerate(plotted):
            ncounts = info["norm_counts"]
            centers = info["centers"]
            color, ls = info["color"], info["ls"]
            lab = info["label"]
            N_total = info["N_total"]

            if ncounts.size == 0:
                continue

            imax = int(np.argmax(ncounts))
            x_peak, y_peak = centers[imax], float(ncounts[imax])

            if curve_label_mode == "name":
                text = f"{lab}"
            elif curve_label_mode == "name+count":
                text = f"{lab} ({total_label_fmt.format(N=N_total)})"
            else:  # "count"
                text = total_label_fmt.format(N=N_total)

            if logy:
                y_text = max(y_peak, 1e-16) * (10.0 ** (label_y_offset_dec + s_idx * label_y_offset_dec))
            else:
                y_text = y_peak + (label_y_offset_frac + s_idx * curve_label_vstack) * ax_span

            ax.text(
                x_peak + label_x_offset, y_text, text,
                color='black', fontsize=label_fontsize, ha='center', va='bottom',
                bbox=dict(facecolor='white', edgecolor=color, linestyle=ls,
                          boxstyle=f'round,pad=0.3', linewidth=2.2, alpha=0.9)
            )

    # Axes
    ax.set_xlabel(x_label, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    if title:
        ax.set_title(title, fontsize=20)

    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)

    if logy:
        ax.set_yscale('log')

    ax.tick_params(axis='both', which='major', labelsize=20)

    # Two legends via proxy artists
    if show_survey_legend and survey_styles:
        handles1, labels1 = [], []
        for survey, ls in survey_styles.items():
            disp = survey_labels.get(survey, survey)
            h = Line2D([0], [0], color='black', linestyle=ls, linewidth=linewidth, label=disp)
            handles1.append(h); labels1.append(disp)
        leg1 = ax.legend(handles=handles1, labels=labels1, title="Survey", ncols=1,
                         loc=survey_legend_loc, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        ax.add_artist(leg1)

    if show_class_legend and class_colors:
        handles2, labels2 = [], []
        for cls, col in class_colors.items():
            disp = class_labels.get(cls, cls)
            h = Line2D([0], [0], color=col, linestyle='-', linewidth=linewidth, label=disp)
            handles2.append(h); labels2.append(disp)
        leg2 = ax.legend(handles=handles2, labels=labels2, ncols=1,
                         loc=class_legend_loc, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
        ax.add_artist(leg2)

    fig.tight_layout()
    return fig, ax

def plot_histogram_with_ranges_multiple(
    mag_dict,
    ranges,
    colors,
    bins=200,
    x_label='DESI Magnitude (R)',
    title='Histogram of DESI R Magnitudes (by split and loader)',
    labels_dict=None,               
    class_names=None,               
    pct_decimals=1,                 
    annotate_mode='text',
    legend_fontsize=10              
):
    """
    Plot multiple histograms of magnitudes with color-coded background bands and annotate
    object counts and, if provided, class percentages per magnitude band.

    Returns:
        masks_all: dict[(key_dset,key_loader)][(lower,upper)] -> boolean mask on filtered magnitudes
        stats_all: dict[(key_dset,key_loader)][(lower,upper)] with:
                   'N', 'per_class_counts', 'per_class_pct', 'class_labels'
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    masks_all = {}
    stats_all = {}

    # Background colored regions for magnitude ranges
    for (lower, upper), color in zip(ranges, colors):
        ax.axvspan(lower, upper, color=color, alpha=0.35)

    # Prepare class naming
    if labels_dict is not None:
        all_labels = np.concatenate(
            [np.asarray(labels_dict[k]).ravel() for k in mag_dict.keys() if k in labels_dict]
        )
        unique_classes = np.unique(all_labels)
        n_classes = unique_classes.size
        if class_names is None:
            class_names = [f"c{int(c)}" for c in unique_classes]
        else:
            n_classes = len(class_names)
    else:
        unique_classes = None
        n_classes = 0

    keys_list = list(mag_dict.keys())
    for idx, (key_dset, key_loader) in enumerate(keys_list):
        magnitudes = np.asarray(mag_dict[(key_dset, key_loader)]).ravel()
        valid = np.isfinite(magnitudes)
        magnitudes = magnitudes[valid]

        labels = None
        if labels_dict is not None and (key_dset, key_loader) in labels_dict:
            labels_full = np.asarray(labels_dict[(key_dset, key_loader)]).ravel()
            if labels_full.shape[0] != valid.shape[0]:
                raise ValueError(f"labels length mismatch for {key_dset}|{key_loader}")
            labels = labels_full[valid]

        label_curve = f"{key_dset} | {key_loader}"
        color_index = idx % len(colors)

        if magnitudes.size == 0:
            continue
        x_min, x_max = float(np.nanmin(magnitudes)), float(np.nanmax(magnitudes))
        hist_counts, edges = np.histogram(magnitudes, bins=bins, range=(x_min, x_max))
        total = hist_counts.sum()
        centers = (edges[:-1] + edges[1:]) / 2.0
        line_color = colors[color_index]
        if total > 0:
            ax.plot(centers, hist_counts, label=label_curve, color=line_color, linewidth=2)
        else:
            ax.plot(centers, np.zeros_like(centers), label=label_curve, color=line_color, linewidth=2)

        masks_all[(key_dset, key_loader)] = {}
        stats_all[(key_dset, key_loader)] = {}

        ybase = np.mean(hist_counts) if total > 0 else 1.0
        for (lower, upper) in ranges:
            mask = (magnitudes >= lower) & (magnitudes < upper)
            N = int(mask.sum())
            masks_all[(key_dset, key_loader)][(lower, upper)] = mask

            per_class_counts = None
            per_class_pct = None
            text_lines = [f"N={N}"]

            if labels is not None and N > 0:
                if unique_classes is not None and not np.array_equal(unique_classes, np.arange(n_classes)):
                    inv = {lab: i for i, lab in enumerate(unique_classes)}
                    lab_idx = np.vectorize(inv.get)(labels[mask])
                else:
                    lab_idx = labels[mask].astype(int)

                per_class_counts = np.bincount(lab_idx, minlength=n_classes)
                for i in range(n_classes):
                    text_lines.append(f"{class_names[i]}: {per_class_counts[i]:.{pct_decimals}f}")

            stats_all[(key_dset, key_loader)][(lower, upper)] = {
                "N": N,
                "per_class_counts": per_class_counts,
                "per_class_pct": per_class_pct,
                "class_labels": class_names if labels is not None else None,
            }

            if annotate_mode == 'text':
                # make the position of the text change based on the number of lines to avoid overlap if multiple magnitude histograms are represented
                x_pos = (lower + upper) / 2.0 + 0.4 * (idx % 2)  # slight offset for each histogram
                y_pos = ybase * (idx % 2 + 2)
                ax.text(
                    x_pos,
                    y_pos,
                    "\n".join(text_lines),
                    color=line_color,
                    fontsize=8,
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', edgecolor=line_color, boxstyle='round,pad=0.25', linewidth=1.0)
                )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=legend_fontsize)  # custom fontsize
    ax.set_xlim(min(r[0] for r in ranges), max(r[1] for r in ranges))
    plt.tight_layout()
    return masks_all, stats_all

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_per_class_counts_together(
    stats_magnitudes: dict,
    *,
    figsize=(11, 6),
    yscale: str = "log",             # "log" or "linear"
    class_order: list = None,        # optional enforced order of classes
    legend_outside: bool = True,
    title: str = "Per-class counts vs. magnitude (all entries)",
    title_fontsize: int = 16,        # <-- NEW: title fontsize
    class_legend_kwargs: dict = None,  # <-- NEW: kwargs for class legend (colors)
    entry_legend_kwargs: dict = None,  # <-- NEW: kwargs for entry legend (styles)
    grid_alpha: float = 0.25,
):
    """
    Plot per-class object-count curves for ALL entries in `stats_magnitudes` on the same axes.

    stats_magnitudes[(source, split)] = {
        (low, high): {'N': int, 'per_class_counts': np.ndarray(C,), 'per_class_pct': None, 'class_labels': list[str]},
        ...,
        'plot_kwargs': { 'linestyle': '--', 'marker': 'o', 'markersize': 8.0, 'label': 'Your label' }
    }

    Style principle:
    - Color = class (consistent across all entries)
    - Linestyle/marker = entry (from entry['plot_kwargs'])

    Legend customization:
    - class_legend_kwargs controls the legend for classes (color legend).
    - entry_legend_kwargs controls the legend for entries (style legend).
      Example:
        {
          "loc": "upper left", "bbox_to_anchor": (0.73, 1.0), "fontsize": 9, "ncol": 1,
          "title": "Evaluation Cases", "frameon": True, "fancybox": True, "shadow": True, "borderaxespad": 0.0,
        }
    """
    entries = list(stats_magnitudes.keys())
    if not entries:
        raise ValueError("stats_magnitudes is empty.")

    # --- Collect global bins (sorted unique) ---
    all_bins = set()
    for entry in entries:
        d = stats_magnitudes[entry]
        for k in d.keys():
            if isinstance(k, tuple) and len(k) == 2:
                all_bins.add(k)
    if not all_bins:
        raise ValueError("No magnitude bins found.")
    bins = sorted(all_bins, key=lambda x: (x[0], x[1]))
    nb = len(bins)
    x = np.arange(nb)
    xticklabels = [f"({lo:g}, {hi:g}]" for (lo, hi) in bins]

    # --- Determine class list/order ---
    if class_order is None:
        seen = []
        for entry in entries:
            d = stats_magnitudes[entry]
            for b in bins:
                if b in d:
                    for lab in d[b]["class_labels"]:
                        if lab not in seen:
                            seen.append(lab)
        classes = seen
    else:
        classes = list(class_order)
    C = len(classes)

    # --- Color mapping per class, consistent over entries ---
    cmap = plt.get_cmap("tab10" if C <= 10 else "tab20")
    class_to_color = {cls: cmap(i % cmap.N) for i, cls in enumerate(classes)}

    # --- Precompute counts per entry aligned to global bins & classes ---
    counts_by_entry = {}
    styles_by_entry = {}
    for entry in entries:
        d = stats_magnitudes[entry]
        pk = dict(d.get("plot_kwargs", {}))
        if "linstyle" in pk and "linestyle" not in pk:
            pk["linestyle"] = pk.pop("linstyle")
        pk.setdefault("linestyle", "-")
        pk.setdefault("marker", None)
        pk.setdefault("markersize", 8.0)
        pk.setdefault("label", f"{entry[0]} — {entry[1]}")
        styles_by_entry[entry] = pk

        M = np.zeros((C, nb), dtype=float)
        for j, b in enumerate(bins):
            if b not in d:
                continue
            info = d[b]
            bin_labels = info["class_labels"]
            bin_counts = np.asarray(info["per_class_counts"]).astype(float)
            idx_map = {lab: i for i, lab in enumerate(bin_labels)}
            for i, cls in enumerate(classes):
                if cls in idx_map:
                    M[i, j] = bin_counts[idx_map[cls]]
        counts_by_entry[entry] = M

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # For legends
    class_handles = {}
    entry_handles = []

    # One line per (class, entry): color by class; linestyle/marker by entry
    for i, cls in enumerate(classes):
        color = class_to_color[cls]
        class_handles.setdefault(cls, Line2D([0], [0], color=color, lw=3, label=cls))
        for entry in entries:
            pk = styles_by_entry[entry]
            ax.plot(
                x, counts_by_entry[entry][i],
                color=color,
                linestyle=pk["linestyle"],
                marker=pk["marker"],
                markersize=pk["markersize"],
                linewidth=2.0,
            )

    # Build entry legend handles using neutral color (black), styled by entry
    for entry in entries:
        pk = styles_by_entry[entry]
        entry_handles.append(
            Line2D([0], [0],
                   color="black",
                   linestyle=pk["linestyle"],
                   marker=pk["marker"],
                   markersize=pk["markersize"],
                   linewidth=2.5,
                   label=pk["label"])
        )

    # Axes cosmetics
    ax.set_title(title, fontsize=title_fontsize)  # <-- use customizable title fontsize
    ax.set_xlabel("Magnitude bin (low, high]")
    ax.set_ylabel("Objects")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=20, ha="right")
    if yscale == "log":
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=grid_alpha)

    # Defaults for both legends (can be overridden by *_legend_kwargs)
    default_class_legend = {"title": "Class (color)", "loc": "upper left", "bbox_to_anchor": (1.01, 1.0)}
    default_entry_legend = {"title": "Entry (style)", "loc": "lower left", "bbox_to_anchor": (1.01, 0.0)}

    cls_kwargs = {**default_class_legend, **(class_legend_kwargs or {})}
    ent_kwargs = {**default_entry_legend, **(entry_legend_kwargs or {})}

    leg1 = ax.legend(handles=list(class_handles.values()), **cls_kwargs)
    ax.add_artist(leg1)
    ax.legend(handles=entry_handles, **ent_kwargs)

    if legend_outside:
        plt.tight_layout(rect=[0, 0, 0.8, 1])
    else:
        plt.tight_layout()

    return fig, ax


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
    axes[0].annotate(
        f"Min @ {epoch_min}",
        xy=(epoch_min, val_min),           # point to the minimum (data coords)
        xycoords='data',
        xytext=(0.98, 0.98),               # place text at top-right of the axes
        textcoords='axes fraction',
        ha='right', va='top',
        fontsize=12,
        arrowprops=dict(arrowstyle='->', lw=1.5)
    )

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
    legend = axes[0].legend(custom_lines, custom_labels, loc='upper left', fancybox=True, shadow=True, fontsize=14)
    axes[0].add_artist(legend)

    plt.tight_layout()
    plt.show()

def f1_radar_plot(
    results_dict,
    model_configs,
    datasets,
    dataset_colors,
    model_styles,
    class_names=None,
    fig_title="F1-score Radar Plot by Class",
    show_model_lines=True,
    show_mean_lines=True,
    linewidth_model=1,
    linewidth_mean=3,
    zero_division=0,
    figsize=(8, 8),
    legend_loc_colors=(0.3, 1.05),
    legend_loc_styles=(1.0, 0.05),
    legend_fontsize=16,
    text_fontsize=14,
    yticks=[0.2, 0.4, 0.6, 0.8, 1.0],
    ylim=(0, 1),
    tick_labelsize=18,
    radial_labelsize=14,
    title_fontsize=20,
    title_pad=50.
):

    # Determine number of classes
    sample_model = model_configs[0]['index']
    sample_ds = datasets[0]
    num_classes = len(np.unique(results_dict[sample_model][sample_ds]['true']))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Compute radar angles
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
    angles += angles[:1]

    # Compute F1 scores per model/dataset
    f1_dict = {ds: [] for ds in datasets}
    for ds in datasets:
        for model_cfg in model_configs:
            model_idx = model_cfg['index']
            yy_true = results_dict[model_idx][ds]['true']
            yy_pred = results_dict[model_idx][ds]['label']
            f1 = f1_score(yy_true, yy_pred, average=None, zero_division=zero_division)
            f1_dict[ds].append(f1)

    # Start plotting
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # Track handles for both legends
    legend_handles_colors = []
    legend_handles_styles = []

    # Place macro-F1 boxes at spaced angles around the radar
    text_box_angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False) + np.pi/4
    radius_box = ylim[1] * 1.05

    for i, ds in enumerate(datasets):
        color = dataset_colors[ds]
        dataset_f1s = []

        for model_cfg, f1_vals in zip(model_configs, f1_dict[ds]):
            style = model_styles[model_cfg['style_group']]
            f1_plot = f1_vals.tolist() + [f1_vals[0]]
            dataset_f1s.append(f1_vals)

            if show_model_lines:
                ax.plot(angles, f1_plot, color=color, linestyle=style, linewidth=linewidth_model)

        if show_mean_lines:
            f1_mean = np.mean(np.array(dataset_f1s), axis=0)
            f1_mean_plot = f1_mean.tolist() + [f1_mean[0]]
            ax.plot(angles, f1_mean_plot, color=color, linestyle='-', linewidth=linewidth_mean)

            # Annotate macro F1
            macro_f1 = np.mean(f1_mean)
            ax.text(
                text_box_angles[i], radius_box,
                f"{ds.replace('_', ' ')}\nF1={macro_f1:.2f}",
                color=color,
                fontsize=text_fontsize,
                ha="center", va="center",
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.4', lw=1.5)
            )

    # Axis formatting
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), class_names, fontsize=tick_labelsize)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks], fontsize=radial_labelsize)
    if fig_title is not None:
        ax.set_title(fig_title, fontsize=title_fontsize, pad=title_pad)

    # Legends
    for ds in datasets:
        legend_handles_colors.append(
            Line2D([0], [0], color=dataset_colors[ds], linestyle='-', lw=3, label=ds.replace('_', ' '))
        )

    for key in set(cfg['style_group'] for cfg in model_configs):
        legend_handles_styles.append(
            Line2D([0], [0], color='gray', linestyle=model_styles[key], lw=2, label=key)
        )

    # First: color legend
    legend_colors = ax.legend(
        handles=legend_handles_colors, title="Dataset",
        bbox_to_anchor=legend_loc_colors, fontsize=legend_fontsize-2, title_fontsize=legend_fontsize, fancybox=True, shadow=True,
    )
    ax.add_artist(legend_colors)

    # Second: linestyle legend
    legend_styles = ax.legend(
        handles=legend_handles_styles, title="Model Type",
        bbox_to_anchor=legend_loc_styles, fontsize=legend_fontsize-2, title_fontsize=legend_fontsize, fancybox=True, shadow=True,
    )
    ax.add_artist(legend_styles)

    plt.show()

def plot_f1_radar_by_bin_and_dataset(
    results_dict,
    model_configs,
    datasets,
    dataset_linestyles,
    class_names,
    bin_labels,
    bin_colors,
    title="F1 Radar Plot",
    figsize=(10, 10),
    linewidth_model=0.6,
    linewidth_mean=3,
    zero_division=0,
    yticks=[0.2, 0.4, 0.6, 0.8, 1.0],
    ylim=(0, 1),
    tick_labelsize=18,
    radial_labelsize=14,
    text_fontsize=14,
    legend_fontsize=16,
    title_fontsize=30,
    title_pad=50,
    show_model_lines=True,
    save_path=None,
    dpi=300,
    tight_pad=1.0,
    dataset_labels = {'test_JPAS_matched': 'JPAS', 'val_DESI_only': 'DESI'},
    show=True
):
    num_bins = len(bin_labels)
    num_classes = len(class_names)
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
    angles += angles[:1]
    text_box_angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False) + np.pi / 6
    radius_box = ylim[1] * 1.05

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    legend_handles_colors = []
    legend_handles_styles = []

    for bin_id in range(num_bins):
        color = bin_colors[bin_id]
        label = bin_labels[bin_id]
        f1_dict_bin = {ds: [] for ds in datasets}

        for ds in datasets:
            for model_cfg in model_configs:
                model_idx = model_cfg['index']
                all_data = results_dict[model_idx][ds]
                mag_bins = all_data['MAG_BIN_ID']
                mask = mag_bins == bin_id
                if np.sum(mask) == 0:
                    continue

                yy_true = all_data['true'][mask]
                yy_pred = all_data['label'][mask]
                f1 = f1_score(yy_true, yy_pred, average=None, zero_division=zero_division)
                f1_dict_bin[ds].append(f1)

                if show_model_lines:
                    f1_plot = f1.tolist() + [f1[0]]
                    ax.plot(angles, f1_plot, color=color, linestyle=dataset_linestyles[ds], linewidth=linewidth_model)

        for i_ds, ds in enumerate(datasets):
            if len(f1_dict_bin[ds]) == 0:
                continue

            f1_mean = np.mean(np.stack(f1_dict_bin[ds]), axis=0)
            f1_mean_plot = f1_mean.tolist() + [f1_mean[0]]

            ax.plot(angles, f1_mean_plot, color=color, linestyle=dataset_linestyles[ds], linewidth=linewidth_mean)

            macro_f1 = np.mean(f1_mean)
            label_ds = dataset_labels.get(ds, ds.replace('_', ' ')) if dataset_labels else ds.replace('_', ' ')
            ax.text(
                text_box_angles[bin_id] + 0.3 * i_ds, radius_box,
                f"{label} | {label_ds}\nF1={macro_f1:.2f}",
                color=color,
                fontsize=text_fontsize,
                ha="center", va="center",
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.4', lw=1.5)
            )

        legend_handles_colors.append(
            Line2D([0], [0], color=color, linestyle='-', lw=3, label=label)
        )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    angle_degrees = np.degrees(angles[:-1])
    ax.set_thetagrids(angle_degrees, labels=class_names)
    for label_obj, angle in zip(ax.get_xticklabels(), angle_degrees):
        angle = angle % 360
        if 90 < angle < 270:
            label_obj.set_rotation(angle + 180)
            label_obj.set_verticalalignment('center')
            label_obj.set_horizontalalignment('right')
        else:
            label_obj.set_rotation(angle)
            label_obj.set_verticalalignment('center')
            label_obj.set_horizontalalignment('left')
        label_obj.set_fontsize(tick_labelsize)

    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks], fontsize=radial_labelsize)
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)

    for ds in datasets:
        label = dataset_labels.get(ds, ds.replace('_', ' '))
        legend_handles_styles.append(
            Line2D([0], [0], color='gray', linestyle=dataset_linestyles[ds], lw=2, label=label)
        )

    legend_colors = ax.legend(
        handles=legend_handles_colors,
        title="Magnitude Bin",
        bbox_to_anchor=(0.35, 0.95),
        fontsize=legend_fontsize - 2,
        title_fontsize=legend_fontsize,
        fancybox=True,
        shadow=True,
        ncol=2
    )
    ax.add_artist(legend_colors)

    legend_styles = ax.legend(
        handles=legend_handles_styles,
        title="Dataset Type",
        bbox_to_anchor=(0.85, 1.05),
        fontsize=legend_fontsize - 2,
        title_fontsize=legend_fontsize,
        fancybox=True,
        shadow=True
    )
    ax.add_artist(legend_styles)

    fig.tight_layout(pad=tight_pad)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()

def plot_f1_difference_single_model(
    results_dict,
    model_index,
    ds_source="val_DESI_only",
    ds_target="test_JPAS_matched",
    class_names=None,
    bin_labels=None,
    bin_colors=None,
    bar_width=0.15,
    figsize=(12, 6),
    ylim=(-0.55, 0.55),
    yticks=None,
    title=None,
    title_fontsize=18,
    title_pad=20,
    ylabel="Δ F1-score",
    ylabel_fontsize=16,
    tick_fontsize=14,
    tick_rotation=15,
    legend_fontsize=16,
    grid=True,
    hline=True,
    hline_style='--',
    hline_color='gray',
    hline_width=1,
    zero_division=0,
    tight_layout=True,
    show=True,
    save_path=None,
    dpi=300
):
    data_src = results_dict[model_index][ds_source]
    data_tgt = results_dict[model_index][ds_target]

    num_bins = np.max(data_src["MAG_BIN_ID"]) + 1
    num_classes = len(np.unique(data_src["true"])) if class_names is None else len(class_names)
    class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    bin_labels = bin_labels or [f"Bin {i}" for i in range(num_bins)]
    bin_colors = bin_colors or plt.cm.viridis(np.linspace(0, 1, num_bins))
    yticks = yticks or np.linspace(ylim[0], ylim[1], 5)

    delta_f1_per_bin = []
    for bin_id in range(num_bins):
        mask_src = data_src['MAG_BIN_ID'] == bin_id
        mask_tgt = data_tgt['MAG_BIN_ID'] == bin_id

        if np.sum(mask_src) == 0 or np.sum(mask_tgt) == 0:
            f1_diff = np.full(num_classes, np.nan)
        else:
            f1_src = f1_score(data_src['true'][mask_src], data_src['label'][mask_src], average=None, zero_division=zero_division)
            f1_tgt = f1_score(data_tgt['true'][mask_tgt], data_tgt['label'][mask_tgt], average=None, zero_division=zero_division)
            f1_diff = f1_tgt - f1_src

        delta_f1_per_bin.append(f1_diff)

    delta_f1_per_bin = np.array(delta_f1_per_bin)
    x = np.arange(num_classes)

    fig, ax = plt.subplots(figsize=figsize)
    for i, delta in enumerate(delta_f1_per_bin):
        offset = (i - num_bins // 2) * bar_width + (bar_width / 2 if num_bins % 2 == 0 else 0)
        ax.bar(x + offset, delta, width=bar_width, color=bin_colors[i], label=bin_labels[i], edgecolor='black')

    if hline:
        ax.axhline(0, color=hline_color, linestyle=hline_style, linewidth=hline_width)
    if grid:
        ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=tick_fontsize, rotation=tick_rotation, ha='right')
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)

    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=title_pad)

    ax.legend(title="Magnitude Bin", fontsize=legend_fontsize - 2, title_fontsize=legend_fontsize)

    if tight_layout:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    if show:
        plt.show()
