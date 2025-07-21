import os
import logging
import random
import numpy as np
import torch
from sklearn.neighbors import BallTree

def set_N_threads_(N_threads=1):
    logging.info(f'N_threads: {N_threads}')
    os.environ["OMP_NUM_THREADS"] = str(N_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(N_threads)
    os.environ["MKL_NUM_THREADS"] = str(N_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(N_threads)
    return N_threads

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using GPU
    torch.cuda.manual_seed_all(seed)  # If multiple GPUs

def count_neighbors_within_radius(JPAS_RADEC, OTHER_RADEC, radius_deg):
    """
    Count number of points in OTHER_RADEC within radius_deg of each JPAS_RADEC point.

    Parameters
    ----------
    JPAS_RADEC : ndarray of shape (N, 2)
        Positions in RA, DEC (degrees).
    OTHER_RADEC : ndarray of shape (M, 2)
        Positions in RA, DEC (degrees) to count neighbors from.
    radius_deg : float
        Search radius in degrees.

    Returns
    -------
    counts : ndarray of shape (N,)
        Number of neighbors in OTHER_RADEC within radius_deg of each JPAS point.
    """
    # Convert degrees to radians
    JPAS_rad = np.radians(JPAS_RADEC)
    OTHER_rad = np.radians(OTHER_RADEC)
    radius_rad = np.radians(radius_deg)

    # Build BallTree with haversine distance
    tree = BallTree(OTHER_rad, metric='haversine')

    # Query the number of neighbors within radius
    # BallTree radius is in radians on the unit sphere
    indices = tree.query_radius(JPAS_rad, r=radius_rad)

    # Count neighbors for each JPAS point
    counts = np.array([len(idxs) for idxs in indices])

    return counts


def compute_kd_tree_number_of_neighbors_as_function_of_radius_in_terms_of_mean_nearest_neighbor_distance(POS, min_R_factor=4, max_R_factor=32, NN=7):
    """
    Computes the number of neighbors within a radius range in terms of the mean nearest-neighbor distance.
    The function builds a BallTree for efficient distance queries, and then it calculates the number of
    neighbors within different radius values based on the mean nearest-neighbor distance.

    Parameters:
    POS (np.ndarray): Array of point positions (shape: [n_points, 2]).
    min_R_factor (int): Minimum factor for radius scaling based on mean nearest neighbor distance.
    max_R_factor (int): Maximum factor for radius scaling.
    NN (int): Number of radius bins to calculate.

    Returns:
    tuple: (R_arcmin, kd_counts) - radii in arcminutes and corresponding neighbor counts.
    """

    # Input validation
    assert isinstance(POS, np.ndarray), "POS must be a numpy array"
    assert POS.ndim == 2 and POS.shape[1] == 2, "POS should have shape [n_points, 2] for (RA, DEC)"
    assert min_R_factor < max_R_factor, "min_R_factor should be less than max_R_factor"
    assert NN > 0, "NN should be a positive integer"

    # Convert positions to radians for the haversine distance calculation
    logging.debug("Converting positions to radians for haversine metric.")
    tmp_pos_rad = np.radians(POS)

    # Build the BallTree for efficient nearest neighbor search
    logging.debug("Building BallTree for distance queries.")
    tree = BallTree(tmp_pos_rad, metric='haversine')

    # Query distances to the nearest neighbor, excluding the point itself (k=2)
    logging.debug("Querying nearest neighbors using BallTree.")
    distances, _ = tree.query(tmp_pos_rad, k=2)

    # Extract the distance to the nearest neighbor (ignoring the point itself)
    nearest_neighbor_radians = distances[:, 1]
    nearest_neighbor_degrees = np.degrees(nearest_neighbor_radians)

    # Compute the mean nearest-neighbor distance in degrees and arcminutes
    mean_nn_deg = nearest_neighbor_degrees.mean()
    mean_nn_arcmin = mean_nn_deg * 60
    logging.info(f"Mean nearest-neighbor distance: {mean_nn_deg:.4f} deg ({mean_nn_arcmin:.2f} arcmin)")

    # Calculate the radii in arcminutes based on the mean nearest-neighbor distance
    R_arcmin = np.linspace(min_R_factor * mean_nn_arcmin, max_R_factor * mean_nn_arcmin, NN)
    kd_counts = []

    # Loop over the radii and calculate the number of neighbors within each radius
    for ii, tmp_R_arcmin in enumerate(R_arcmin):
        logging.info(f"Calculating number of neighbors within radius {tmp_R_arcmin:.2f} arcmin (bin {ii + 1}/{NN})")
        
        # Ensure count_neighbors_within_radius is defined elsewhere in your code
        tmp_kd_counts = count_neighbors_within_radius(POS, POS, tmp_R_arcmin / 60.0)
        
        # Normalize the neighbor count by the volume of a spherical shell (in arcminutes)
        normalized_kd_count = tmp_kd_counts / (4 * np.pi * tmp_R_arcmin**2)
        kd_counts.append(normalized_kd_count)

    # Convert the neighbor counts list to a numpy array
    kd_counts = np.array(kd_counts)

    logging.info(f"Completed kd-tree neighbor count calculation for {NN} radius bins.")

    return R_arcmin, kd_counts