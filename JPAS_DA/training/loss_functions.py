import os
import torch
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import torch
import torch.nn as nn

def cross_entropy(
    yy_true: torch.Tensor,
    yy_pred: torch.Tensor,
    class_weights: torch.Tensor = None,
    *,
    parameters=None,            # e.g., model.parameters()
    l2_lambda: float = 0.0,     # weight for L2 (ridge) regularization
    l1_lambda: float = 0.0,     # weight for L1 (lasso) regularization
    exclude_bias: bool = True   # skip 1D params (bias/BN) in regularization
) -> torch.Tensor:
    """
    Cross-entropy with optional L1/L2 regularization on model parameters.

    Args:
        yy_true: Long tensor of shape (N,) with class indices.
        yy_pred: Float tensor of shape (N, C) with **logits** (not softmax).
        class_weights: Optional tensor of shape (C,) with per-class weights.
        parameters: Iterable of tensors to regularize (e.g., model.parameters()).
        l2_lambda: Coefficient for L2 penalty (sum of squares).
        l1_lambda: Coefficient for L1 penalty (sum of absolute values).
        exclude_bias: If True, skip parameters with ndim == 1.

    Returns:
        Scalar loss tensor.
    """
    if class_weights is not None:
        class_weights = class_weights.to(yy_pred.device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fn(yy_pred, yy_true)

    # Add regularization if requested and parameters are provided
    if (l1_lambda > 0.0 or l2_lambda > 0.0) and parameters is not None:
        # Accept either a model or an iterable of tensors
        if hasattr(parameters, "parameters"):  # a nn.Module
            params_iter = parameters.parameters()
        else:
            params_iter = parameters

        reg = torch.zeros((), device=yy_pred.device, dtype=loss.dtype)
        for p in params_iter:
            if p is None or (not p.requires_grad):
                continue
            if exclude_bias and p.ndim == 1:
                continue
            if l2_lambda > 0.0:
                reg = reg + l2_lambda * torch.sum(p.pow(2))
            if l1_lambda > 0.0:
                reg = reg + l1_lambda * torch.sum(p.abs())
        loss = loss + reg

    return loss / yy_true.size(0)

def weinberger_loss(yy, delta_pull=0.5, delta_push=1.5, c_pull=1., c_push=1., c_reg=0.001, _epsilon=1e-8, save_plots_path=None):
    """
    Compute the Weinberger loss (described in https://arxiv.org/pdf/1708.02551) for a batch of class-separated embeddings.

    This loss function pulls samples from the same class closer together (within
    a margin `delta_pull`) and pushes different class centers further apart (beyond
    a margin `delta_push`). It is designed for supervised contrastive learning
    settings where latent vectors are grouped by class.

    Parameters
    ----------
    yy : torch.Tensor
        Tensor of shape (batch_size, n_classes, latent_dim=2).
        Each entry `yy[b, c, :]` is a latent representation of class `c` in batch element `b`.

    delta_pull : float, optional
        Margin inside which class embeddings are pulled toward their respective cluster centers.

    delta_push : float, optional
        Margin outside which cluster centers of different classes are pushed apart.

    c_pull : float, optional
        Weight of the intra-class pull loss term.

    c_push : float, optional
        Weight of the inter-class push loss term.

    c_reg : float, optional
        Weight of the regularization term (penalizes distance of cluster centers from the origin).

    _epsilon : float, optional
        Small constant to prevent numerical instability during square root or division operations.

    save_plots_path : str or None, optional
        If provided, saves a visualization of the latent space at the given path.

    Returns
    -------
    tuple
        - loss : torch.Tensor
            Total Weinberger loss.
        - L_pull : torch.Tensor
            Intra-class compactness loss.
        - L_push : torch.Tensor
            Inter-class separation loss.
        - L_reg : torch.Tensor
            Regularization term (L2 norm of cluster centers).
        - cluster_centers : torch.Tensor
            Tensor of shape (n_classes, latent_dim) representing class centers.

    Notes
    -----
    This function assumes that each class appears exactly once in each batch sample,
    i.e., the latent vectors are grouped per class (not per sample).
    That is: `yy.shape == (batch_size_per_class, n_classes, latent_dim)`
    """
    # Compute positions cluster centers
    cluster_centers = torch.sum(yy, axis=0) / yy.shape[0]
    logging.debug('Cluster centers: %s', cluster_centers)

    # Compute L_reg 
    cluster_centers_distance_origin = torch.sqrt(torch.sum(cluster_centers**2, axis=-1)+_epsilon)
    L_reg = torch.sum(cluster_centers_distance_origin) / yy.shape[0]

    # Compute L_pull 
    cluster_centers_expanded = cluster_centers[None,:].repeat(yy.shape[0], 1, 1)
    distances_to_cluster_centers = torch.sqrt(torch.sum((yy - cluster_centers_expanded)**2, axis=-1)+_epsilon)
    hinged_distances_to_cluster_centers = distances_to_cluster_centers - delta_pull
    tmp_max_hinged =torch.max(torch.FloatTensor([0., torch.max(hinged_distances_to_cluster_centers)]))
    ind_L_pull_terms = torch.clip(hinged_distances_to_cluster_centers, 0., tmp_max_hinged)**2
    L_pull = torch.sum(ind_L_pull_terms)

    # Compute L_push
    casted_cluster_centers = cluster_centers.repeat(cluster_centers.shape[0], 1, 1)
    relative_posistions_cluster_centers = casted_cluster_centers - torch.transpose(casted_cluster_centers, 0, 1)
    distances_between_cluster_centers = torch.sqrt(torch.sum(relative_posistions_cluster_centers**2, axis=-1)+_epsilon)
    hinged_distances_between_cluster_centers = 2*delta_push - distances_between_cluster_centers
    ind_L_push_terms = torch.triu(torch.clip(hinged_distances_between_cluster_centers, 0., 2.*delta_push)**2, diagonal=1)
    L_push = torch.sum(ind_L_push_terms)

    # total loss
    loss = c_pull*L_pull + c_push*L_push + c_reg*L_reg
    logging.debug('Total Weinberger loss: %s', loss)

    if save_plots_path is not None:
        logging.info(f"📸 Saving Weinberger cluster plot to {save_plots_path}")
        plot_weinberger_clusters(
            yy=yy,
            cluster_centers=cluster_centers,
            delta_pull=delta_pull,
            delta_push=delta_push,
            save_path=save_plots_path
        )

    return loss, L_pull, L_push, L_reg, cluster_centers

def plot_weinberger_clusters(yy, cluster_centers, delta_pull, delta_push, save_path):
    """
    Create and save a 2D visualization of clustered points, pull/push distances, 
    and inter-class force indicators for Weinberger contrastive loss.

    Parameters
    ----------
    yy : torch.Tensor
        Tensor of shape (batch_size, n_classes, 2), representing class-separated latent vectors.
    cluster_centers : torch.Tensor
        Tensor of shape (n_classes, 2), the cluster center of each class.
    delta_pull : float
        Radius of the pull-margin for intra-class samples.
    delta_push : float
        Radius of the push-margin between different cluster centers.
    save_path : str
        Path to save the resulting figure.

    Example
    -------
    >>> import torch
    >>> batch_size = 32
    >>> n_classes = 5
    >>> latent_dim = 2
    >>> centers = torch.tensor([
    ...     [-1.7, -1.8],
    ...     [1.5, -1.],
    ...     [-1.5, 1.5],
    ...     [1.5, 1.5],
    ...     [0.0, 0.0]
    ... ])
    >>> yy = torch.stack([
    ...     centers[c] + 0.5 * torch.randn(batch_size, latent_dim)
    ...     for c in range(n_classes)
    ... ], dim=1)  # shape: (batch_size, n_classes, 2)
    >>> cluster_centers = torch.sum(yy, axis=0) / yy.shape[0]
    >>> plot_weinberger_clusters(
    ...     yy, cluster_centers, delta_pull=0.5, delta_push=1.5,
    ...     save_path="./weinberger_example.png"
    ... )
    """

    logging.debug("🎨 Initializing color map and figure layout...")
    colors = plt.cm.get_cmap("rainbow", yy.shape[1])
    fig, ax = plt.subplots(figsize=(7, 7))

    custom_lines = []
    custom_labels = []

    logging.debug("📌 Plotting points and legends for each class...")
    for cls in range(yy.shape[1]):
        points = yy[:, cls, :].cpu().detach().numpy()
        ax.scatter(points[:, 0], points[:, 1], s=40, alpha=0.5, color=colors(cls), label=f"Class {cls}")
        custom_lines.append(mpl.lines.Line2D([0], [0], color=colors(cls), ls='-', lw=4))
        custom_labels.append(f"Class {cls}")

    logging.debug("📍 Drawing cluster centers and pull/push circles...")
    centers = cluster_centers.cpu().detach().numpy()
    for cls in range(len(centers)):
        center = centers[cls]
        ax.scatter(center[0], center[1], marker="X", s=120, color=colors(cls), edgecolor='white', linewidth=1.5)
        pull_circle = patches.Circle(center, delta_pull, edgecolor=colors(cls), linestyle='-', facecolor='none', linewidth=2)
        push_circle = patches.Circle(center, delta_push, edgecolor=colors(cls), linestyle='--', facecolor='none', linewidth=2)
        ax.add_patch(pull_circle)
        ax.add_patch(push_circle)

        # Pull arrows toward the center
        for angle in [0, 90, 180, 270]:
            dx = np.cos(np.radians(angle))
            dy = np.sin(np.radians(angle))
            start_x = center[0] + dx * (2 * delta_pull + 0.15)
            start_y = center[1] + dy * (2 * delta_pull + 0.15)

            ax.add_patch(
                patches.FancyArrow(
                    start_x, start_y,
                    -dx*delta_pull, -dy*delta_pull,
                    width=0.02, head_width=0.15, head_length=0.15,
                    color=colors(cls), alpha=0.8
                )
            )

    logging.debug("↔️ Drawing push arrows between overlapping push circles...")
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            ci = centers[i]
            cj = centers[j]
            dist = np.linalg.norm(ci - cj)
            if dist < 2 * delta_push:
                direction = (cj - ci) / dist
                overlap_length = 2 * delta_push - dist
                mid = (ci + cj) / 2
                start = mid - 0.5 * direction * overlap_length
                end = mid + 0.5 * direction * overlap_length
                ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="<->", lw=2, color="k", alpha=0.9))

    logging.debug("🧭 Finalizing layout and legends...")
    ax.set_xlabel("Pseudospace x Position")
    ax.set_ylabel("Pseudospace y Position")
    ax.set_aspect("equal")

    legend = ax.legend(custom_lines, custom_labels, loc='lower right', fancybox=True, shadow=True, fontsize=10)
    ax.add_artist(legend)

    custom_lines = [
        patches.Patch(facecolor='none', edgecolor='black', linestyle='-', linewidth=1.5),
        patches.Patch(facecolor='none', edgecolor='black', linestyle='--', linewidth=1.5),
        plt.Line2D([0], [0], marker='X', color='black', linestyle='None', markersize=10, markeredgecolor='white')
    ]
    custom_labels = ["Pull distance", "Push distance", "Cluster center"]
    legend = ax.legend(custom_lines, custom_labels, loc='upper left', fancybox=True, shadow=True, fontsize=10)
    ax.add_artist(legend)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    logging.debug(f"✅ Cluster plot saved to {save_path}")
