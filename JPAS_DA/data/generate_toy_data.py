import numpy as np

# === Function to generate shared centers and covariances ===
def generate_centers_covs(n_classes, n_features, center_range, cov_range, seed):
    np.random.seed(seed)
    centers, covs = [], []
    for _ in range(n_classes):
        center = np.array([np.random.uniform(low, high) for (low, high) in center_range])
        A = np.random.randn(n_features, n_features)
        cov = np.dot(A, A.T) + np.eye(n_features) * np.random.uniform(*cov_range)
        centers.append(center)
        covs.append(cov)
    centers, covs = np.array(centers), np.array(covs)
    return centers, covs

# === Function to generate a dataset from given centers and covariances ===
def generate_dataset_from_structure(n_samples, centers, covs, class_proportions, seed, banana=False):
    np.random.seed(seed)
    n_classes = len(centers)
    n_features = centers[0].shape[0]

    samples_per_class = (class_proportions * n_samples).astype(int)
    samples_per_class[-1] += n_samples - samples_per_class.sum()

    X_list, y_list = [], []
    for class_idx in range(n_classes):
        n_class = samples_per_class[class_idx]
        center = centers[class_idx]
        cov = covs[class_idx]

        samples = np.random.multivariate_normal(center, cov, size=n_class)

        if banana and n_features == 2:
            # Apply curvature around center[0] and center[1]
            x_centered = samples[:, 0] - center[0]
            curvature = np.random.uniform(-0.3, 0.3)
            vertical_shift = curvature * (x_centered ** 2)
            noise = np.random.normal(scale=0.1, size=n_class)
            samples[:, 1] += vertical_shift + noise
            # The curvature preserves x_center and roughly preserves y_center

        labels = np.full(n_class, class_idx)
        X_list.append(samples)
        y_list.append(labels)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    return {"OBS": X}, {"SPECTYPE_int": y}, samples_per_class

# === Function to shift centers and covariances for test set ===
def shift_centers_covs(centers, covs, center_shift, cov_shift, seed):
    np.random.seed(seed)
    shifted_centers, shifted_covs = [], []
    for center, cov in zip(centers, covs):
        shift = np.sign(np.random.randn(*center.shape)) * center_shift
        shifted_centers.append(center + shift)
        shifted_covs.append(cov + np.eye(cov.shape[0]) * cov_shift)
    shifted_centers, shifted_covs = np.array(shifted_centers), np.array(shifted_covs)
    return shifted_centers, shifted_covs