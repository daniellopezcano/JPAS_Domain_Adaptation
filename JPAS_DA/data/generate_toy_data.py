import numpy as np

# =========================
# Spec builders (convenience)
# =========================
def spec_gaussian(center, cov=None, sigma=None, angle=0.0):
    """
    center: (2,) array-like
    cov: (2,2) covariance (if given, overrides sigma/angle)
    sigma: (sx, sy) axis-aligned std devs if cov is None
    angle: rotation (radians) if cov is None
    """
    return {"type": "gaussian", "center": np.asarray(center, float),
            "cov": None if cov is None else np.asarray(cov, float),
            "sigma": None if sigma is None else np.asarray(sigma, float),
            "angle": float(angle)}

def spec_ring(center, radius, width, arc=None, jitter=0.0):
    """
    radius: mean radius
    width: radial std
    arc: (theta0, theta1) in radians; None => full annulus
    jitter: isotropic gaussian jitter (std)
    """
    return {"type": "ring", "center": np.asarray(center, float),
            "radius": float(radius), "width": float(width),
            "arc": None if arc is None else tuple(map(float, arc)),
            "jitter": float(jitter)}

def spec_spiral(center, a=0.0, b=1.0, turns=2.0, theta0=0.0, radial_noise=0.1, jitter=0.05):
    """
    r(t) = a + b t, theta(t) = theta0 + 2π * turns * t, t∈[0,1]
    """
    return {"type": "spiral", "center": np.asarray(center, float),
            "a": float(a), "b": float(b), "turns": float(turns), "theta0": float(theta0),
            "radial_noise": float(radial_noise), "jitter": float(jitter)}

def spec_spline(control_points, thickness=0.15, jitter=0.02, closed=False):
    """
    Piecewise-linear curve thickened by Gaussian noise orthogonal to the path.
    control_points: (K,2) array
    thickness: std dev perpendicular to the path
    jitter: small isotropic jitter
    closed: if True, also connect last->first
    """
    return {"type": "spline", "control_points": np.asarray(control_points, float),
            "thickness": float(thickness), "jitter": float(jitter), "closed": bool(closed)}

def spec_mixture(components, weights=None):
    """
    components: list of spec dicts (e.g., [spec_ring(...), spec_gaussian(...), ...])
    weights: list of positive weights; if None, equal weights
    """
    return {"type": "mixture", "components": components,
            "weights": None if weights is None else np.asarray(weights, float)}

# =========================
# Samplers for each spec type
# =========================
def _rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],[s, c]], float)

def _sample_gaussian(spec, n, rng):
    c = spec["center"]
    if spec["cov"] is not None:
        cov = np.asarray(spec["cov"], float)
    else:
        sx, sy = spec["sigma"]
        R = _rot2d(spec["angle"])
        cov = R @ np.diag([sx**2, sy**2]) @ R.T
    return rng.multivariate_normal(c, cov, size=n)

def _sample_ring(spec, n, rng):
    c = spec["center"]
    radius, width = spec["radius"], spec["width"]
    jitter = spec["jitter"]
    if spec["arc"] is None:
        theta = rng.uniform(0, 2*np.pi, size=n)
    else:
        t0, t1 = spec["arc"]
        theta = rng.uniform(t0, t1, size=n)
    r = radius + rng.normal(0, width, size=n)
    X = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    if jitter > 0:
        X += rng.normal(0, jitter, size=X.shape)
    return X + c

def _sample_spiral(spec, n, rng):
    c = spec["center"]
    t = rng.uniform(0, 1, size=n)
    theta = spec["theta0"] + 2*np.pi*spec["turns"] * t
    r = spec["a"] + spec["b"]*t + rng.normal(0, spec["radial_noise"], size=n)
    X = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    if spec["jitter"] > 0:
        X += rng.normal(0, spec["jitter"], size=X.shape)
    return X + c

def _sample_spline(spec, n, rng):
    P = np.asarray(spec["control_points"], float)
    closed = bool(spec["closed"])
    if closed:
        segs = np.vstack([P, P[0]])
    else:
        segs = P
    # Build segment lengths
    V = segs[1:] - segs[:-1]
    L = np.linalg.norm(V, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    total = cum[-1]
    if total == 0:
        # degenerate, fallback to small Gaussian around the point
        return P[0] + rng.normal(0, spec["thickness"], size=(n,2))
    # choose segments by length
    probs = L / total
    idx = rng.choice(len(L), size=n, p=probs)
    u = rng.random(n)
    base = segs[idx] + (V[idx] * u[:, None])
    # perpendicular “thickness” displacement
    T = V[idx]
    # handle zero-length segments robustly
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    norms = np.where(norms==0, 1.0, norms)
    T_unit = T / norms
    perp = np.column_stack([-T_unit[:,1], T_unit[:,0]])
    off = perp * rng.normal(0, spec["thickness"], size=(n,1))
    X = base + off
    if spec["jitter"] > 0:
        X += rng.normal(0, spec["jitter"], size=X.shape)
    return X

def _sample_mixture(spec, n, rng):
    comps = spec["components"]
    if spec["weights"] is None:
        w = np.ones(len(comps), float) / len(comps)
    else:
        w = np.asarray(spec["weights"], float)
        w = w / w.sum()
    # allocate counts
    counts = np.floor(w * n).astype(int)
    counts[-1] += n - counts.sum()
    parts = []
    for c_spec, k in zip(comps, counts):
        if k <= 0: 
            continue
        parts.append(_sample_any(c_spec, k, rng))
    return np.vstack(parts) if parts else np.zeros((0,2), float)

def _sample_any(spec, n, rng):
    t = spec["type"]
    if t == "gaussian": return _sample_gaussian(spec, n, rng)
    if t == "ring":     return _sample_ring(spec, n, rng)
    if t == "spiral":   return _sample_spiral(spec, n, rng)
    if t == "spline":   return _sample_spline(spec, n, rng)
    if t == "mixture":  return _sample_mixture(spec, n, rng)
    raise ValueError(f"Unknown spec type: {t}")

# =========================
# Public API
# =========================
def generate_dataset_from_specs(n_samples, class_specs, class_proportions, seed):
    """
    class_specs: list of spec dicts (length = n_classes)
    class_proportions: array summing to 1.0
    """
    rng = np.random.default_rng(seed)
    n_classes = len(class_specs)
    class_proportions = np.asarray(class_proportions, float)
    assert np.isclose(class_proportions.sum(), 1.0), "class_proportions must sum to 1"

    samples_per_class = (class_proportions * n_samples).astype(int)
    samples_per_class[-1] += n_samples - samples_per_class.sum()

    X_list, y_list = [], []
    for ci, spec in enumerate(class_specs):
        k = samples_per_class[ci]
        if k <= 0:
            continue
        Xc = _sample_any(spec, k, rng)
        y_list.append(np.full(k, ci, dtype=np.int64))
        X_list.append(Xc.astype(np.float32))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return {"OBS": X}, {"SPECTYPE_int": y}, samples_per_class
