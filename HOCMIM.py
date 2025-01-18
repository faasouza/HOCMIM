import numpy as np
from sklearn.metrics import mutual_info_score


def mutual_info(x, y):
    """
    Compute mutual information between two variables.

    Parameters:
    x (np.ndarray): Feature vector.
    y (np.ndarray): Target vector.

    Returns:
    float: Mutual information.
    """
    return mutual_info_score(x, y)


def conditional_mutual_info(x, y, z):
    """
    Compute conditional mutual information I(x; y | z).

    Parameters:
    x (np.ndarray): Feature vector.
    y (np.ndarray): Target vector.
    z (np.ndarray): Conditioning variable.

    Returns:
    float: Conditional mutual information.
    """
    xyz = np.column_stack((x, z))
    yz = np.column_stack((y, z))

    mi_xyz = mutual_info(xyz.ravel(), y)
    mi_z = mutual_info(z.ravel(), y)

    return mi_xyz - mi_z


def n_order_total_redundancy_max(xk, s, y, order):
    """
    Compute n-order total redundancy maximum.

    Parameters:
    xk (np.ndarray): Candidate feature.
    s (np.ndarray): Selected features.
    y (np.ndarray): Target vector.
    order (int): Maximum redundancy order.

    Returns:
    tuple: Maximum redundancy and iteration order.
    """
    ns = s.shape[1] if s.ndim > 1 else 0
    max_rn = -np.inf

    selected_indices = []
    rn_max = mutual_info(xk, y)

    for i in range(min(ns, order)):
        max_rn = -np.inf
        for j in range(ns):
            if j not in selected_indices:
                mi = mutual_info(xk, np.hstack([s[:, selected_indices], s[:, j:j+1]]))
                cmi = conditional_mutual_info(xk, s[:, j], y)
                if mi - cmi > max_rn:
                    max_rn = mi - cmi
                    best_index = j

        selected_indices.append(best_index)
        if max_rn / rn_max < 0.05:
            break

    return max_rn, len(selected_indices)


def hocmim(x, y, k, n, verbose=False):
    """
    High-order Conditional Mutual Information Maximization (HOCMIM) for feature selection.

    Parameters:
    x (np.ndarray): Training data (samples x features).
    y (np.ndarray): Target labels.
    k (int): Number of features to select.
    n (int): Order of HOCMIM approximation.
    verbose (bool): Print progress if True.

    Returns:
    tuple: Selected features, elapsed time, CMI approximations, and selection order.
    """
    import time

    start_time = time.time()
    n_features = x.shape[1]
    k = min(k, n_features)

    # Mutual information for all features
    mi = np.array([mutual_info(x[:, i], y) for i in range(n_features)])

    # Initialize selected features
    selected = []
    cmi_app = []
    order = []

    for t in range(k):
        max_cmi = -np.inf
        best_feature = None

        for i in range(n_features):
            if i not in selected:
                redundancy, it_order = n_order_total_redundancy_max(
                    x[:, i], x[:, selected], y, n
                )
                cmi = mi[i] - redundancy
                if cmi > max_cmi:
                    max_cmi = cmi
                    best_feature = i

        if best_feature is not None:
            selected.append(best_feature)
            cmi_app.append(max_cmi)
            order.append(it_order)

        if verbose:
            print(f"Step {t+1}/{k}: Selected feature {best_feature}")

    elapsed_time = time.time() - start_time
    return selected, elapsed_time, cmi_app, order
