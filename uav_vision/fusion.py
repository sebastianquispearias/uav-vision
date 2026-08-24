"""
Multi-view fusion algorithms for bearing-only target geolocation.

Adapted from Lampesberger's algorithms (sar-geotag-prototype) to work
in local cartesian coordinates (x, y, z in meters) instead of ECEF.

A measurement is a tuple (origin, direction) where:
    - origin: (x, y, z) position of the camera in meters
    - direction: (dx, dy, dz) unit vector pointing toward the target

All functions are pure. No simulator or protocol dependencies.

References:
    - Traa [21]: Loss function and gradient for orthogonal distance minimisation
    - Lampesberger (2024): Ray intersection and gradient descent
    - Dogançay (2005): Weighted least squares for bearing-only localisation
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Type alias: a measurement is (position, direction_vector)
Measurement = Tuple[Tuple[float, float, float], Tuple[float, float, float]]

# Gradient descent defaults (from Lampesberger / Traa)
DEFAULT_LR: float = 0.065
DEFAULT_MAX_ITER: int = 1000
DEFAULT_TOL: float = 1e-7

# Parallel-ray detection threshold
_PARALLEL_EPS: float = 1e-10


def orthogonal_distance(
    point: Tuple[float, float, float],
    ray_origin: Tuple[float, float, float],
    ray_direction: Tuple[float, float, float],
) -> float:
    """Perpendicular distance from a point to a ray in 3D.

    The ray direction is normalised internally if needed.

    Formula:
        d = ||(p - q) - ((p - q) . v) * v||
    where p = ray_origin, v = ray_direction (unit), q = point.

    This is the square root of Traa's d(p, v; q) = (p-q)^T (I - vv^T) (p-q).

    Args:
        point: Target point (x, y, z) in meters.
        ray_origin: Ray start position (x, y, z) in meters.
        ray_direction: Ray direction vector (will be normalised).

    Returns:
        Perpendicular distance in meters (always >= 0).
    """
    q = np.asarray(point, dtype=np.float64)
    p = np.asarray(ray_origin, dtype=np.float64)
    v = np.asarray(ray_direction, dtype=np.float64)

    norm_v = np.linalg.norm(v)
    if norm_v < _PARALLEL_EPS:
        return float(np.linalg.norm(q - p))
    v = v / norm_v

    diff = p - q
    return float(np.linalg.norm(diff - np.dot(diff, v) * v))


def ray_intersection_2(
    m1: Measurement,
    m2: Measurement,
) -> Tuple[float, float, float]:
    """Find the point of closest approach between two rays in 3D.

    Uses the cross-product method from Lampesberger (2024, Section 4.2):
        u = v1 x v2   (common perpendicular direction)
        Solve [v1 | u | -v2] @ [alpha, beta, gamma] = p2 - p1
        q = p1 + alpha*v1 + (beta/2)*u   (midpoint of closest approach)

    Forward-ray clamping: alpha and gamma are clamped to >= 0 so the
    intersection point lies in the forward direction of both rays.

    Degenerate case (parallel rays): when ||v1 x v2|| < eps, the rays
    are (nearly) parallel and the system is singular. In this case the
    function returns the midpoint of the two ray origins, which is a
    stable fallback — the caller should check ray geometry before
    relying on the result.

    Args:
        m1: First measurement (origin, direction).
        m2: Second measurement (origin, direction).

    Returns:
        Estimated target position (x, y, z) in meters.
    """
    p1 = np.asarray(m1[0], dtype=np.float64)
    v1 = np.asarray(m1[1], dtype=np.float64)
    p2 = np.asarray(m2[0], dtype=np.float64)
    v2 = np.asarray(m2[1], dtype=np.float64)

    # Normalise directions
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    u = np.cross(v1, v2)
    u_norm = np.linalg.norm(u)

    if u_norm < _PARALLEL_EPS:
        # Parallel rays — return midpoint of origins as stable fallback
        mid = (p1 + p2) / 2.0
        return (float(mid[0]), float(mid[1]), float(mid[2]))

    u = u / u_norm

    # Build and solve  [v1 | u | -v2] @ [alpha, beta, gamma] = p2 - p1
    A = np.column_stack([v1, u, -v2])
    b = p2 - p1

    try:
        solution = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        mid = (p1 + p2) / 2.0
        return (float(mid[0]), float(mid[1]), float(mid[2]))

    alpha, beta, gamma = solution

    # Clamp to forward direction of each ray
    alpha = max(alpha, 0.0)
    gamma = max(gamma, 0.0)

    # Closest points on each ray
    q1 = p1 + alpha * v1
    q2 = p2 + gamma * v2

    # Midpoint
    q = (q1 + q2) / 2.0
    return (float(q[0]), float(q[1]), float(q[2]))


def _find_most_divergent_pair(
    measurements: List[Measurement],
) -> Tuple[int, int]:
    """Find the pair of measurements with the largest angle between rays.

    Used to pick a good initial guess for gradient descent: the two most
    divergent rays give the best-conditioned intersection.

    Returns:
        Indices (i, j) of the most divergent pair.
    """
    n = len(measurements)
    best_i, best_j = 0, 1
    min_abs_dot = float("inf")
    for i in range(n):
        vi = np.asarray(measurements[i][1], dtype=np.float64)
        vi = vi / np.linalg.norm(vi)
        for j in range(i + 1, n):
            vj = np.asarray(measurements[j][1], dtype=np.float64)
            vj = vj / np.linalg.norm(vj)
            abs_dot = abs(float(np.dot(vi, vj)))
            if abs_dot < min_abs_dot:
                min_abs_dot = abs_dot
                best_i, best_j = i, j
    return best_i, best_j


def _gradient_descent_core(
    measurements: List[Measurement],
    weights: Optional[List[float]],
    learning_rate: float,
    max_iterations: int,
    tolerance: float,
) -> Tuple[float, float, float]:
    """Core gradient descent shared by uniform and weighted variants.

    Loss (Traa Eq. 10, extended with weights):
        L(q) = sum_i  w_i * (p_i - q)^T (I - v_i v_i^T) (p_i - q)

    Gradient (Traa Eq. 12, extended with weights):
        dL/dq = -2 * sum_i  w_i * (I - v_i v_i^T) (p_i - q)
    """
    n = len(measurements)
    if n < 2:
        raise ValueError(
            f"At least 2 measurements required, got {n}."
        )

    # Pre-compute projection matrices and origins
    proj_matrices: List[np.ndarray] = []
    origins: List[np.ndarray] = []
    for origin, direction in measurements:
        v = np.asarray(direction, dtype=np.float64)
        v = v / np.linalg.norm(v)
        v = v.reshape(3, 1)
        proj_matrices.append(np.eye(3) - v @ v.T)
        origins.append(np.asarray(origin, dtype=np.float64))

    # Weights: normalise so they sum to 1 (mean-gradient instead of sum-gradient).
    # This keeps the learning rate stable regardless of n or weight scale.
    # The optimum is unchanged because
    # argmin sum(w_i * d_i^2) == argmin sum((w_i/c) * d_i^2) for any c > 0.
    if weights is None:
        w = [1.0 / n] * n
    else:
        w_sum = sum(weights)
        w = [wi / w_sum for wi in weights]

    # Initialise with midpoint of the two most divergent rays
    i, j = _find_most_divergent_pair(measurements)
    q = np.asarray(ray_intersection_2(measurements[i], measurements[j]),
                   dtype=np.float64)

    q_prev = np.zeros(3)

    for _ in range(max_iterations):
        grad = np.zeros(3)
        for k in range(n):
            grad -= 2.0 * w[k] * (proj_matrices[k] @ (origins[k] - q))

        q = q - learning_rate * grad

        if np.linalg.norm(q - q_prev) < tolerance:
            break
        q_prev = q.copy()

    return (float(q[0]), float(q[1]), float(q[2]))


def gradient_descent_uniform(
    measurements: List[Measurement],
    learning_rate: float = DEFAULT_LR,
    max_iterations: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOL,
) -> Tuple[float, float, float]:
    """Estimate target position by minimising sum of squared orthogonal distances.

    All measurements contribute equally (uniform weights).

    Initialisation: ray_intersection_2 of the two most divergent rays.

    Args:
        measurements: List of (origin, direction) tuples. Minimum 2.
        learning_rate: Step size (default 0.065, from Lampesberger).
        max_iterations: Maximum iterations (default 1000).
        tolerance: Convergence threshold on position change (default 1e-7).

    Returns:
        Estimated position (x, y, z) in meters.

    Raises:
        ValueError: If fewer than 2 measurements.
    """
    return _gradient_descent_core(
        measurements, weights=None,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def gradient_descent_weighted(
    measurements: List[Measurement],
    weights: List[float],
    learning_rate: float = DEFAULT_LR,
    max_iterations: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOL,
) -> Tuple[float, float, float]:
    """Estimate target position with confidence-weighted gradient descent.

    Loss: L(q) = sum_i  w_i * d_orth(q, ray_i)^2

    Weights typically come from confidence_to_weight() in confidence.py.
    They are NOT normalised internally — the caller controls the scale.

    Args:
        measurements: List of (origin, direction) tuples. Minimum 2.
        weights: Per-measurement weights (positive). Must match len(measurements).
        learning_rate: Step size (default 0.065).
        max_iterations: Maximum iterations (default 1000).
        tolerance: Convergence threshold on position change (default 1e-7).

    Returns:
        Estimated position (x, y, z) in meters.

    Raises:
        ValueError: If fewer than 2 measurements, weight count mismatch,
            or any weight is non-positive.
    """
    if len(weights) != len(measurements):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of measurements ({len(measurements)})."
        )
    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be positive.")

    return _gradient_descent_core(
        measurements, weights=weights,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def _gradient_descent_ground(
    measurements: List[Measurement],
    weights: Optional[List[float]],
    learning_rate: float,
    max_iterations: int,
    tolerance: float,
    target_z: float = 0.0,
) -> Tuple[float, float, float]:
    """Ground-constrained gradient descent: optimises only (X, Y) with Z fixed.

    Same loss as _gradient_descent_core but the update step zeroes the Z
    component of the gradient so the estimate never leaves the ground plane.
    """
    n = len(measurements)
    if n < 2:
        raise ValueError(f"At least 2 measurements required, got {n}.")

    proj_matrices: List[np.ndarray] = []
    origins: List[np.ndarray] = []
    for origin, direction in measurements:
        v = np.asarray(direction, dtype=np.float64)
        v = v / np.linalg.norm(v)
        v = v.reshape(3, 1)
        proj_matrices.append(np.eye(3) - v @ v.T)
        origins.append(np.asarray(origin, dtype=np.float64))

    if weights is None:
        w = [1.0 / n] * n
    else:
        w_sum = sum(weights)
        w = [wi / w_sum for wi in weights]

    i, j = _find_most_divergent_pair(measurements)
    q = np.asarray(ray_intersection_2(measurements[i], measurements[j]),
                   dtype=np.float64)
    q[2] = target_z

    q_prev = np.zeros(3)

    for _ in range(max_iterations):
        grad = np.zeros(3)
        for k in range(n):
            grad -= 2.0 * w[k] * (proj_matrices[k] @ (origins[k] - q))

        grad[2] = 0.0  # constrain to ground plane
        q = q - learning_rate * grad

        if np.linalg.norm(q - q_prev) < tolerance:
            break
        q_prev = q.copy()

    return (float(q[0]), float(q[1]), float(q[2]))


def gradient_descent_uniform_ground(
    measurements: List[Measurement],
    learning_rate: float = DEFAULT_LR,
    max_iterations: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOL,
    target_z: float = 0.0,
) -> Tuple[float, float, float]:
    """Ground-constrained uniform GD: optimises (X, Y) only, Z = target_z."""
    return _gradient_descent_ground(
        measurements, weights=None,
        learning_rate=learning_rate, max_iterations=max_iterations,
        tolerance=tolerance, target_z=target_z,
    )


def gradient_descent_weighted_ground(
    measurements: List[Measurement],
    weights: List[float],
    learning_rate: float = DEFAULT_LR,
    max_iterations: int = DEFAULT_MAX_ITER,
    tolerance: float = DEFAULT_TOL,
    target_z: float = 0.0,
) -> Tuple[float, float, float]:
    """Ground-constrained weighted GD: optimises (X, Y) only, Z = target_z."""
    if len(weights) != len(measurements):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of measurements ({len(measurements)})."
        )
    if any(w <= 0 for w in weights):
        raise ValueError("All weights must be positive.")

    return _gradient_descent_ground(
        measurements, weights=weights,
        learning_rate=learning_rate, max_iterations=max_iterations,
        tolerance=tolerance, target_z=target_z,
    )


def ransac_fusion(
    measurements: List[Measurement],
    n_iterations: int = 100,
    threshold_m: float = 5.0,
    rng: Optional[np.random.Generator] = None,
    ground_z: Optional[float] = None,
    confidences: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """RANSAC-based robust fusion (baseline for Experiment 6).

    Algorithm:
        1. For each iteration, sample 2 rays at random.
        2. Compute hypothesis via ray_intersection_2.
        3. Count inliers: measurements with orthogonal_distance < threshold_m.
        4. Keep the hypothesis with the most inliers.
        5. Re-estimate the final position using ALL inliers of the best
           hypothesis via gradient descent (weighted if confidences given).

    Args:
        measurements: List of (origin, direction) tuples. Minimum 2.
        n_iterations: Number of RANSAC iterations (default 100).
        threshold_m: Inlier distance threshold in meters (default 5.0).
        rng: Numpy random generator for reproducibility.
        ground_z: If set, re-estimation uses ground-constrained GD at this Z.
        confidences: If set, inlier refinement uses confidence-weighted GD.
            Must be aligned with measurements (same length, same order).

    Returns:
        Estimated position (x, y, z) in meters.

    Raises:
        ValueError: If fewer than 2 measurements.
    """
    n = len(measurements)
    if n < 2:
        raise ValueError(f"At least 2 measurements required, got {n}.")

    if rng is None:
        rng = np.random.default_rng()

    best_inlier_indices: List[int] = []
    best_hypothesis = ray_intersection_2(measurements[0], measurements[1])

    for _ in range(n_iterations):
        # Sample 2 distinct indices
        i, j = rng.choice(n, size=2, replace=False)
        hypothesis = ray_intersection_2(measurements[i], measurements[j])

        # Count inliers
        inlier_indices = []
        for k in range(n):
            dist = orthogonal_distance(
                hypothesis, measurements[k][0], measurements[k][1]
            )
            if dist < threshold_m:
                inlier_indices.append(k)

        if len(inlier_indices) > len(best_inlier_indices):
            best_inlier_indices = inlier_indices
            best_hypothesis = hypothesis

    # Re-estimate using all inliers of the best hypothesis
    if len(best_inlier_indices) >= 2:
        inlier_measurements = [measurements[i] for i in best_inlier_indices]

        if confidences is not None:
            from uav_vision.confidence import confidence_to_weight
            w = [confidence_to_weight(confidences[i]) for i in best_inlier_indices]
            if ground_z is not None:
                return gradient_descent_weighted_ground(
                    inlier_measurements, w, target_z=ground_z,
                )
            return gradient_descent_weighted(inlier_measurements, w)

        if ground_z is not None:
            return gradient_descent_uniform_ground(
                inlier_measurements, target_z=ground_z,
            )
        return gradient_descent_uniform(inlier_measurements)

    # Fallback: not enough inliers, return best 2-ray hypothesis
    return best_hypothesis


def prosac_fusion(
    measurements: List[Measurement],
    confidences: List[float],
    n_iterations: int = 100,
    threshold_m: float = 5.0,
    rng: Optional[np.random.Generator] = None,
    ground_z: Optional[float] = None,
) -> Tuple[float, float, float]:
    """PROSAC fusion: sample from progressively larger confidence-ranked sets.

    Samples high-confidence measurements first, expanding the pool over
    iterations (Chum & Matas, CVPR 2005). Same model estimation and inlier
    test as ransac_fusion.

    Args:
        measurements: List of (origin, direction) tuples. Minimum 2.
        confidences: Per-measurement confidence scores (used for ranking).
        n_iterations: Number of iterations (default 100).
        threshold_m: Inlier distance threshold in meters (default 5.0).
        rng: Numpy random generator for reproducibility.
        ground_z: If set, re-estimation uses ground-constrained GD at this Z.

    Returns:
        Estimated position (x, y, z) in meters.

    Raises:
        ValueError: If fewer than 2 measurements or confidence count mismatch.
    """
    n = len(measurements)
    if n < 2:
        raise ValueError(f"At least 2 measurements required, got {n}.")
    if len(confidences) != n:
        raise ValueError("confidences length must match measurements length.")

    if rng is None:
        rng = np.random.default_rng()

    order = sorted(range(n), key=lambda i: -confidences[i])
    sorted_meas = [measurements[order[i]] for i in range(n)]

    best_inlier_indices: List[int] = []
    best_hypothesis = ray_intersection_2(sorted_meas[0], sorted_meas[1])

    for t in range(n_iterations):
        pool_size = min(n, max(2, 2 + t * (n - 2) // n_iterations))

        i = pool_size - 1
        j = rng.integers(0, pool_size - 1)
        hypothesis = ray_intersection_2(sorted_meas[i], sorted_meas[j])

        inlier_indices = []
        for k in range(n):
            dist = orthogonal_distance(
                hypothesis, sorted_meas[k][0], sorted_meas[k][1]
            )
            if dist < threshold_m:
                inlier_indices.append(k)

        if len(inlier_indices) > len(best_inlier_indices):
            best_inlier_indices = inlier_indices
            best_hypothesis = hypothesis

    if len(best_inlier_indices) >= 2:
        inlier_meas = [sorted_meas[i] for i in best_inlier_indices]
        if ground_z is not None:
            return gradient_descent_uniform_ground(inlier_meas, target_z=ground_z)
        return gradient_descent_uniform(inlier_meas)

    return best_hypothesis


def lo_ransac_fusion(
    measurements: List[Measurement],
    n_iterations: int = 100,
    threshold_m: float = 5.0,
    lo_steps: int = 3,
    rng: Optional[np.random.Generator] = None,
    ground_z: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Locally-Optimized RANSAC: refine each new best hypothesis with its inliers.

    Like vanilla RANSAC, but whenever a new best hypothesis is found, runs
    lo_steps rounds of re-estimation and inlier recount to improve it
    (Chum, Matas & Kittler, DAGM 2003).

    Args:
        measurements: List of (origin, direction) tuples. Minimum 2.
        n_iterations: Number of outer iterations (default 100).
        threshold_m: Inlier distance threshold in meters (default 5.0).
        lo_steps: Number of local-optimization refinement rounds (default 3).
        rng: Numpy random generator for reproducibility.
        ground_z: If set, uses ground-constrained GD at this Z.

    Returns:
        Estimated position (x, y, z) in meters.

    Raises:
        ValueError: If fewer than 2 measurements.
    """
    n = len(measurements)
    if n < 2:
        raise ValueError(f"At least 2 measurements required, got {n}.")

    if rng is None:
        rng = np.random.default_rng()

    def _gd(meas):
        if ground_z is not None:
            return gradient_descent_uniform_ground(meas, target_z=ground_z)
        return gradient_descent_uniform(meas)

    def _count_inliers(hypothesis):
        return [
            k for k in range(n)
            if orthogonal_distance(
                hypothesis, measurements[k][0], measurements[k][1],
            ) < threshold_m
        ]

    best_inlier_indices: List[int] = []
    best_hypothesis = ray_intersection_2(measurements[0], measurements[1])

    for _ in range(n_iterations):
        i, j = rng.choice(n, size=2, replace=False)
        hypothesis = ray_intersection_2(measurements[i], measurements[j])
        inlier_indices = _count_inliers(hypothesis)

        if len(inlier_indices) > len(best_inlier_indices):
            for _ in range(lo_steps):
                if len(inlier_indices) < 2:
                    break
                inlier_meas = [measurements[k] for k in inlier_indices]
                hypothesis = _gd(inlier_meas)
                inlier_indices = _count_inliers(hypothesis)

            if len(inlier_indices) > len(best_inlier_indices):
                best_inlier_indices = inlier_indices
                best_hypothesis = hypothesis

    if len(best_inlier_indices) >= 2:
        return _gd([measurements[i] for i in best_inlier_indices])

    return best_hypothesis
