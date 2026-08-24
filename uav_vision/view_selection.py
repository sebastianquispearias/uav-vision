"""
View selection algorithms for bearing-only multi-UAV target geolocation.

The main contribution of the paper: a joint criterion that combines
geometric angular diversity with detection confidence to greedily
select the best K views from a large set of candidate measurements.

Also provides baseline selectors (confidence-only, geometry-only, random)
for ablation in Experiment 6.

A measurement follows the same format as fusion.py:
    (origin, direction) where origin = (x,y,z), direction = (dx,dy,dz)

All functions are pure. No simulator or protocol dependencies.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

# Type alias (same as fusion.py, kept independent to avoid coupling)
Measurement = Tuple[Tuple[float, float, float], Tuple[float, float, float]]

# Defaults from CLAUDE.md experimental parameters
DEFAULT_K: int = 30
DEFAULT_ALPHA: float = 0.2
DEFAULT_MIN_ANGLE_DEG: float = 10.0
_MAX_DIVERSITY_DEG: float = 90.0  # cap for angular diversity


def compute_angular_diversity(
    candidate_dir: Tuple[float, float, float],
    selected_dirs: List[Tuple[float, float, float]],
) -> float:
    """Minimum angle (degrees) between a candidate ray and all selected rays.

    Measures how geometrically different the candidate is from the rays
    already in the selection. Higher = more diverse = better geometry.

    The result is capped to [0, 90] degrees. Angles above 90 are clamped
    because the selection score normalises by 90 and larger angles do not
    provide additional geometric benefit for bearing-only localisation.
    Uses abs(dot product) so that parallel and anti-parallel rays both
    yield ~0 diversity.

    Args:
        candidate_dir: Direction vector of the candidate ray.
        selected_dirs: Direction vectors of already-selected rays.

    Returns:
        Minimum angle in degrees, in [0, 90]. Returns 90.0 if
        selected_dirs is empty (maximum diversity by convention).
    """
    if not selected_dirs:
        return _MAX_DIVERSITY_DEG

    c = np.asarray(candidate_dir, dtype=np.float64)
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-12:
        return 0.0
    c = c / c_norm

    min_angle = _MAX_DIVERSITY_DEG
    for s in selected_dirs:
        sv = np.asarray(s, dtype=np.float64)
        sv_norm = np.linalg.norm(sv)
        if sv_norm < 1e-12:
            continue
        sv = sv / sv_norm
        dot = float(np.clip(np.dot(c, sv), -1.0, 1.0))
        # abs(dot) so parallel and anti-parallel both give ~0 diversity
        angle_deg = math.degrees(math.acos(abs(dot)))
        if angle_deg < min_angle:
            min_angle = angle_deg

    return min(min_angle, _MAX_DIVERSITY_DEG)


def compute_selection_score(
    angular_diversity_deg: float,
    confidence: float,
    alpha: float = DEFAULT_ALPHA,
    max_angle: float = _MAX_DIVERSITY_DEG,
) -> float:
    """Combined view selection score (geometry + detection quality).

    score = alpha * (angular_diversity / max_angle) + (1 - alpha) * confidence

    Args:
        angular_diversity_deg: Angular diversity in degrees (clamped to [0, max_angle]).
        confidence: Detection confidence in [0, 1].
        alpha: Balance parameter. 0 = confidence only, 1 = geometry only.
        max_angle: Normalisation angle in degrees (default 90).

    Returns:
        Score in [0, 1].

    Raises:
        ValueError: If alpha is not in [0, 1].
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

    geo_norm = min(max(angular_diversity_deg, 0.0), max_angle) / max_angle
    return alpha * geo_norm + (1.0 - alpha) * confidence


def select_best_views(
    measurements: List[Measurement],
    confidences: List[float],
    k: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
) -> List[int]:
    """Greedy joint view selection (MAIN CONTRIBUTION).

    Algorithm:
        1. Select the measurement with the highest confidence first.
        2. For each remaining candidate:
           a. Compute angular diversity w.r.t. already-selected rays.
           b. Discard if min angle < min_angle_deg.
           c. Compute combined score.
        3. Select candidate with highest score (ties: lowest index).
        4. Repeat until k selected or no valid candidates remain.

    Args:
        measurements: List of (origin, direction) tuples.
        confidences: Per-measurement confidence scores.
        k: Maximum number of views to select.
        alpha: Balance between geometry (1) and confidence (0).
        min_angle_deg: Minimum angular diversity to accept a candidate.

    Returns:
        List of selected indices in order of selection. Length <= k.

    Raises:
        ValueError: If inputs are empty or lengths don't match.
    """
    n = len(measurements)
    if n == 0:
        raise ValueError("No measurements provided.")
    if len(confidences) != n:
        raise ValueError(
            f"Number of confidences ({len(confidences)}) must match "
            f"number of measurements ({n})."
        )

    # Extract direction vectors
    directions = [m[1] for m in measurements]

    # Step 1: select the measurement with highest confidence (tie: lowest index)
    first_idx = max(range(n), key=lambda i: (confidences[i], -i))

    selected_indices: List[int] = [first_idx]
    selected_dirs: List[Tuple[float, float, float]] = [directions[first_idx]]
    remaining = set(range(n)) - {first_idx}

    # Steps 2-4: greedy selection
    while len(selected_indices) < k and remaining:
        best_idx = -1
        best_score = -1.0

        for idx in sorted(remaining):  # sorted for deterministic tie-breaking
            ang_div = compute_angular_diversity(directions[idx], selected_dirs)
            if ang_div < min_angle_deg:
                continue
            score = compute_selection_score(ang_div, confidences[idx], alpha)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            break  # no valid candidates left

        selected_indices.append(best_idx)
        selected_dirs.append(directions[best_idx])
        remaining.discard(best_idx)

    return selected_indices


# ---------------------------------------------------------------------------
# FIM-based greedy selector
# ---------------------------------------------------------------------------


def _estimate_target_from_rays(
    measurements: List[Measurement],
    ground_z: float = 0.0,
) -> Tuple[float, float]:
    """Estimate target (x, y) from ray-ground intersections (median)."""
    xs, ys = [], []
    for origin, direction in measurements:
        if direction[2] >= 0:
            continue
        t = (ground_z - origin[2]) / direction[2]
        if t <= 0:
            continue
        xs.append(origin[0] + t * direction[0])
        ys.append(origin[1] + t * direction[1])
    if xs:
        return (float(np.median(xs)), float(np.median(ys)))
    ox = np.mean([m[0][0] for m in measurements])
    oy = np.mean([m[0][1] for m in measurements])
    return (float(ox), float(oy))


def _pinhole_jacobian_2d(
    drone_pos: Tuple[float, float, float],
    target_xy: Tuple[float, float],
    yaw_deg: float,
    pitch_deg: float = -75.0,
    focal_px: float = 1130.0,
    img_w: int = 1920,
    img_h: int = 1080,
    ground_z: float = 0.0,
    delta: float = 0.01,
) -> np.ndarray:
    """Numerical Jacobian of pinhole projection w.r.t. target (tx, ty).

    Returns 2x2 matrix [[du/dtx, du/dty], [dv/dtx, dv/dty]].
    Returns zeros if the target is outside the camera FOV.
    """
    from uav_vision.pinhole_local import project_to_pixel

    tx, ty = target_xy
    target_0 = (tx, ty, ground_z)
    target_dx = (tx + delta, ty, ground_z)
    target_dy = (tx, ty + delta, ground_z)

    px0 = project_to_pixel(drone_pos, target_0, yaw_deg, pitch_deg, focal_px, img_w, img_h)
    px_dx = project_to_pixel(drone_pos, target_dx, yaw_deg, pitch_deg, focal_px, img_w, img_h)
    px_dy = project_to_pixel(drone_pos, target_dy, yaw_deg, pitch_deg, focal_px, img_w, img_h)

    if px0 is None or px_dx is None or px_dy is None:
        return np.zeros((2, 2))

    return np.array([
        [(px_dx[0] - px0[0]) / delta, (px_dy[0] - px0[0]) / delta],
        [(px_dx[1] - px0[1]) / delta, (px_dy[1] - px0[1]) / delta],
    ])


def select_greedy_fim(
    measurements: List[Measurement],
    confidences: List[float],
    k: int = DEFAULT_K,
    target_estimate: Optional[Tuple[float, float]] = None,
    ground_z: float = 0.0,
    pitch_deg: float = -75.0,
    focal_px: float = 1130.0,
    img_w: int = 1920,
    img_h: int = 1080,
    sigma_constant: float = 5.0,
) -> List[int]:
    """Greedy FIM-based view selection.

    Selects measurements that maximise det(FIM) of the ground-constrained
    bearing-only localisation problem. The FIM encodes both geometry
    (via the pinhole Jacobian) and measurement quality (via 1/sigma^2).

    Args:
        measurements: List of (origin, direction) tuples.
        confidences: Per-measurement confidence scores.
        k: Maximum number of views to select.
        target_estimate: Approximate target (tx, ty). Computed from
            ray-ground intersections if None.
        ground_z: Target height (ground constraint).
        pitch_deg: Camera pitch in degrees.
        focal_px: Focal length in pixels.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        sigma_constant: Noise model constant C where sigma = C / conf.

    Returns:
        List of selected indices in order of selection. Length <= k.

    Raises:
        ValueError: If inputs are empty or lengths don't match.
    """
    n = len(measurements)
    if n == 0:
        raise ValueError("No measurements provided.")
    if len(confidences) != n:
        raise ValueError(
            f"Number of confidences ({len(confidences)}) must match "
            f"number of measurements ({n})."
        )

    if target_estimate is None:
        target_estimate = _estimate_target_from_rays(measurements, ground_z)

    fim_contributions: List[np.ndarray] = []
    for i in range(n):
        drone_pos = measurements[i][0]
        direction = measurements[i][1]
        yaw_approx = math.degrees(math.atan2(direction[0], direction[1]))
        J = _pinhole_jacobian_2d(
            drone_pos, target_estimate, yaw_approx,
            pitch_deg, focal_px, img_w, img_h, ground_z,
        )
        w = (confidences[i] / sigma_constant) ** 2
        fim_contributions.append(w * J.T @ J)

    selected: List[int] = []
    remaining = set(range(n))
    fim_total = np.zeros((2, 2))

    while len(selected) < k and remaining:
        best_idx = -1
        best_det = -1.0

        for idx in sorted(remaining):
            candidate_fim = fim_total + fim_contributions[idx]
            d = candidate_fim[0, 0] * candidate_fim[1, 1] - candidate_fim[0, 1] * candidate_fim[1, 0]
            if d > best_det:
                best_det = d
                best_idx = idx

        if best_idx == -1:
            break

        selected.append(best_idx)
        fim_total = fim_total + fim_contributions[best_idx]
        remaining.discard(best_idx)

    return selected


# ---------------------------------------------------------------------------
# Bias-aware FIM selector (experimental)
# ---------------------------------------------------------------------------


def _fim_drone_contribution(
    S_j: np.ndarray,
    sigma_gps: float,
) -> np.ndarray:
    """Marginal FIM contribution of one drone after bias marginalization.

    Uses the Schur complement to marginalize out a shared 2D GPS bias:
        F_marginal = S_j - S_j @ inv(S_j + (1/sigma_gps^2)*I) @ S_j

    Saturates at (1/sigma_gps^2)*I as S_j grows, naturally limiting
    the information from any single drone.
    """
    if sigma_gps <= 0:
        return S_j.copy()
    if np.allclose(S_j, 0):
        return np.zeros((2, 2))
    prior = (1.0 / sigma_gps**2) * np.eye(2)
    F_bb = S_j + prior
    return S_j - S_j @ np.linalg.solve(F_bb, S_j)


def select_greedy_fim_standard(
    measurements: List[Measurement],
    confidences: List[float],
    k: int = DEFAULT_K,
    drone_ids: Optional[List[int]] = None,
    sigma_gps: float = 3.0,
    target_estimate: Optional[Tuple[float, float]] = None,
    ground_z: float = 0.0,
    pitch_deg: float = -75.0,
    focal_px: float = 1130.0,
    img_w: int = 1920,
    img_h: int = 1080,
    sigma_constant: float = 5.0,
) -> List[int]:
    """Standard FIM selector (ignores per-drone bias). Wrapper for comparison."""
    return select_greedy_fim(
        measurements, confidences, k, target_estimate,
        ground_z, pitch_deg, focal_px, img_w, img_h, sigma_constant,
    )


def select_greedy_fim_bias_corrected(
    measurements: List[Measurement],
    confidences: List[float],
    k: int = DEFAULT_K,
    drone_ids: Optional[List[int]] = None,
    sigma_gps: float = 3.0,
    target_estimate: Optional[Tuple[float, float]] = None,
    ground_z: float = 0.0,
    pitch_deg: float = -75.0,
    focal_px: float = 1130.0,
    img_w: int = 1920,
    img_h: int = 1080,
    sigma_constant: float = 5.0,
) -> List[int]:
    """Bias-aware FIM view selection via Schur-complement marginalization.

    Accounts for per-drone correlated GPS bias as a 2D nuisance variable.
    Information from each drone saturates at (1/sigma_gps^2)*I, naturally
    favoring cross-drone diversity.

    Args:
        measurements: List of (origin, direction) tuples.
        confidences: Per-measurement confidence scores.
        k: Maximum number of views to select.
        drone_ids: Per-measurement drone identifier (required).
        sigma_gps: GPS noise std in meters (from ScenarioConfig.gps_sigma_m).
        target_estimate: Approximate target (tx, ty). Auto-computed if None.
        ground_z: Target height constraint.
        pitch_deg: Camera pitch in degrees.
        focal_px: Focal length in pixels.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        sigma_constant: Noise model constant C where sigma_pixel = C / conf.

    Returns:
        List of selected indices in order of selection. Length <= k.

    Raises:
        ValueError: If inputs are empty, lengths don't match, or drone_ids missing.
    """
    n = len(measurements)
    if n == 0:
        raise ValueError("No measurements provided.")
    if len(confidences) != n:
        raise ValueError(
            f"Number of confidences ({len(confidences)}) must match "
            f"number of measurements ({n})."
        )
    if drone_ids is None or len(drone_ids) != n:
        raise ValueError("drone_ids is required and must match measurement count.")

    if target_estimate is None:
        target_estimate = _estimate_target_from_rays(measurements, ground_z)

    fim_contributions: List[np.ndarray] = []
    for i in range(n):
        drone_pos = measurements[i][0]
        direction = measurements[i][1]
        yaw_approx = math.degrees(math.atan2(direction[0], direction[1]))
        J = _pinhole_jacobian_2d(
            drone_pos, target_estimate, yaw_approx,
            pitch_deg, focal_px, img_w, img_h, ground_z,
        )
        w = (confidences[i] / sigma_constant) ** 2
        fim_contributions.append(w * J.T @ J)

    per_drone_S: dict = {}
    for did in set(drone_ids):
        per_drone_S[did] = np.zeros((2, 2))

    selected: List[int] = []
    remaining = set(range(n))
    fim_total = np.zeros((2, 2))

    while len(selected) < k and remaining:
        best_idx = -1
        best_det = -1.0

        for idx in sorted(remaining):
            did = drone_ids[idx]
            old_S = per_drone_S[did]
            new_S = old_S + fim_contributions[idx]
            old_marginal = _fim_drone_contribution(old_S, sigma_gps)
            new_marginal = _fim_drone_contribution(new_S, sigma_gps)
            candidate_fim = fim_total - old_marginal + new_marginal
            d = candidate_fim[0, 0] * candidate_fim[1, 1] - candidate_fim[0, 1] * candidate_fim[1, 0]
            if d > best_det:
                best_det = d
                best_idx = idx

        if best_idx == -1:
            break

        did = drone_ids[best_idx]
        per_drone_S[did] = per_drone_S[did] + fim_contributions[best_idx]
        old_marginal = _fim_drone_contribution(
            per_drone_S[did] - fim_contributions[best_idx], sigma_gps
        )
        new_marginal = _fim_drone_contribution(per_drone_S[did], sigma_gps)
        fim_total = fim_total - old_marginal + new_marginal
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


# ---------------------------------------------------------------------------
# Adaptive-alpha selector (EXPERIMENTAL — does NOT replace paper method)
# ---------------------------------------------------------------------------


def compute_angular_coverage(
    measurements: List[Measurement],
    drone_ids: List[int],
    target_estimate: Tuple[float, float],
) -> Tuple[float, dict]:
    """Compute circular angular span of drones around a target estimate.

    For each unique drone, computes its mean azimuth relative to target_estimate.
    Then computes the circular span (360 - largest angular gap).

    Args:
        measurements: List of (origin, direction) tuples.
        drone_ids: Per-measurement drone identifiers.
        target_estimate: (x, y) of the coarse target position.

    Returns:
        (span_deg, info_dict) where info_dict contains:
            - 'per_drone_azimuth': {drone_id: azimuth_deg}
            - 'sorted_azimuths': list of sorted azimuths
            - 'max_gap_deg': largest angular gap
    """
    from collections import defaultdict

    tx, ty = target_estimate
    drone_positions = defaultdict(list)
    for i, did in enumerate(drone_ids):
        ox, oy = measurements[i][0][0], measurements[i][0][1]
        drone_positions[did].append((ox, oy))

    per_drone_azimuth = {}
    for did, positions in drone_positions.items():
        mean_x = sum(p[0] for p in positions) / len(positions)
        mean_y = sum(p[1] for p in positions) / len(positions)
        az = math.degrees(math.atan2(mean_x - tx, mean_y - ty)) % 360
        per_drone_azimuth[did] = az

    azimuths = sorted(per_drone_azimuth.values())
    n_drones = len(azimuths)

    if n_drones < 2:
        return 0.0, {
            'per_drone_azimuth': per_drone_azimuth,
            'sorted_azimuths': azimuths,
            'max_gap_deg': 360.0,
        }

    max_gap = 0.0
    for i in range(n_drones):
        next_i = (i + 1) % n_drones
        if next_i == 0:
            gap = (azimuths[0] + 360.0) - azimuths[-1]
        else:
            gap = azimuths[next_i] - azimuths[i]
        if gap > max_gap:
            max_gap = gap

    span_deg = 360.0 - max_gap

    return span_deg, {
        'per_drone_azimuth': per_drone_azimuth,
        'sorted_azimuths': azimuths,
        'max_gap_deg': max_gap,
    }


def select_best_views_adaptive_alpha(
    measurements: List[Measurement],
    confidences: List[float],
    k: int = DEFAULT_K,
    drone_ids: Optional[List[int]] = None,
    alpha_min: float = 0.05,
    alpha_max: float = 0.30,
    span_ref_deg: float = 120.0,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
    target_estimate: Optional[Tuple[float, float]] = None,
    ground_z: float = 0.0,
) -> Tuple[List[int], dict]:
    """EXPERIMENTAL adaptive-alpha greedy view selector.

    Same greedy logic as select_best_views, but alpha is computed online
    based on the angular coverage of drones around a coarse target estimate.

    Narrow coverage (same-side geometry) -> low alpha (trust confidence).
    Broad coverage (multi-side geometry) -> high alpha (trust geometry).

    Args:
        measurements: List of (origin, direction) tuples.
        confidences: Per-measurement confidence scores.
        k: Maximum number of views to select.
        drone_ids: Per-measurement drone identifiers (required).
        alpha_min: Alpha when coverage is zero.
        alpha_max: Alpha when coverage is at or above span_ref_deg.
        span_ref_deg: Reference span for full coverage (degrees).
        min_angle_deg: Minimum angular diversity to accept a candidate.
        target_estimate: Coarse (x, y) target estimate. Auto-computed if None.
        ground_z: Ground height for ray-ground intersection.

    Returns:
        (selected_indices, debug_info) where debug_info contains:
            - 'alpha_eff': the computed adaptive alpha
            - 'span_deg': angular span of drones
            - 'q': coverage metric in [0, 1]
            - 'target_estimate': (x, y) used
            - 'coverage_info': full dict from compute_angular_coverage
    """
    n = len(measurements)
    if n == 0:
        raise ValueError("No measurements provided.")
    if len(confidences) != n:
        raise ValueError(
            f"Number of confidences ({len(confidences)}) must match "
            f"number of measurements ({n})."
        )
    if drone_ids is None or len(drone_ids) != n:
        raise ValueError("drone_ids is required and must match measurement count.")

    if target_estimate is None:
        target_estimate = _estimate_target_from_rays(measurements, ground_z)

    span_deg, coverage_info = compute_angular_coverage(
        measurements, drone_ids, target_estimate,
    )

    q = max(0.0, min(span_deg / span_ref_deg, 1.0))
    alpha_eff = alpha_min + q * (alpha_max - alpha_min)

    # --- Greedy selection (same logic as select_best_views) ---
    directions = [m[1] for m in measurements]
    first_idx = max(range(n), key=lambda i: (confidences[i], -i))

    selected_indices: List[int] = [first_idx]
    selected_dirs: List[Tuple[float, float, float]] = [directions[first_idx]]
    remaining = set(range(n)) - {first_idx}

    while len(selected_indices) < k and remaining:
        best_idx = -1
        best_score = -1.0

        for idx in sorted(remaining):
            ang_div = compute_angular_diversity(directions[idx], selected_dirs)
            if ang_div < min_angle_deg:
                continue
            score = compute_selection_score(ang_div, confidences[idx], alpha_eff)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        selected_dirs.append(directions[best_idx])
        remaining.discard(best_idx)

    debug_info = {
        'alpha_eff': alpha_eff,
        'span_deg': span_deg,
        'q': q,
        'target_estimate': target_estimate,
        'coverage_info': coverage_info,
    }

    return selected_indices, debug_info


# ---------------------------------------------------------------------------
# Drone-saturation penalty selector (EXPERIMENTAL)
# ---------------------------------------------------------------------------


def select_best_views_drone_penalty(
    measurements: List[Measurement],
    confidences: List[float],
    k: int = DEFAULT_K,
    drone_ids: Optional[List[int]] = None,
    w_conf: float = 0.8,
    w_geom: float = 0.2,
    lambda_rep: float = 0.25,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
) -> List[int]:
    """EXPERIMENTAL greedy selector with per-drone saturation penalty.

    Like select_best_views, but penalizes candidates from drones already
    heavily represented in the selected set. Encourages cross-drone balance
    without requiring FIM computation or sigma_gps knowledge.
    Default weights match best_views (alpha=0.2 -> w_geom=0.2, w_conf=0.8).

    Score:
        score_i = w_conf * conf_i + w_geom * (ang_div_i / 90)
                  - lambda_rep * (n_selected_same_drone / max(1, n_selected))

    Args:
        measurements: List of (origin, direction) tuples.
        confidences: Per-measurement confidence scores.
        k: Maximum number of views to select.
        drone_ids: Per-measurement drone identifiers (required).
        w_conf: Weight for confidence term.
        w_geom: Weight for geometry term.
        lambda_rep: Weight for repetition penalty.
        min_angle_deg: Minimum angular diversity to accept a candidate.

    Returns:
        List of selected indices in order of selection. Length <= k.
    """
    n = len(measurements)
    if n == 0:
        raise ValueError("No measurements provided.")
    if len(confidences) != n:
        raise ValueError(
            f"Number of confidences ({len(confidences)}) must match "
            f"number of measurements ({n})."
        )
    if drone_ids is None or len(drone_ids) != n:
        raise ValueError("drone_ids is required and must match measurement count.")

    directions = [m[1] for m in measurements]
    first_idx = max(range(n), key=lambda i: (confidences[i], -i))

    selected_indices: List[int] = [first_idx]
    selected_dirs: List[Tuple[float, float, float]] = [directions[first_idx]]
    remaining = set(range(n)) - {first_idx}

    from collections import Counter
    drone_counts: Counter = Counter()
    drone_counts[drone_ids[first_idx]] += 1

    while len(selected_indices) < k and remaining:
        best_idx = -1
        best_score = -float('inf')
        n_selected = len(selected_indices)

        for idx in sorted(remaining):
            ang_div = compute_angular_diversity(directions[idx], selected_dirs)
            if ang_div < min_angle_deg:
                continue

            conf_norm = confidences[idx]
            geom_norm = min(ang_div, _MAX_DIVERSITY_DEG) / _MAX_DIVERSITY_DEG
            penalty = drone_counts[drone_ids[idx]] / max(1, n_selected)

            score = w_conf * conf_norm + w_geom * geom_norm - lambda_rep * penalty
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        selected_dirs.append(directions[best_idx])
        drone_counts[drone_ids[best_idx]] += 1
        remaining.discard(best_idx)

    return selected_indices


def select_best_views_relative_penalty(
    measurements: List[Measurement],
    confidences: List[float],
    k: int = DEFAULT_K,
    drone_ids: Optional[List[int]] = None,
    w_conf: float = 0.8,
    w_geom: float = 0.2,
    lambda_rel: float = 0.25,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
) -> Tuple[List[int], dict]:
    """EXPERIMENTAL greedy selector with relative per-drone penalty.

    Only penalizes a drone when it is selected MORE than its availability
    share (fraction of total measurements from that drone). This avoids
    forcing balance when the default distribution is already reasonable.

    Score:
        score_i = w_conf * conf_i + w_geom * (ang_div_i / 90)
                  - lambda_rel * max(0, R_j - A_j)

    Where:
        A_j = N_j / sum(N_k)  -- availability fraction
        R_j = n_j_selected / max(1, n_selected)  -- current selection fraction

    Returns:
        (selected_indices, debug_info) where debug_info contains penalty stats.
    """
    n = len(measurements)
    if n == 0:
        raise ValueError("No measurements provided.")
    if len(confidences) != n:
        raise ValueError(
            f"Number of confidences ({len(confidences)}) must match "
            f"number of measurements ({n})."
        )
    if drone_ids is None or len(drone_ids) != n:
        raise ValueError("drone_ids is required and must match measurement count.")

    from collections import Counter

    # Compute per-drone availability fraction (once)
    total_per_drone = Counter(drone_ids)
    availability = {did: count / n for did, count in total_per_drone.items()}

    directions = [m[1] for m in measurements]
    first_idx = max(range(n), key=lambda i: (confidences[i], -i))

    selected_indices: List[int] = [first_idx]
    selected_dirs: List[Tuple[float, float, float]] = [directions[first_idx]]
    remaining = set(range(n)) - {first_idx}

    drone_selected: Counter = Counter()
    drone_selected[drone_ids[first_idx]] += 1

    # Debug: track penalty activations
    penalty_activations = []  # list of (drone_id, penalty_value) for each eval with penalty > 0
    total_evals = 0

    while len(selected_indices) < k and remaining:
        best_idx = -1
        best_score = -float('inf')
        n_selected = len(selected_indices)

        for idx in sorted(remaining):
            ang_div = compute_angular_diversity(directions[idx], selected_dirs)
            if ang_div < min_angle_deg:
                continue

            total_evals += 1
            conf_norm = confidences[idx]
            geom_norm = min(ang_div, _MAX_DIVERSITY_DEG) / _MAX_DIVERSITY_DEG

            did = drone_ids[idx]
            r_j = drone_selected[did] / max(1, n_selected)
            a_j = availability[did]
            rel_pen = max(0.0, r_j - a_j)

            if rel_pen > 0:
                penalty_activations.append((did, rel_pen))

            score = w_conf * conf_norm + w_geom * geom_norm - lambda_rel * rel_pen
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        selected_dirs.append(directions[best_idx])
        drone_selected[drone_ids[best_idx]] += 1
        remaining.discard(best_idx)

    debug_info = {
        'availability': dict(availability),
        'final_selection_frac': {did: drone_selected[did] / max(1, len(selected_indices))
                                 for did in sorted(total_per_drone.keys())},
        'total_evals': total_evals,
        'penalty_activations': len(penalty_activations),
        'mean_penalty_when_active': (float(np.mean([p[1] for p in penalty_activations]))
                                     if penalty_activations else 0.0),
        'activations_per_drone': dict(Counter(p[0] for p in penalty_activations)),
    }

    return selected_indices, debug_info


# ---------------------------------------------------------------------------
# Initial-coverage selector (EXPERIMENTAL)
# ---------------------------------------------------------------------------


def select_best_views_initial_coverage(
    measurements: List[Measurement],
    confidences: List[float],
    k: int = DEFAULT_K,
    drone_ids: Optional[List[int]] = None,
    alpha: float = DEFAULT_ALPHA,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
    n_bootstrap: int = 6,
    max_per_drone_bootstrap: int = 2,
) -> Tuple[List[int], dict]:
    """EXPERIMENTAL: best_views with early cross-drone coverage enforcement.

    During the first n_bootstrap selections, caps each drone at
    max_per_drone_bootstrap views. After that, reverts to pure best_views.

    Returns:
        (selected_indices, debug_info) with bootstrap diagnostics.
    """
    n = len(measurements)
    if n == 0:
        raise ValueError("No measurements provided.")
    if len(confidences) != n:
        raise ValueError(
            f"Number of confidences ({len(confidences)}) must match "
            f"number of measurements ({n})."
        )
    if drone_ids is None or len(drone_ids) != n:
        raise ValueError("drone_ids is required and must match measurement count.")

    from collections import Counter

    directions = [m[1] for m in measurements]
    first_idx = max(range(n), key=lambda i: (confidences[i], -i))

    selected_indices: List[int] = [first_idx]
    selected_dirs: List[Tuple[float, float, float]] = [directions[first_idx]]
    remaining = set(range(n)) - {first_idx}

    drone_counts: Counter = Counter()
    drone_counts[drone_ids[first_idx]] += 1

    cap_blocks = 0
    d4_id = max(set(drone_ids))
    d4_first_step = None
    if drone_ids[first_idx] == d4_id:
        d4_first_step = 1

    while len(selected_indices) < k and remaining:
        best_idx = -1
        best_score = -1.0
        step = len(selected_indices) + 1
        in_bootstrap = step <= n_bootstrap

        # First pass: respect cap
        for idx in sorted(remaining):
            ang_div = compute_angular_diversity(directions[idx], selected_dirs)
            if ang_div < min_angle_deg:
                continue
            if in_bootstrap and drone_counts[drone_ids[idx]] >= max_per_drone_bootstrap:
                cap_blocks += 1
                continue
            score = compute_selection_score(ang_div, confidences[idx], alpha)
            if score > best_score:
                best_score = score
                best_idx = idx

        # Fallback: if all valid candidates are capped, lift cap for this step
        if best_idx == -1 and in_bootstrap:
            for idx in sorted(remaining):
                ang_div = compute_angular_diversity(directions[idx], selected_dirs)
                if ang_div < min_angle_deg:
                    continue
                score = compute_selection_score(ang_div, confidences[idx], alpha)
                if score > best_score:
                    best_score = score
                    best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        selected_dirs.append(directions[best_idx])
        drone_counts[drone_ids[best_idx]] += 1
        remaining.discard(best_idx)

        if drone_ids[best_idx] == d4_id and d4_first_step is None:
            d4_first_step = step

    debug_info = {
        'cap_blocks': cap_blocks,
        'n_bootstrap': n_bootstrap,
        'max_per_drone_bootstrap': max_per_drone_bootstrap,
        'd4_first_step': d4_first_step,
        'final_drone_dist': dict(drone_counts),
    }

    return selected_indices, debug_info


# ---------------------------------------------------------------------------
# Drone-quota selector (EXPERIMENTAL)
# ---------------------------------------------------------------------------


def _compute_drone_quotas(
    drone_ids: List[int],
    confidences: List[float],
    k_eff: int = 12,
    min_per_drone: int = 2,
) -> dict:
    """Compute target quotas per drone based on availability and confidence."""
    from collections import Counter, defaultdict

    total_per_drone = Counter(drone_ids)
    conf_per_drone = defaultdict(list)
    for i, did in enumerate(drone_ids):
        conf_per_drone[did].append(confidences[i])

    n_total = len(drone_ids)
    scores = {}
    for did in sorted(total_per_drone.keys()):
        avail = total_per_drone[did] / n_total
        mean_conf = float(np.mean(conf_per_drone[did]))
        scores[did] = avail * mean_conf

    score_sum = sum(scores.values())
    if score_sum < 1e-12:
        # Uniform fallback
        n_drones = len(scores)
        return {did: max(min_per_drone, k_eff // n_drones) for did in scores}

    # Raw quotas
    raw = {did: k_eff * s / score_sum for did, s in scores.items()}

    # Apply minimum, clamp to available candidates
    quotas = {}
    for did in sorted(raw.keys()):
        q = max(round(raw[did]), min_per_drone if total_per_drone[did] >= min_per_drone else 0)
        q = min(q, total_per_drone[did])
        quotas[did] = q

    # Adjust sum to k_eff
    diff = sum(quotas.values()) - k_eff
    sorted_drones = sorted(quotas.keys(), key=lambda d: raw[d] - quotas[d])
    i = 0
    while diff != 0 and i < len(sorted_drones) * 2:
        did = sorted_drones[i % len(sorted_drones)]
        if diff > 0 and quotas[did] > min_per_drone:
            quotas[did] -= 1
            diff -= 1
        elif diff < 0 and quotas[did] < total_per_drone[did]:
            quotas[did] += 1
            diff += 1
        i += 1

    return quotas


def select_best_views_drone_quota(
    measurements: List[Measurement],
    confidences: List[float],
    drone_ids: Optional[List[int]] = None,
    alpha: float = DEFAULT_ALPHA,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
    k_eff: int = 12,
    min_per_drone: int = 2,
) -> Tuple[List[int], dict]:
    """EXPERIMENTAL: quota-constrained greedy view selector.

    Computes per-drone target quotas, then selects views using best_views
    scoring while enforcing quotas. Stops at k_eff total selections.

    Drones at quota are blocked while any drone is below quota.
    Fallback: if all valid candidates are from satisfied drones, lift block.
    """
    n = len(measurements)
    if n == 0:
        raise ValueError("No measurements provided.")
    if len(confidences) != n:
        raise ValueError(
            f"Number of confidences ({len(confidences)}) must match "
            f"number of measurements ({n})."
        )
    if drone_ids is None or len(drone_ids) != n:
        raise ValueError("drone_ids is required and must match measurement count.")

    from collections import Counter

    quotas = _compute_drone_quotas(drone_ids, confidences, k_eff, min_per_drone)

    directions = [m[1] for m in measurements]
    first_idx = max(range(n), key=lambda i: (confidences[i], -i))

    selected_indices: List[int] = [first_idx]
    selected_dirs: List[Tuple[float, float, float]] = [directions[first_idx]]
    remaining = set(range(n)) - {first_idx}

    drone_counts: Counter = Counter()
    drone_counts[drone_ids[first_idx]] += 1

    d4_id = max(set(drone_ids))
    d4_first_step = 1 if drone_ids[first_idx] == d4_id else None
    quota_blocks = 0

    while len(selected_indices) < k_eff and remaining:
        step = len(selected_indices) + 1
        any_below_quota = any(
            drone_counts[did] < quotas[did] for did in quotas
        )

        best_idx = -1
        best_score = -1.0

        # First pass: respect quota blocking
        for idx in sorted(remaining):
            did = drone_ids[idx]
            if any_below_quota and drone_counts[did] >= quotas[did]:
                quota_blocks += 1
                continue
            ang_div = compute_angular_diversity(directions[idx], selected_dirs)
            if ang_div < min_angle_deg:
                continue
            score = compute_selection_score(ang_div, confidences[idx], alpha)
            if score > best_score:
                best_score = score
                best_idx = idx

        # Fallback: lift quota block
        if best_idx == -1:
            for idx in sorted(remaining):
                ang_div = compute_angular_diversity(directions[idx], selected_dirs)
                if ang_div < min_angle_deg:
                    continue
                score = compute_selection_score(ang_div, confidences[idx], alpha)
                if score > best_score:
                    best_score = score
                    best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        selected_dirs.append(directions[best_idx])
        drone_counts[drone_ids[best_idx]] += 1
        remaining.discard(best_idx)

        if drone_ids[best_idx] == d4_id and d4_first_step is None:
            d4_first_step = step

    debug_info = {
        'quotas': dict(quotas),
        'final_counts': dict(drone_counts),
        'quota_blocks': quota_blocks,
        'd4_first_step': d4_first_step,
        'k_eff': k_eff,
    }

    return selected_indices, debug_info


# ---------------------------------------------------------------------------
# Balanced-quota selector (EXPERIMENTAL)
# ---------------------------------------------------------------------------


def select_best_views_balanced_quota(
    measurements: List[Measurement],
    confidences: List[float],
    drone_ids: Optional[List[int]] = None,
    alpha: float = DEFAULT_ALPHA,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
    quota_per_drone: Optional[dict] = None,
    k_eff: int = 12,
) -> Tuple[List[int], dict]:
    """EXPERIMENTAL: greedy selector with deliberate per-drone balance.

    Uses an explicit target quota per drone (default: equal share of k_eff).
    Blocks candidates from drones at quota while any drone is below.
    Selection stops at k_eff. No fallback to free selection after quotas met.
    """
    n = len(measurements)
    if n == 0:
        raise ValueError("No measurements provided.")
    if len(confidences) != n:
        raise ValueError(
            f"Number of confidences ({len(confidences)}) must match "
            f"number of measurements ({n})."
        )
    if drone_ids is None or len(drone_ids) != n:
        raise ValueError("drone_ids is required and must match measurement count.")

    from collections import Counter

    unique_drones = sorted(set(drone_ids))

    if quota_per_drone is None:
        base = k_eff // len(unique_drones)
        remainder = k_eff % len(unique_drones)
        quota_per_drone = {}
        for i, did in enumerate(unique_drones):
            quota_per_drone[did] = base + (1 if i < remainder else 0)

    directions = [m[1] for m in measurements]
    first_idx = max(range(n), key=lambda i: (confidences[i], -i))

    selected_indices: List[int] = [first_idx]
    selected_dirs: List[Tuple[float, float, float]] = [directions[first_idx]]
    remaining = set(range(n)) - {first_idx}

    drone_counts: Counter = Counter()
    drone_counts[drone_ids[first_idx]] += 1

    d4_id = max(unique_drones)
    d4_first_step = 1 if drone_ids[first_idx] == d4_id else None
    quota_blocks = Counter()

    while len(selected_indices) < k_eff and remaining:
        step = len(selected_indices) + 1
        any_below = any(
            drone_counts[did] < quota_per_drone.get(did, 0) for did in unique_drones
        )

        best_idx = -1
        best_score = -1.0

        for idx in sorted(remaining):
            did = drone_ids[idx]
            if any_below and drone_counts[did] >= quota_per_drone.get(did, 0):
                quota_blocks[did] += 1
                continue
            ang_div = compute_angular_diversity(directions[idx], selected_dirs)
            if ang_div < min_angle_deg:
                continue
            score = compute_selection_score(ang_div, confidences[idx], alpha)
            if score > best_score:
                best_score = score
                best_idx = idx

        # Fallback: if all valid candidates are from satisfied drones
        if best_idx == -1:
            for idx in sorted(remaining):
                ang_div = compute_angular_diversity(directions[idx], selected_dirs)
                if ang_div < min_angle_deg:
                    continue
                score = compute_selection_score(ang_div, confidences[idx], alpha)
                if score > best_score:
                    best_score = score
                    best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        selected_dirs.append(directions[best_idx])
        drone_counts[drone_ids[best_idx]] += 1
        remaining.discard(best_idx)

        if drone_ids[best_idx] == d4_id and d4_first_step is None:
            d4_first_step = step

    quota_met = all(
        drone_counts[did] >= quota_per_drone.get(did, 0) for did in unique_drones
    )

    debug_info = {
        'target_quota': dict(quota_per_drone),
        'final_counts': dict(drone_counts),
        'quota_met': quota_met,
        'quota_blocks': dict(quota_blocks),
        'total_blocks': sum(quota_blocks.values()),
        'd4_first_step': d4_first_step,
    }

    return selected_indices, debug_info


# ---------------------------------------------------------------------------
# Regime-switch selector (EXPERIMENTAL)
# ---------------------------------------------------------------------------


def select_best_views_switched(
    measurements: List[Measurement],
    confidences: List[float],
    k: int = DEFAULT_K,
    drone_ids: Optional[List[int]] = None,
    alpha: float = DEFAULT_ALPHA,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
    lambda_rep: float = 0.25,
    tau_avail: float = 0.65,
    tau_conf: float = 0.85,
) -> Tuple[List[int], dict]:
    """EXPERIMENTAL: regime-switch between best_views and simple_penalty.

    Detects whether the weakest drone is significantly under-available and
    lower-confidence. If so, switches to simple_penalty. Otherwise uses
    standard best_views.

    Switch fires if BOTH:
        availability_ratio_min < tau_avail
        conf_ratio_min < tau_conf
    """
    n = len(measurements)
    if n == 0:
        raise ValueError("No measurements provided.")
    if len(confidences) != n:
        raise ValueError(
            f"Number of confidences ({len(confidences)}) must match "
            f"number of measurements ({n})."
        )
    if drone_ids is None or len(drone_ids) != n:
        raise ValueError("drone_ids is required and must match measurement count.")

    from collections import defaultdict

    # Per-drone stats
    drone_meas_count: dict = {}
    drone_conf_sum: dict = {}
    for i, did in enumerate(drone_ids):
        drone_meas_count[did] = drone_meas_count.get(did, 0) + 1
        drone_conf_sum[did] = drone_conf_sum.get(did, 0.0) + confidences[i]

    drone_avail = {did: c / n for did, c in drone_meas_count.items()}
    drone_conf_mean = {did: drone_conf_sum[did] / drone_meas_count[did]
                       for did in drone_meas_count}

    max_avail = max(drone_avail.values())
    max_conf = max(drone_conf_mean.values())

    # Weakest drone by availability
    weakest_did = min(drone_avail, key=drone_avail.get)
    avail_ratio = drone_avail[weakest_did] / max_avail if max_avail > 0 else 1.0
    conf_ratio = drone_conf_mean[weakest_did] / max_conf if max_conf > 0 else 1.0

    avail_triggered = avail_ratio < tau_avail
    conf_triggered = conf_ratio < tau_conf
    use_penalty = avail_triggered and conf_triggered

    if use_penalty:
        indices = select_best_views_drone_penalty(
            measurements, confidences, k=k, drone_ids=drone_ids,
            lambda_rep=lambda_rep, min_angle_deg=min_angle_deg,
        )
        branch = "simple_penalty"
    else:
        indices = select_best_views(
            measurements, confidences, k=k, alpha=alpha,
            min_angle_deg=min_angle_deg,
        )
        branch = "best_views"

    debug_info = {
        'branch': branch,
        'weakest_drone': weakest_did,
        'avail_ratio': avail_ratio,
        'conf_ratio': conf_ratio,
        'tau_avail': tau_avail,
        'tau_conf': tau_conf,
        'avail_triggered': avail_triggered,
        'conf_triggered': conf_triggered,
        'drone_avail': dict(drone_avail),
        'drone_conf_mean': {did: round(v, 3) for did, v in drone_conf_mean.items()},
    }

    return indices, debug_info


def select_top_k_by_confidence(
    confidences: List[float],
    k: int = DEFAULT_K,
) -> List[int]:
    """Baseline: select k measurements with highest confidence.

    Ignores geometry entirely.

    Args:
        confidences: Per-measurement confidence scores.
        k: Maximum number to select.

    Returns:
        Indices sorted by descending confidence (ties: lowest index first).
    """
    # Sort by (-confidence, index) for stable descending order
    ranked = sorted(range(len(confidences)),
                    key=lambda i: (-confidences[i], i))
    return ranked[:k]


def select_top_k_by_geometry(
    measurements: List[Measurement],
    k: int = DEFAULT_K,
    min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
) -> List[int]:
    """Baseline: greedy selection maximising angular diversity only.

    Ignores confidence. Initialises with the pair of rays that have
    the largest angular separation, then greedily adds the candidate
    with the largest minimum angle to already-selected rays.

    Args:
        measurements: List of (origin, direction) tuples.
        k: Maximum number to select.
        min_angle_deg: Minimum angular diversity to accept a candidate.

    Returns:
        Indices in order of selection. Length <= k.
    """
    n = len(measurements)
    if n == 0:
        return []
    if n == 1:
        return [0]

    directions = [m[1] for m in measurements]

    # Find the most divergent pair to seed the selection
    best_i, best_j = 0, 1
    best_angle = -1.0
    for i in range(n):
        vi = np.asarray(directions[i], dtype=np.float64)
        ni = np.linalg.norm(vi)
        if ni < 1e-12:
            continue
        vi = vi / ni
        for j in range(i + 1, n):
            vj = np.asarray(directions[j], dtype=np.float64)
            nj = np.linalg.norm(vj)
            if nj < 1e-12:
                continue
            vj = vj / nj
            dot = float(np.clip(np.dot(vi, vj), -1.0, 1.0))
            angle = math.degrees(math.acos(abs(dot)))
            if angle > best_angle or (angle == best_angle and (i, j) < (best_i, best_j)):
                best_angle = angle
                best_i, best_j = i, j

    selected_indices = [best_i, best_j]
    selected_dirs = [directions[best_i], directions[best_j]]
    remaining = set(range(n)) - {best_i, best_j}

    while len(selected_indices) < k and remaining:
        best_idx = -1
        best_div = -1.0

        for idx in sorted(remaining):  # sorted for deterministic tie-breaking
            ang_div = compute_angular_diversity(directions[idx], selected_dirs)
            if ang_div < min_angle_deg:
                continue
            if ang_div > best_div:
                best_div = ang_div
                best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        selected_dirs.append(directions[best_idx])
        remaining.discard(best_idx)

    return selected_indices


def select_random(
    n_total: int,
    k: int = DEFAULT_K,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Baseline: random selection without replacement.

    Args:
        n_total: Total number of measurements available.
        k: Number to select (capped at n_total).
        rng: Numpy random generator for reproducibility.

    Returns:
        List of unique indices (unordered).
    """
    if rng is None:
        rng = np.random.default_rng()
    actual_k = min(k, n_total)
    return list(rng.choice(n_total, size=actual_k, replace=False))
