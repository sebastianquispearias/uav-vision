"""
Sensor noise models for UAV geolocation simulation.

All functions are pure (no side effects, no simulator dependencies).
Coordinates are local cartesian (x, y, z) in meters.
Yaw is in degrees. Pixel coordinates are in pixels.

Reproducibility: pass a numpy Generator via the `rng` parameter.
"""

from typing import Optional, Tuple

import numpy as np


def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Return the provided RNG or create a new unseeded one."""
    return rng if rng is not None else np.random.default_rng()


def add_gps_noise(
    position_xyz: Tuple[float, float, float],
    sigma_m: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Add Gaussian noise to a 3D position (x, y, z) in meters.

    Noise is applied only to x and y (horizontal plane). The z component
    is returned unchanged because altitude is assumed known from the
    barometric altimeter / controlled in simulation.

    Args:
        position_xyz: Position as (x, y, z) in meters.
        sigma_m: Standard deviation of GPS noise in meters (per axis).
            If 0, the position is returned unchanged.
        rng: Numpy random generator for reproducibility.

    Returns:
        Noisy position as (x, y, z).
    """
    if sigma_m == 0:
        return position_xyz
    rng = _get_rng(rng)
    x, y, z = position_xyz
    return (
        x + rng.normal(0, sigma_m),
        y + rng.normal(0, sigma_m),
        z,
    )


def add_yaw_noise(
    yaw_deg: float,
    sigma_deg: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Add Gaussian noise to a yaw angle in degrees.

    The result is wrapped to [0, 360).

    Args:
        yaw_deg: Yaw angle in degrees.
        sigma_deg: Standard deviation of yaw noise in degrees.
            If 0, the yaw is returned unchanged.
        rng: Numpy random generator for reproducibility.

    Returns:
        Noisy yaw in degrees, wrapped to [0, 360).
    """
    if sigma_deg == 0:
        return yaw_deg
    rng = _get_rng(rng)
    noisy = yaw_deg + rng.normal(0, sigma_deg)
    return noisy % 360


def add_pixel_noise(
    pixel_xy: Tuple[float, float],
    sigma_px: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Add Gaussian noise to a pixel coordinate.

    No clamping to image bounds is applied — the caller is responsible
    for that if needed.

    Args:
        pixel_xy: Pixel position as (x, y) in pixels.
        sigma_px: Standard deviation of pixel noise in pixels.
            If 0, the pixel is returned unchanged.
        rng: Numpy random generator for reproducibility.

    Returns:
        Noisy pixel as (x, y).
    """
    if sigma_px == 0:
        return pixel_xy
    rng = _get_rng(rng)
    px, py = pixel_xy
    return (
        px + rng.normal(0, sigma_px),
        py + rng.normal(0, sigma_px),
    )
