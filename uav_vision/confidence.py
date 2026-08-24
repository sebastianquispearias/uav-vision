"""
Simulated detection confidence model and derived quantities.

Models how YOLO-like detector confidence varies with the position of an
object in the image (higher near the center, lower near edges) and maps
confidence to pixel noise and gradient-descent weights.

All functions are pure (no side effects, no simulator dependencies).
"""

import math
from typing import Optional, Tuple

import numpy as np

from uav_vision.camera_config import DEFAULT_CAMERA

C_NOISE: float = 5.0  # hyperparameter: sigma_pixel = C_NOISE / confidence
C_VISDRONE_1D: float = 1.13  # empirical fit from VisDrone2019-DET-val + YOLOv8s
MIN_CONFIDENCE: float = 0.3
MAX_CONFIDENCE: float = 1.0


def simulate_confidence(
    pixel_xy: Tuple[float, float],
    image_center: Tuple[float, float] = DEFAULT_CAMERA.image_center,
    max_radius: float = DEFAULT_CAMERA.max_radius,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Simulate a detection confidence score based on pixel position.

    Model:
        conf = 0.95 - 0.4 * (dist_center / max_radius) + N(0, 0.05)
        clamped to [MIN_CONFIDENCE, MAX_CONFIDENCE]

    Objects near the image center get higher confidence; objects near
    the edges get lower confidence. Gaussian jitter (sigma=0.05)
    simulates stochastic detector behavior.

    Args:
        pixel_xy: Detection pixel position (x, y).
        image_center: Image center in pixels.
        max_radius: Normalization radius in pixels.
        rng: Numpy random generator for reproducibility.

    Returns:
        Confidence score in [MIN_CONFIDENCE, MAX_CONFIDENCE].
        Values that would fall below MIN_CONFIDENCE are clamped to it
        (representing the threshold below which detections are discarded).
    """
    rng = rng if rng is not None else np.random.default_rng()
    cx, cy = image_center
    px, py = pixel_xy
    dist_center = math.hypot(px - cx, py - cy)
    conf = 0.95 - 0.4 * (dist_center / max_radius) + rng.normal(0, 0.05)
    return float(max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, conf)))


def confidence_to_pixel_sigma(
    confidence: float,
    c: float = C_NOISE,
) -> float:
    """Map detection confidence to pixel-coordinate noise (sigma).

    Model: sigma_pixel = c / confidence

    Higher confidence -> smaller pixel noise (tighter bounding box).

    Args:
        confidence: Detection confidence in [MIN_CONFIDENCE, 1.0].
        c: Noise hyperparameter (default 5.0).

    Returns:
        Standard deviation of pixel noise in pixels.

    Raises:
        ValueError: If confidence < MIN_CONFIDENCE.
    """
    if confidence < MIN_CONFIDENCE:
        raise ValueError(
            f"Confidence {confidence:.3f} is below minimum {MIN_CONFIDENCE}. "
            "Detections below this threshold should be discarded."
        )
    return c / confidence


def confidence_to_weight(
    confidence: float,
    focal_length_px: float = DEFAULT_CAMERA.focal_length_px,
    c: float = C_NOISE,
) -> float:
    """Map detection confidence to a weight for weighted gradient descent.

    Derivation:
        sigma_pixel  = c / confidence
        sigma_bearing = sigma_pixel / focal_length_px
        weight = 1 / sigma_bearing^2
               = (focal_length_px * confidence)^2 / c^2

    Higher confidence -> larger weight (measurement is more trusted).

    Args:
        confidence: Detection confidence in [MIN_CONFIDENCE, 1.0].
        focal_length_px: Camera focal length in pixels.
        c: Noise hyperparameter (default 5.0).

    Returns:
        Non-negative weight for use in weighted least squares / gradient descent.

    Raises:
        ValueError: If confidence < MIN_CONFIDENCE.
    """
    if confidence < MIN_CONFIDENCE:
        raise ValueError(
            f"Confidence {confidence:.3f} is below minimum {MIN_CONFIDENCE}. "
            "Detections below this threshold should be discarded."
        )
    return (focal_length_px * confidence) ** 2 / c ** 2


_NOISE_MODELS = {
    "heuristic": C_NOISE,
    "visdrone_1d": C_VISDRONE_1D,
}


def confidence_to_pixel_sigma_model(
    confidence: float,
    model_name: str = "heuristic",
) -> float:
    """Map confidence to pixel sigma using a named noise model.

    Supported models:
        "heuristic"   — sigma = 5.0 / confidence  (original)
        "visdrone_1d" — sigma = 1.13 / confidence  (empirical VisDrone fit)
    """
    c = _NOISE_MODELS.get(model_name)
    if c is None:
        raise ValueError(
            f"Unknown noise model '{model_name}'. "
            f"Available: {list(_NOISE_MODELS.keys())}"
        )
    return confidence_to_pixel_sigma(confidence, c=c)
