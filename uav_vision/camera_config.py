"""
Camera configuration dataclass and predefined camera presets.

Centralizes all camera intrinsic parameters so that switching cameras
only requires changing DEFAULT_CAMERA (or passing a different CameraConfig).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CameraConfig:
    """Intrinsic camera parameters for simulation and geolocation."""

    name: str
    focal_length_px: float
    image_width: int
    image_height: int
    fov_deg: float  # horizontal FOV in degrees

    @property
    def image_center(self) -> Tuple[float, float]:
        """Principal point assuming no lens distortion (width/2, height/2)."""
        return (self.image_width / 2.0, self.image_height / 2.0)

    @property
    def max_radius(self) -> float:
        """Half of image width — used to normalize pixel distance from center."""
        return self.image_width / 2.0


# ---------------------------------------------------------------------------
# Predefined camera presets
# ---------------------------------------------------------------------------

# Valores estimados a partir de specs publicadas de SIYI.
# Pendiente calibración con tablero de ajedrez para valores exactos.
# Sensor: Sony 1/1.7" (7.6 x 5.7 mm), FOV diagonal 93°.
# f_real = (9.41/2) / tan(46.5°) ≈ 4.47 mm
# f_px = 4.47 * 1920 / 7.6 ≈ 1130
# FOV horizontal = 2 * atan(7.6 / (2 * 4.47)) ≈ 80.8°
SIYI_A8_MINI = CameraConfig(
    name="SIYI A8 mini",
    focal_length_px=1130.0,
    image_width=1920,
    image_height=1080,
    fov_deg=80.8,
)

# Valores REALES, tomados de los run_info.json que el propio onboard.py
# escribio en los tres vuelos (26jul, 01ago, 02ago):
#     "focal_px": 1407.0, "principal_point": [945.7, 547.1]
# El punto principal calibrado NO es el centro geometrico (960, 540); ver
# README, seccion "punto principal".
# FOV horizontal = 2 * atan(1920 / (2 * 1407)) = 68.6 grados
ARDUCAM_MODULE_3 = CameraConfig(
    name="Arducam Module 3",
    focal_length_px=1407.0,
    image_width=1920,
    image_height=1080,
    fov_deg=68.6,
)

# Valores calibrados para simulación (specs oficiales Raspberry Pi).
# f_px = (3.04 mm * 640 px) / 3.68 mm = 529
RPI_CAMERA_V2 = CameraConfig(
    name="Raspberry Pi Camera v2",
    focal_length_px=529.0,
    image_width=640,
    image_height=480,
    fov_deg=62.2,
)

DEFAULT_CAMERA = SIYI_A8_MINI
