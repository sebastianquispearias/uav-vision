"""
Camera configuration dataclass and predefined camera presets.

Centralizes all camera intrinsic parameters so that switching cameras
only requires changing DEFAULT_CAMERA (or passing a different CameraConfig).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class CameraConfig:
    """Intrinsic camera parameters for simulation and geolocation."""

    name: str
    focal_length_px: float
    image_width: int
    image_height: int
    fov_deg: float  # horizontal FOV in degrees
    calibrated_principal_point: Optional[Tuple[float, float]] = None

    @property
    def image_center(self) -> Tuple[float, float]:
        """
        The geometric centre of the image, in pixels. It is always
        (width/2, height/2) and does not depend on lens calibration.
        """
        return (self.image_width / 2.0, self.image_height / 2.0)

    @property
    def principal_point(self) -> Tuple[float, float]:
        """
        The point where the optical axis meets the sensor, in pixels. On a
        perfectly aligned lens it coincides with the geometric image centre,
        so that is the value returned when no calibration is available.
        A calibrated value shifts every back-projected ray: an offset of
        14 px at a focal length of 1407 px tilts the ray by 0.65 degrees,
        which displaces the ground intersection by roughly 0.4 m when
        flying at 35 m.
        """
        if self.calibrated_principal_point is None:
            return self.image_center
        return self.calibrated_principal_point

    @property
    def max_radius(self) -> float:
        """Half of image width — used to normalize pixel distance from center."""
        return self.image_width / 2.0

    def rotated_180(self) -> "CameraConfig":
        """
        Returns the configuration for the same camera mounted upside-down.

        When the ISP un-flips the captured image (hflip + vflip), every pixel coordinate c maps
        to (size - 1) - c, and the calibrated principal point must follow. Skipping the
        reflection tilts every back-projected ray by the angular equivalent of the principal
        point offset (about a degree for this camera, roughly a meter on the ground at mission
        altitude).
        """
        px, py = self.principal_point
        return replace(
            self,
            name=f"{self.name} (rot180)",
            calibrated_principal_point=(
                self.image_width - 1 - px,
                self.image_height - 1 - py,
            ),
        )


# ---------------------------------------------------------------------------
# Predefined camera presets
# ---------------------------------------------------------------------------

# Estimated from the published SIYI specifications (sensor Sony 1/1.7", 7.6 x 5.7 mm,
# diagonal FOV 93 degrees):
#   f_real = (9.41 / 2) / tan(46.5 deg) ~= 4.47 mm
#   f_px = 4.47 * 1920 / 7.6 ~= 1130
#   horizontal FOV = 2 * atan(7.6 / (2 * 4.47)) ~= 80.8 deg
SIYI_A8_MINI = CameraConfig(
    name="SIYI A8 mini",
    focal_length_px=1130.0,
    image_width=1920,
    image_height=1080,
    fov_deg=80.8,
)

# Intrinsics recovered from the logs of the real flights (see NOTES.md). The calibrated
# principal point is 14 px away from the geometric centre.
# Horizontal FOV = 2 * atan(1920 / (2 * 1407)) = 68.6 degrees.
ARDUCAM_MODULE_3 = CameraConfig(
    name="Arducam Module 3",
    focal_length_px=1407.0,
    image_width=1920,
    image_height=1080,
    fov_deg=68.6,
    calibrated_principal_point=(945.7, 547.1),
)

# From the official Raspberry Pi specifications: f_px = (3.04 mm * 640 px) / 3.68 mm = 529.
RPI_CAMERA_V2 = CameraConfig(
    name="Raspberry Pi Camera v2",
    focal_length_px=529.0,
    image_width=640,
    image_height=480,
    fov_deg=62.2,
)

DEFAULT_CAMERA = SIYI_A8_MINI
