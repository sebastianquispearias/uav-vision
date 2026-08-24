"""
Pinhole camera model for local cartesian coordinates.

Provides projection (3D->pixel) and back-projection (pixel->ray) for
simulation in GrADyS-SIM, where all positions are (x, y, z) in meters.

Coordinate convention (consistent with NED/aviation):
    - World frame: x = East, y = North, z = Up
    - Yaw: 0° = North (+Y), 90° = East (+X), clockwise
    - Pitch: -90° = looking straight down (camera optical axis -> -Z)
    - Image frame: u = right in image, v = down in image

No ECEF, no WGS84. Pure functions, no simulator dependencies.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from uav_vision.camera_config import DEFAULT_CAMERA

# Type alias (same as fusion.py)
Measurement = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


def _world_to_camera_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Build rotation matrix from world frame to camera frame.

    Camera frame convention: X_cam = right, Y_cam = down, Z_cam = forward
    (optical axis). This is the standard pinhole convention.

    Construction from basis vectors:
        heading = (sin(yaw), cos(yaw), 0)   — direction drone faces
        right   = (cos(yaw), -sin(yaw), 0)  — 90° clockwise from heading
        cam_z   = heading * cos(pitch) + up * sin(pitch)  — optical axis
                  (pitch=-90° -> cam_z = (0,0,-1) = straight down)
        cam_x   = right  — always horizontal
        cam_y   = cam_z x cam_x  — completes right-hand frame

    The rows of the returned matrix are the camera axes in world coords.

    Returns:
        3x3 rotation matrix R such that p_cam = R @ (p_world - drone_pos).
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    sy, cy = math.sin(y), math.cos(y)
    cp, sp = math.cos(p), math.sin(p)

    heading = np.array([sy, cy, 0.0])
    right = np.array([cy, -sy, 0.0])
    up = np.array([0.0, 0.0, 1.0])

    cam_z = heading * cp + up * sp  # optical axis
    cam_x = right
    cam_y = np.cross(cam_z, cam_x)  # down in image

    return np.array([cam_x, cam_y, cam_z])


def project_to_pixel(
    drone_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    yaw_deg: float,
    pitch_deg: float = -90.0,
    focal_px: float = DEFAULT_CAMERA.focal_length_px,
    img_w: int = DEFAULT_CAMERA.image_width,
    img_h: int = DEFAULT_CAMERA.image_height,
) -> Optional[Tuple[float, float]]:
    """Project a 3D target position to pixel coordinates.

    Returns None if the target is behind the camera or outside the image
    frame. No clamping to image edges is performed.

    Args:
        drone_pos: Camera position (x, y, z) in meters.
        target_pos: Target position (x, y, z) in meters.
        yaw_deg: Camera yaw in degrees (0°=North, 90°=East, clockwise).
        pitch_deg: Camera pitch in degrees (-90° = looking straight down).
        focal_px: Focal length in pixels.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Pixel coordinates (px, py) if target is visible, None otherwise.
    """
    d = np.asarray(target_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)

    R = _world_to_camera_rotation(yaw_deg, pitch_deg)
    d_cam = R @ d

    # Camera frame: X=right, Y=down, Z=forward (optical axis)
    X, Y, Z = d_cam

    if Z <= 0:
        return None

    cx, cy = img_w / 2.0, img_h / 2.0
    px = focal_px * (X / Z) + cx
    py = focal_px * (Y / Z) + cy

    if px < 0 or px >= img_w or py < 0 or py >= img_h:
        return None

    return (float(px), float(py))


def pixel_to_ray(
    drone_pos: Tuple[float, float, float],
    yaw_deg: float,
    pixel_xy: Tuple[float, float],
    pitch_deg: float = -90.0,
    focal_px: float = DEFAULT_CAMERA.focal_length_px,
    img_w: int = DEFAULT_CAMERA.image_width,
    img_h: int = DEFAULT_CAMERA.image_height,
) -> Measurement:
    """Back-project a pixel to a 3D ray (measurement for fusion).

    Inverse of project_to_pixel: converts a pixel coordinate back to
    a unit direction vector in world frame.

    Args:
        drone_pos: Camera position (x, y, z) in meters.
        yaw_deg: Camera yaw in degrees (0°=North, 90°=East, clockwise).
        pixel_xy: Pixel position (px, py).
        pitch_deg: Camera pitch in degrees (-90° = looking straight down).
        focal_px: Focal length in pixels.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        Measurement tuple ((ox, oy, oz), (dx, dy, dz)) where origin is
        the drone position and direction is a unit vector in world frame.
    """
    px, py = pixel_xy
    cx, cy = img_w / 2.0, img_h / 2.0

    # Pixel to camera-frame direction (X=right, Y=down, Z=forward)
    d_cam = np.array([
        (px - cx) / focal_px,
        (py - cy) / focal_px,
        1.0,
    ])

    # Rotate from camera frame to world frame (R^T = R^{-1} for orthogonal R)
    R = _world_to_camera_rotation(yaw_deg, pitch_deg)
    d_world = R.T @ d_cam

    d_world = d_world / np.linalg.norm(d_world)

    origin = (float(drone_pos[0]), float(drone_pos[1]), float(drone_pos[2]))
    direction = (float(d_world[0]), float(d_world[1]), float(d_world[2]))
    return (origin, direction)
