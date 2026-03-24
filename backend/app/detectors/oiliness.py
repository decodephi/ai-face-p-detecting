from __future__ import annotations

import cv2
import numpy as np


def detect_oiliness(face_rgb: np.ndarray, skin_mask: np.ndarray) -> dict[str, np.ndarray | float | str]:
    hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
    value_channel = hsv[:, :, 2]
    saturation_channel = hsv[:, :, 1]
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
    local_contrast = cv2.absdiff(gray, blurred)

    brightness = value_channel.astype(np.float32) / 255.0
    saturation = saturation_channel.astype(np.float32) / 255.0
    texture_penalty = cv2.normalize(local_contrast.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    specular_weight = 1.0 - np.clip(saturation * 1.1, 0.0, 0.85)
    shine_map = brightness * (1.0 - texture_penalty) * specular_weight
    shine_map *= (skin_mask > 0).astype(np.float32)

    height, width = shine_map.shape
    t_zone_mask = np.zeros_like(shine_map, dtype=np.float32)
    cv2.ellipse(
        t_zone_mask,
        (width // 2, int(height * 0.46)),
        (int(width * 0.18), int(height * 0.32)),
        0,
        0,
        360,
        1.0,
        thickness=-1,
    )
    shine_map *= (0.55 + (0.45 * t_zone_mask))

    if np.count_nonzero(skin_mask):
        skin_values = shine_map[skin_mask > 0]
        hotspot_threshold = float(np.percentile(skin_values, 92))
    else:
        hotspot_threshold = 0.72

    oily_mask = np.where(shine_map >= hotspot_threshold, shine_map, 0.0)
    oily_mask = np.uint8(np.clip(oily_mask * 255.0, 0, 255))
    oily_mask = cv2.GaussianBlur(oily_mask, (11, 11), 0)

    oily_pixels = cv2.countNonZero(cv2.inRange(oily_mask, 135, 255))
    skin_pixels = max(cv2.countNonZero(skin_mask), 1)
    mean_shine = float(np.mean(shine_map[skin_mask > 0])) if np.count_nonzero(skin_mask) else 0.0
    hotspot_ratio = oily_pixels / skin_pixels
    score = round(min(1.0, (mean_shine * 0.48) + (hotspot_ratio * 1.1)), 4)

    if score >= 0.42:
        skin_type = "Oily"
    elif score >= 0.2:
        skin_type = "Normal"
    else:
        skin_type = "Dry"

    return {"mask": oily_mask, "score": score, "type": skin_type}
