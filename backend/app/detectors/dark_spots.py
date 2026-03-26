from __future__ import annotations

import cv2
import numpy as np


def detect_dark_spots(face_rgb: np.ndarray, skin_mask: np.ndarray) -> dict[str, np.ndarray | float | int]:
    lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]

    # Reduce noise while preserving edges for a finer dark spot baseline
    smooth_lightness = cv2.bilateralFilter(lightness, 9, 75, 75)
    baseline = cv2.GaussianBlur(smooth_lightness, (31, 31), 0)
    difference = cv2.subtract(baseline, smooth_lightness)

    if np.count_nonzero(skin_mask):
        mean_diff = np.mean(difference[skin_mask > 0])
        std_diff = np.std(difference[skin_mask > 0])
        dynamic_threshold = int(mean_diff + std_diff * 0.62)
        dynamic_threshold = np.clip(dynamic_threshold, 6, 34)
    else:
        dynamic_threshold = 12

    _, dark_mask = cv2.threshold(difference, dynamic_threshold, 255, cv2.THRESH_BINARY)
    dark_mask = cv2.bitwise_and(dark_mask, skin_mask)
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    dark_mask = cv2.medianBlur(dark_mask, 5)

    filtered_mask = np.zeros_like(dark_mask)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 8 <= area <= 450:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=-1)

    dark_pixels = cv2.countNonZero(filtered_mask)
    skin_pixels = max(cv2.countNonZero(skin_mask), 1)
    return {
        "mask": filtered_mask,
        "area_ratio": round(dark_pixels / skin_pixels, 4),
        "pixel_count": int(dark_pixels),
    }
