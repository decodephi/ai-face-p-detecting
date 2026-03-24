from __future__ import annotations

import cv2
import numpy as np


LABEL_COLORS = {
    "pimple": (255, 64, 64),
    "whitehead": (255, 184, 77),
    "blackhead": (80, 180, 255),
}


def annotate_face(
    face_rgb: np.ndarray,
    detections: list[dict],
    dark_spot_mask: np.ndarray,
    oiliness_mask: np.ndarray,
) -> np.ndarray:
    annotated = face_rgb.copy()
    annotated = cv2.detailEnhance(annotated, sigma_s=8, sigma_r=0.18)

    dark_contours, _ = cv2.findContours(dark_spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_dark = [contour for contour in dark_contours if cv2.contourArea(contour) >= 8]
    cv2.drawContours(annotated, filtered_dark, -1, (183, 88, 88), 1)

    heatmap = cv2.applyColorMap(oiliness_mask, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    oil_alpha = (oiliness_mask.astype(np.float32) / 255.0)[:, :, None] * 0.14
    annotated = (annotated * (1.0 - oil_alpha) + heatmap * oil_alpha).astype(np.uint8)

    for detection in detections:
        box = detection["bbox"]
        color = LABEL_COLORS[detection["label"]]
        x, y = box["x"], box["y"]
        w, h = box["width"], box["height"]
        center_x = x + (w // 2)
        center_y = y + (h // 2)
        radius = max(4, min(10, max(w, h) // 2 + 2))
        cv2.circle(annotated, (center_x, center_y), radius, color, 2)
        cv2.circle(annotated, (center_x, center_y), 2, color, -1)

    return annotated
