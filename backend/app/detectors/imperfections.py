from __future__ import annotations

import cv2
import numpy as np


def _iou(box_a: dict, box_b: dict) -> float:
    ax1, ay1 = box_a["x"], box_a["y"]
    ax2, ay2 = ax1 + box_a["width"], ay1 + box_a["height"]
    bx1, by1 = box_b["x"], box_b["y"]
    bx2, by2 = bx1 + box_b["width"], by1 + box_b["height"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = box_a["width"] * box_a["height"]
    area_b = box_b["width"] * box_b["height"]
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _deduplicate(detections: list[dict]) -> list[dict]:
    deduplicated: list[dict] = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if any(_iou(detection["bbox"], kept["bbox"]) > 0.35 for kept in deduplicated):
            continue
        deduplicated.append(detection)
    return deduplicated


def detect_imperfections(face_rgb: np.ndarray, skin_mask: np.ndarray, max_detections: int) -> list[dict]:
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]

    dog_small = cv2.GaussianBlur(gray, (0, 0), 1.2)
    dog_large = cv2.GaussianBlur(gray, (0, 0), 3.8)
    detail = cv2.absdiff(dog_small, dog_large)
    detail = cv2.bitwise_and(detail, detail, mask=skin_mask)
    adaptive_threshold = (
        max(6, int(np.percentile(detail[skin_mask > 0], 76)))
        if np.count_nonzero(skin_mask)
        else 8
    )
    _, thresholded = cv2.threshold(detail, adaptive_threshold, 255, cv2.THRESH_BINARY)
    thresholded = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    thresholded = cv2.dilate(thresholded, np.ones((2, 2), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[dict] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 4 or area > 220:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w > 28 or h > 28:
            continue

        roi_gray = gray[y : y + h, x : x + w]
        roi_mask = skin_mask[y : y + h, x : x + w]
        roi_light = lightness[y : y + h, x : x + w]
        if roi_gray.size == 0 or cv2.countNonZero(roi_mask) == 0:
            continue

        mask_coverage = cv2.countNonZero(roi_mask) / roi_mask.size
        if mask_coverage < 0.5:
            continue

        mean_intensity = float(cv2.mean(roi_gray, mask=roi_mask)[0])
        mean_lightness = float(cv2.mean(roi_light, mask=roi_mask)[0])
        circularity = 0.0
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = float(4 * np.pi * area / (perimeter * perimeter))
        if circularity < 0.08:
            continue

        confidence = min(
            0.98,
            0.38
            + (area / 210.0)
            + (circularity * 0.20)
            + (0.12 if mean_lightness < 115 and mean_lightness > 70 else 0.0),
        )

        if mean_lightness < 96:
            label = "blackhead"
        elif mean_lightness > 162:
            label = "whitehead"
        else:
            label = "pimple"

        detections.append(
            {
                "label": label,
                "confidence": round(confidence, 2),
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "area": int(area),
            }
        )

    detections = _deduplicate(detections)
    detections.sort(key=lambda item: (item["confidence"], item["area"]), reverse=True)
    return detections[:max_detections]
