from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SkinSegmentationResult:
    skin_mask: np.ndarray
    refined_mask: np.ndarray
    skin_only_bgr: np.ndarray


def segment_skin(face_bgr: np.ndarray) -> SkinSegmentationResult:
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)

    hsv_mask = cv2.inRange(hsv, np.array([0, 18, 35]), np.array([28, 220, 255]))
    ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    skin_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    height, width = skin_mask.shape
    exclusion_mask = np.ones_like(skin_mask, dtype=np.uint8) * 255

    eye_band_top = int(height * 0.18)
    eye_band_bottom = int(height * 0.42)
    lip_band_top = int(height * 0.68)
    lip_band_bottom = int(height * 0.92)

    cv2.rectangle(exclusion_mask, (0, eye_band_top), (width, eye_band_bottom), 200, -1)
    cv2.rectangle(
        exclusion_mask,
        (int(width * 0.18), lip_band_top),
        (int(width * 0.82), lip_band_bottom),
        180,
        -1,
    )

    masked = cv2.bitwise_and(skin_mask, exclusion_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    refined = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
    refined = cv2.medianBlur(refined, 5)

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        largest_mask = np.zeros_like(refined)
        cv2.drawContours(largest_mask, [largest], -1, 255, thickness=-1)
        refined = cv2.bitwise_and(refined, largest_mask)

    skin_only = cv2.bitwise_and(face_bgr, face_bgr, mask=refined)

    return SkinSegmentationResult(
        skin_mask=skin_mask,
        refined_mask=refined,
        skin_only_bgr=skin_only,
    )
