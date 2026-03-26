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

    hsv_mask = cv2.inRange(hsv, np.array([0, 15, 25]), np.array([50, 230, 255]))
    ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 133, 70]), np.array([255, 180, 135]))
    skin_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    # Apply an adaptive histogram equalization on value channel to improve contrast for fair/dark tones
    v = hsv[:, :, 2]
    equalized_v = cv2.equalizeHist(v)
    ycrcb[:, :, 0] = equalized_v

    # Engage morphological operations to reduce pixel noise and fill small skin gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    height, width = skin_mask.shape
    exclusion_mask = np.full_like(skin_mask, 255, dtype=np.uint8)

    eye_band_top = int(height * 0.15)
    eye_band_bottom = int(height * 0.45)
    lip_band_top = int(height * 0.65)
    lip_band_bottom = int(height * 0.92)

    cv2.rectangle(exclusion_mask, (0, eye_band_top), (width, eye_band_bottom), 150, -1)
    cv2.rectangle(
        exclusion_mask,
        (int(width * 0.14), lip_band_top),
        (int(width * 0.86), lip_band_bottom),
        130,
        -1,
    )

    masked = cv2.bitwise_and(skin_mask, exclusion_mask)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    refined = cv2.morphologyEx(masked, cv2.MORPH_OPEN, kernel2, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel2, iterations=2)
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
