from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceRegion:
    face_bgr: np.ndarray
    bbox: tuple[int, int, int, int]


def _clip_bbox(x: int, y: int, w: int, h: int, max_w: int, max_h: int) -> tuple[int, int, int, int]:
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, max_w - x))
    h = max(1, min(h, max_h - y))
    return x, y, w, h


def detect_face(image_bgr: np.ndarray) -> FaceRegion:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(100, 100))

    height, width = image_bgr.shape[:2]
    if len(faces) == 0:
        # Fallback: assume the face is centered and includes most of the frame.
        margin_x = int(width * 0.08)
        margin_y = int(height * 0.06)
        x, y = margin_x, margin_y
        w = width - 2 * margin_x
        h = height - 2 * margin_y
        return FaceRegion(face_bgr=image_bgr[y : y + h, x : x + w], bbox=(x, y, w, h))

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    # Expand coverage to ensure forehead/cheeks/chin are included
    padding = int(0.16 * max(w, h))
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # If detected face is too small, widen to around half of the image dimensions
    min_detect_w = int(width * 0.5)
    min_detect_h = int(height * 0.5)
    if w < min_detect_w:
        diff_w = (min_detect_w - w) // 2
        x -= diff_w
        w = min_detect_w
    if h < min_detect_h:
        diff_h = (min_detect_h - h) // 2
        y -= diff_h
        h = min_detect_h

    x, y, w, h = _clip_bbox(x, y, w, h, width, height)

    # If expansion still misses significant image portion, use center crop fallback
    if w < width * 0.6 or h < height * 0.6:
        margin_x = int(width * 0.08)
        margin_y = int(height * 0.06)
        x, y = margin_x, margin_y
        w = width - 2 * margin_x
        h = height - 2 * margin_y

    return FaceRegion(face_bgr=image_bgr[y : y + h, x : x + w], bbox=(x, y, w, h))
