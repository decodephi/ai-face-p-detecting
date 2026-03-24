from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceRegion:
    face_bgr: np.ndarray
    bbox: tuple[int, int, int, int]


def detect_face(image_bgr: np.ndarray) -> FaceRegion:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))

    if len(faces) == 0:
        height, width = image_bgr.shape[:2]
        margin_x = int(width * 0.1)
        margin_y = int(height * 0.08)
        x, y = margin_x, margin_y
        w = width - 2 * margin_x
        h = height - 2 * margin_y
        return FaceRegion(face_bgr=image_bgr[y : y + h, x : x + w], bbox=(x, y, w, h))

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    padding = int(0.08 * max(w, h))
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image_bgr.shape[1] - x, w + 2 * padding)
    h = min(image_bgr.shape[0] - y, h + 2 * padding)
    return FaceRegion(face_bgr=image_bgr[y : y + h, x : x + w], bbox=(x, y, w, h))
