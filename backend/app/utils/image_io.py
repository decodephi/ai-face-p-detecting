from __future__ import annotations

import base64
from typing import Final

import cv2
import numpy as np


BASE64_PREFIX: Final[str] = "base64,"


def decode_upload_bytes(file_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_base64_image(encoded: str) -> np.ndarray:
    payload = encoded.split(BASE64_PREFIX, maxsplit=1)[-1] if BASE64_PREFIX in encoded else encoded
    image_bytes = base64.b64decode(payload)
    return decode_upload_bytes(image_bytes)


def encode_image_to_base64(image_rgb: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("Unable to encode annotated image.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")
